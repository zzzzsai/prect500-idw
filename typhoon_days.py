from __future__ import annotations

import csv
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

from data_eda import DEFAULT_STATION_OBS_CSV

ZHEJIANG_BOX = (118.0, 123.19, 27.05, 31.25)
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"


def _parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_date(iso_time: str) -> str | None:
    dt = _parse_datetime(iso_time)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d")


def _parse_datetime(iso_time: str) -> datetime | None:
    iso_time = (iso_time or "").strip()
    if not iso_time:
        return None

    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(iso_time, fmt)
        except ValueError:
            continue
    return None


def _point_in_box(lat: float | None, lon: float | None, box: tuple[float, float, float, float]) -> bool:
    lon_min, lon_max, lat_min, lat_max = box
    return (
        lat is not None
        and lon is not None
        and lon_min <= lon <= lon_max
        and lat_min <= lat <= lat_max
    )


def _read_ibtracs_csv(input_csv: str | Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    input_path = Path(input_csv)

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        units_row = next(reader, [])
        rows: list[dict[str, str]] = []

        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue

            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]

            rows.append(dict(zip(header, row)))

    return header, units_row, rows


def read_unique_station_locations(
    obs_csv: str | Path = DEFAULT_STATION_OBS_CSV,
    station_col: str = "stationId",
    lat_col: str = "lat",
    lon_col: str = "lng",
    chunksize: int = 300_000,
):
    """
    从观测文件中提取唯一站点坐标，用于叠加绘图。
    """
    import pandas as pd

    station_frames = []
    usecols = [station_col, lat_col, lon_col]
    for chunk in pd.read_csv(obs_csv, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk.copy()
        chunk[lat_col] = pd.to_numeric(chunk[lat_col], errors="coerce")
        chunk[lon_col] = pd.to_numeric(chunk[lon_col], errors="coerce")
        chunk = chunk.dropna(subset=[station_col, lat_col, lon_col])
        chunk = chunk.drop_duplicates(subset=[station_col], keep="first")
        if not chunk.empty:
            station_frames.append(chunk[usecols])

    if not station_frames:
        raise ValueError(f"观测文件中没有可用站点坐标: {obs_csv}")

    stations = pd.concat(station_frames, ignore_index=True)
    stations = stations.drop_duplicates(subset=[station_col], keep="first").reset_index(drop=True)
    return stations


def filter_typhoon_tracks_in_zhejiang(
    input_csv: str | Path,
    output_csv: str | Path,
    box: tuple[float, float, float, float] = ZHEJIANG_BOX,
    t1_hours: float = 0,
    t2_hours: float = 0,
    start_month: int | None = None,
    end_month: int | None = None,
) -> dict:
    """
    按台风中心点是否进入浙江矩形框筛选路径记录。

    对每个台风（SID）：
    1. 找到首次进入矩形框的记录；
    2. 找到最后一次仍位于矩形框内的记录；
    3. 可选向前扩展 t1_hours、向后扩展 t2_hours；
    4. 可选仅保留 start_month-end_month 月份范围内的路径点；
    5. 输出新的 CSV，并返回影响日期与台风摘要。
    """
    header, units_row, rows = _read_ibtracs_csv(input_csv)

    storms: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        sid = (row.get("SID") or "").strip()
        if not sid:
            continue
        storms.setdefault(sid, []).append(row)

    kept_rows: list[dict[str, str]] = []
    affected_dates: set[str] = set()
    storm_summaries: list[dict[str, str]] = []

    for sid, storm_rows in storms.items():
        inside_indices: list[int] = []

        for idx, row in enumerate(storm_rows):
            lat = _parse_float(row.get("LAT", ""))
            lon = _parse_float(row.get("LON", ""))
            if _point_in_box(lat, lon, box):
                inside_indices.append(idx)

        if not inside_indices:
            continue

        first_inside_row = storm_rows[inside_indices[0]]
        last_inside_row = storm_rows[inside_indices[-1]]
        first_inside_dt = _parse_datetime(first_inside_row.get("ISO_TIME", ""))
        last_inside_dt = _parse_datetime(last_inside_row.get("ISO_TIME", ""))

        if first_inside_dt is None or last_inside_dt is None:
            start_idx = inside_indices[0]
            end_idx = inside_indices[-1]
            segment = storm_rows[start_idx : end_idx + 1]
        else:
            window_start = first_inside_dt - timedelta(hours=t1_hours)
            window_end = last_inside_dt + timedelta(hours=t2_hours)
            segment = []
            for row in storm_rows:
                row_dt = _parse_datetime(row.get("ISO_TIME", ""))
                if row_dt is None:
                    continue
                if window_start <= row_dt <= window_end:
                    segment.append(row)

        if start_month is not None and end_month is not None:
            filtered_segment: list[dict[str, str]] = []
            for row in segment:
                row_dt = _parse_datetime(row.get("ISO_TIME", ""))
                if row_dt is None:
                    continue
                if start_month <= row_dt.month <= end_month:
                    filtered_segment.append(row)
            segment = filtered_segment

        if not segment:
            continue

        kept_rows.extend(segment)

        name = next((r.get("NAME", "").strip() for r in segment if r.get("NAME", "").strip()), sid)
        start_time = (segment[0].get("ISO_TIME") or "").strip()
        end_time = (segment[-1].get("ISO_TIME") or "").strip()

        for row in segment:
            day = _parse_date(row.get("ISO_TIME", ""))
            if day:
                affected_dates.add(day)

        storm_summaries.append(
            {
                "SID": sid,
                "NAME": name,
                "start_time": start_time,
                "end_time": end_time,
                "record_count": str(len(segment)),
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(units_row) < len(header):
        units_row = units_row + [""] * (len(header) - len(units_row))
    elif len(units_row) > len(header):
        units_row = units_row[: len(header)]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if units_row:
            writer.writerow(units_row)

        dict_writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for row in kept_rows:
            dict_writer.writerow(row)

    return {
        "output_csv": str(output_path),
        "storm_count": len(storm_summaries),
        "record_count": len(kept_rows),
        "affected_dates": sorted(affected_dates),
        "storms": storm_summaries,
    }


def plot_zhejiang_typhoon_tracks(
    track_csv: str | Path,
    out_png: str | Path,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    label_storms: bool = True,
    station_obs_csv: str | Path | None = None,
    station_col: str = "stationId",
    station_lat_col: str = "lat",
    station_lon_col: str = "lng",
) -> str:
    """
    绘制进入浙江范围的台风路径图，以浙江省边界作为底图。

    若提供 `station_obs_csv`，则从观测文件中提取唯一站点并叠加显示。
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt

    _, _, rows = _read_ibtracs_csv(track_csv)

    storms: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        sid = (row.get("SID") or "").strip()
        if not sid:
            continue
        storms.setdefault(sid, []).append(row)

    gdf = gpd.read_file(province_shp_path)
    #gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)

    station_count = 0
    if station_obs_csv is not None:
        stations = read_unique_station_locations(
            station_obs_csv,
            station_col=station_col,
            lat_col=station_lat_col,
            lon_col=station_lon_col,
        )
        station_count = len(stations)
        ax.scatter(
            stations[station_lon_col],
            stations[station_lat_col],
            s=6,
            color="#4d4d4d",
            alpha=0.45,
            edgecolors="none",
            label=f"Stations (n={station_count})",
            zorder=1,
        )

    cmap = plt.get_cmap("tab20")
    all_lons: list[float] = []
    all_lats: list[float] = []
    for idx, storm_rows in enumerate(storms.values()):
        lons: list[float] = []
        lats: list[float] = []
        for row in storm_rows:
            lat = _parse_float(row.get("LAT", ""))
            lon = _parse_float(row.get("LON", ""))
            if lat is None or lon is None:
                continue
            lons.append(lon)
            lats.append(lat)

        if not lons:
            continue

        all_lons.extend(lons)
        all_lats.extend(lats)

        color = cmap(idx % cmap.N)
        name = next((r.get("NAME", "").strip() for r in storm_rows if r.get("NAME", "").strip()), "UNKNOWN")
        ax.plot(lons, lats, "-", color=color, linewidth=1.5, alpha=0.9, zorder=2)
        ax.scatter(lons[0], lats[0], s=12, color=color, zorder=4)
        ax.scatter(lons[-1], lats[-1], s=18, color=color, edgecolors="white", linewidths=0.5, zorder=5)

        if label_storms:
            ax.text(
                lons[-1] + 0.03,
                lats[-1] + 0.03,
                name,
                fontsize=7,
                color=color,
                zorder=6,
            )

    minx, miny, maxx, maxy = gdf.total_bounds
    if all_lons and all_lats:
        minx = min(180, min(all_lons))
        maxx = max( 0,max(all_lons))
        miny = min(180, min(all_lats))
        maxy = max( 0,max(all_lats))

    xspan = maxx - minx
    yspan = maxy - miny
    xpad = max(xspan * 0.08, 0.2)
    ypad = max(yspan * 0.08, 0.2)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "Typhoon Tracks Affecting Zhejiang"
    if station_count:
        title += f" with Stations (n={station_count})"
        ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


if __name__ == "__main__":
    #台风最佳路径 数据源：  https://www.ncei.noaa.gov/products/international-best-track-archive 
    input_csv = "ibtracs.WP.list.v04r01-2016-2025.csv"
    output_csv_with_offset = "newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t3h.csv"
    output_png_with_offset = "newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t3h.png"

    # 1. 使用台风中心位置判断是否进入浙江矩形框，并向前后各扩展 12 小时。
    # 如需不扩展窗口，可将 t1_hours、t2_hours 保持默认 0。
    #(3个超出暖季的台风 KONG-REY2024/10/31-2024/11/1； MITAG,2019/9/30-2019/10/2；  HAIMA,2016/10/22-HAIMA,2016/10/23)
    summary = filter_typhoon_tracks_in_zhejiang(
        input_csv,
        output_csv_with_offset,
        t1_hours=3,
        t2_hours=3,
        start_month=1,
        end_month=12,
    )
    print(f"已输出: {summary['output_csv']}")
    print(f"进入浙江范围的台风个数: {summary['storm_count']}")
    print(f"保留路径记录数: {summary['record_count']}")

    #2.过浙江台风路径图
    plot_path = plot_zhejiang_typhoon_tracks(
        output_csv_with_offset,
        output_png_with_offset,
        station_obs_csv=DEFAULT_STATION_OBS_CSV,
    )
    print(f"路径图已输出: {plot_path}")

#台风日列表:
#['2016-09-15', '2016-09-16', '2018-07-21', '2018-07-22', '2018-07-23', '2018-08-02', '2018-08-03',
# '2018-08-12', '2018-08-13', '2018-08-16', '2018-08-17', '2019-08-09', '2019-08-10', '2019-09-30',
# '2020-08-03', '2020-08-04', '2021-07-24', '2021-07-25', '2021-07-26', '2021-07-27', '2021-09-12',
# '2021-09-13', '2022-09-13', '2022-09-14', '2022-09-15', '2024-09-15', '2024-09-16', '2024-09-19',
# '2025-07-07', '2025-07-08', '2025-07-29', '2025-07-30', '2025-07-31']
#台风日总数: 33