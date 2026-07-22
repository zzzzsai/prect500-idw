from collections.abc import Iterable

import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_STATION_OBS_CSV = "/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv"
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"

def merge_station_obs_with_2025_monthly_files(
    base_csv_path: str = DEFAULT_STATION_OBS_CSV,
    monthly_dir: str = "/home/zsai/python_projects/prect500-data/nb_stations_obs",
    monthly_pattern: str = "2025-*.csv",
    out_csv_path: str = "/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv",
    station_col: str = "stationId",
    date_col: str = "datetime",
) -> pd.DataFrame:
    """
    将 2025 年各月份降水文件与 2014-2024 合并文件拼接，生成 2014-2025 合并文件。
    """
    base_path = Path(base_csv_path)
    if not base_path.exists():
        raise FileNotFoundError(f"未找到历史合并文件: {base_path}")

    monthly_paths = sorted(Path(monthly_dir).glob(monthly_pattern))
    if not monthly_paths:
        raise FileNotFoundError(
            f"未在目录 {monthly_dir} 下找到匹配 {monthly_pattern} 的 2025 月文件。"
        )

    base_df = pd.read_csv(base_path, low_memory=False)
    required_cols = [station_col, date_col]
    missing_base_cols = [c for c in required_cols if c not in base_df.columns]
    if missing_base_cols:
        raise ValueError(f"历史合并文件缺少必要列: {missing_base_cols}")

    base_df = base_df.copy()
    base_df[station_col] = _normalize_station_id_series(base_df[station_col])
    base_df[date_col] = pd.to_datetime(base_df[date_col], errors="coerce").dt.normalize()
    base_df = base_df.dropna(subset=[station_col, date_col]).copy()

    monthly_frames: list[pd.DataFrame] = []
    for p in monthly_paths:
        month_df = pd.read_csv(p, low_memory=False)
        missing_month_cols = [c for c in required_cols if c not in month_df.columns]
        if missing_month_cols:
            raise ValueError(f"月文件 {p} 缺少必要列: {missing_month_cols}")
        month_df = month_df.copy()
        month_df[station_col] = _normalize_station_id_series(month_df[station_col])
        month_df[date_col] = pd.to_datetime(month_df[date_col], errors="coerce").dt.normalize()
        month_df = month_df.dropna(subset=[station_col, date_col]).copy()
        monthly_frames.append(month_df)

    df_2025 = pd.concat(monthly_frames, ignore_index=True)
    merged_df = pd.concat([base_df, df_2025], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=[station_col, date_col], keep="last")
    merged_df = merged_df.sort_values(by=[station_col, date_col], ascending=[True, True]).reset_index(drop=True)
    merged_df[date_col] = merged_df[date_col].dt.strftime("%Y-%m-%d")

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    print(
        f"[hazard_map] 已拼接 2025 月文件({len(monthly_paths)} 个)，输出 -> {out_path}; "
        f"总记录数={len(merged_df)}"
    )
    return merged_df


def find_stations_with_valid_observation_ratio(
    csv_path: str = DEFAULT_STATION_OBS_CSV,
    station_col: str = "stationId",
    date_col: str = "datetime",
    value_col: str = "r2020",
    lat_col: str = "lat",
    lon_col: str = "lng",
    start_date: str = "2016-01-01",
    end_date: str = "2024-12-31",
    valid_ratio_threshold: float = 0.90,
    months: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    找出在给定时段内有效观测天数占比超过阈值的站点。

    有效观测的定义为：
    1. 该站该日存在记录；
    2. `value_col` 可转为数值；
    3. 数值有限且 >= 0。
    """
    df = pd.read_csv(csv_path, low_memory=False)
    
    required_cols = [station_col, date_col, value_col, lat_col, lon_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}；当前列为: {list(df.columns)}")

    df[station_col] = _normalize_station_id_series(df[station_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[station_col, date_col])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df[date_col] >= start_ts) & (df[date_col] <= end_ts)].copy()
    if df.empty:
        raise ValueError("给定时段内没有任何站点观测记录。")

    expected_dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
    if months is not None:
        month_set = set(months)
        df = df[df[date_col].dt.month.isin(month_set)].copy()
        expected_dates = expected_dates[expected_dates.month.isin(month_set)]

    
    expected_days = len(expected_dates)
    if expected_days == 0:
        raise ValueError("筛选后的统计时段为空，请检查 start_date/end_date/months。")

    df["date"] = df[date_col].dt.normalize()
    valid_mask = df[value_col].notna() & np.isfinite(df[value_col]) & (df[value_col] >= 0)
    df_valid = (
        df.loc[valid_mask, [station_col, lat_col, lon_col, "date"]]
        .drop_duplicates(subset=[station_col, "date"])
        .copy()
    )

    station_meta = df[[station_col, lat_col, lon_col]].drop_duplicates(subset=[station_col]).copy()
    coverage = df_valid.groupby(station_col, as_index=False).agg(valid_obs_days=("date", "nunique"))

    out = station_meta.merge(coverage, on=station_col, how="left")
    out["valid_obs_days"] = out["valid_obs_days"].fillna(0).astype(int)
    out["expected_days"] = expected_days
    out["valid_ratio"] = out["valid_obs_days"] / expected_days
    out = out.sort_values(
        by=["valid_ratio", "valid_obs_days", station_col],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    passed = out[out["valid_ratio"] >= valid_ratio_threshold].copy().reset_index(drop=True)
    print(
        f"[hazard_map] 时段 {start_ts.date()} ~ {end_ts.date()}，"
        f"期望天数={expected_days}，通过阈值 {valid_ratio_threshold:.0%} 的站点数="
        f"{len(passed)}/{len(out)}"
    )
    passed = passed.sort_values(by=['valid_obs_days'], ascending=[True])
    if len(passed):
        print(passed.head(10).to_string(index=False))
    else:
        print("[hazard_map] 没有站点满足覆盖率阈值。")
    return passed



def count_daily_valid_station_observations(
    csv_path: str = DEFAULT_STATION_OBS_CSV,
    station_col: str = "stationId",
    date_col: str = "datetime",
    value_col: str = "r2020",
    start_year: int = 2014,
    end_year: int = 2024,
    warm_month_start: int = 5,
    warm_month_end: int = 9,
    chunksize: int = 500_000,
    out_csv_path: str | None = "/home/zsai/python_projects/prect500-v2/valid_station_daily_counts_2016_2024_5to9.csv",
    allowed_station_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    统计 2016-2024 年 5-9 月每天“有有效观测”的站点数量。

    有效观测定义与前文保持一致：
    - 站点 ID 有效
    - 日期可解析
    - 降水值可转为数值，且有限并且 >= 0

    若提供 `allowed_station_ids`，仅统计这些站点（须与 `find_stations_with_valid_observation_ratio`
    中使用的 `station_col` 规范化规则一致）。
    """
    allowed_set: frozenset[str] | None = None
    if allowed_station_ids is not None:
        allowed_set = frozenset(_normalize_station_id_series(pd.Series(list(allowed_station_ids))).tolist())

    dates = iter_warm_season_dates(
        start_year=start_year,
        end_year=end_year,
        warm_month_start=warm_month_start,
        warm_month_end=warm_month_end,
    )
    if len(dates) == 0:
        raise ValueError("统计日期为空，请检查年份或月份范围。")

    min_date = dates.min()
    max_date = dates.max()
    month_set = set(dates.month.tolist())
    daily_station_sets: dict[pd.Timestamp, set[str]] = {}

    for chunk in pd.read_csv(
        csv_path,
        usecols=[station_col, date_col, value_col],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk[station_col] = _normalize_station_id_series(chunk[station_col])
        if allowed_set is not None:
            chunk = chunk.loc[chunk[station_col].isin(allowed_set)].copy()
            if chunk.empty:
                continue
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk = chunk.dropna(subset=[station_col, date_col])
        chunk = chunk[(chunk[date_col] >= min_date) & (chunk[date_col] <= max_date)].copy()
        if chunk.empty:
            continue

        chunk = chunk[chunk[date_col].dt.month.isin(month_set)].copy()
        if chunk.empty:
            continue

        valid_mask = chunk[value_col].notna() & np.isfinite(chunk[value_col]) & (chunk[value_col] >= 0)
        chunk = chunk.loc[valid_mask, [station_col, date_col]].copy()
        if chunk.empty:
            continue

        chunk["date"] = chunk[date_col].dt.normalize()
        chunk = chunk.drop_duplicates(subset=[station_col, "date"])

        for date_val, station_ids in chunk.groupby("date")[station_col]:
            date_key = pd.Timestamp(date_val).normalize()
            if date_key not in daily_station_sets:
                daily_station_sets[date_key] = set()
            daily_station_sets[date_key].update(station_ids.tolist())

    out = pd.DataFrame({"date": dates.normalize()})
    out["year"] = out["date"].dt.year.astype(int)
    out["valid_station_count"] = out["date"].map(lambda d: len(daily_station_sets.get(pd.Timestamp(d), set()))).astype(int)

    vc = out["valid_station_count"]
    q = vc.quantile([0.05,0.1, 0.5, 0.9])
    print(
        "[hazard_map] Daily valid-station counts summary: "
        f"days={len(out)}, min={vc.min()}, "
        f"p05={q.loc[0.05]:.4g},p10={q.loc[0.1]:.4g},  p50={q.loc[0.5]:.4g}, p90={q.loc[0.9]:.4g}, "
        f"max={vc.max()}"
    )

    out = out.sort_values("valid_station_count", ascending=True).reset_index(drop=True)

    if out_csv_path:
        out_path = Path(out_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(out.head(30).to_string(index=False))
        print(f"[hazard_map] Saved daily valid-station counts -> {out_path}")
    return out


def daily_valid_station_counts_after_ratio_screen(
    csv_path: str = DEFAULT_STATION_OBS_CSV,
    station_col: str = "stationId",
    date_col: str = "datetime",
    value_col: str = "r2020",
    lat_col: str = "lat",
    lon_col: str = "lng",
    screen_start_date: str = "2016-01-01",
    screen_end_date: str = "2024-12-31",
    valid_ratio_threshold: float = 0.95,
    screen_months: Iterable[int] | None = None,
    count_start_year: int = 2016,
    count_end_year: int = 2024,
    warm_month_start: int = 5,
    warm_month_end: int = 9,
    chunksize: int = 500_000,
    filtered_obs_csv_path: str | None = None,
    daily_counts_csv_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    串联分析：
    1. 在给定时段内按 `find_stations_with_valid_observation_ratio` 规则筛选有效观测占比
       不低于 `valid_ratio_threshold` 的站点；
    2. 可选将原始 CSV 中仅属于这些站点的行写出到 `filtered_obs_csv_path`（全列保留）；
    3. 调用 `count_daily_valid_station_observations`，只统计这些站点在暖季逐日的有效观测站数。

    返回 `(通过筛选的站点表, 逐日有效站数表)`。
    """
    passed = find_stations_with_valid_observation_ratio(
        csv_path=csv_path,
        station_col=station_col,
        date_col=date_col,
        value_col=value_col,
        lat_col=lat_col,
        lon_col=lon_col,
        start_date=screen_start_date,
        end_date=screen_end_date,
        valid_ratio_threshold=valid_ratio_threshold,
        months=screen_months,
    )
    if passed.empty:
        raise ValueError("没有站点通过有效观测占比筛选，无法继续统计逐日有效站数。")

    allowed_ids = passed[station_col].tolist()

    if filtered_obs_csv_path is not None:
        out_path = Path(filtered_obs_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        allowed_set = frozenset(
            _normalize_station_id_series(pd.Series(allowed_ids)).tolist()
        )
        header_written = False
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            if station_col not in chunk.columns:
                raise ValueError(f"写出筛选观测时 CSV 缺少列 {station_col!r}。")
            chunk = chunk.copy()
            chunk[station_col] = _normalize_station_id_series(chunk[station_col])
            chunk = chunk.loc[chunk[station_col].isin(allowed_set)]
            if chunk.empty:
                continue
            chunk.to_csv(
                out_path,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
        print(f"[hazard_map] 已写出仅含筛选站点的原始观测 -> {out_path.resolve()}")

    daily = count_daily_valid_station_observations(
        csv_path=csv_path,
        station_col=station_col,
        date_col=date_col,
        value_col=value_col,
        start_year=count_start_year,
        end_year=count_end_year,
        warm_month_start=warm_month_start,
        warm_month_end=warm_month_end,
        chunksize=chunksize,
        out_csv_path=daily_counts_csv_path,
        allowed_station_ids=allowed_ids,
    )
    return passed, daily


def iter_warm_season_dates(
    start_year: int = 2016,
    end_year: int = 2024,
    warm_month_start: int = 5,
    warm_month_end: int = 9,
) -> pd.DatetimeIndex:
    dates: list[pd.Timestamp] = []
    for year in range(int(start_year), int(end_year) + 1):
        dr = pd.date_range(
            start=f"{year}-{warm_month_start:02d}-01",
            end=f"{year}-{warm_month_end:02d}-30",
            freq="D",
        )
        dates.extend(dr.to_pydatetime())
    return pd.DatetimeIndex(dates).normalize()


def plot_station_distribution(
    stations_df: pd.DataFrame,
    out_png: str | Path,
    lat_col: str = "lat",
    lon_col: str = "lng",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    color_by: str | None = "valid_ratio",
) -> str:
    """
    绘制站点空间分布图，以浙江省边界作为底图。

    若 `color_by` 在表中存在，则按该列着色；否则统一颜色绘制。
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt

    required_cols = [lat_col, lon_col]
    missing_cols = [c for c in required_cols if c not in stations_df.columns]
    if missing_cols:
        raise ValueError(f"站点表缺少必要列: {missing_cols}")
    if stations_df.empty:
        raise ValueError("站点表为空，无法绘制分布图。")

    df = stations_df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    if df.empty:
        raise ValueError("站点坐标均为空，无法绘图。")

    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=3)

    if color_by is not None and color_by in df.columns:
        sc = ax.scatter(
            df[lon_col],
            df[lat_col],
            c=df[color_by],
            s=10,
            cmap="viridis",
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(color_by)
    else:
        ax.scatter(
            df[lon_col],
            df[lat_col],
            s=10,
            color="#1f78b4",
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )

    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = max((maxx - minx) * 0.03, 0.08)
    ypad = max((maxy - miny) * 0.03, 0.08)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Station Distribution (n={len(df)})")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)
    plt.tight_layout()

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _normalize_station_id_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    return out.str.replace(r"\.0$", "", regex=True)




if __name__ == "__main__":
    #merge_station_obs_with_2025_monthly_files()
    #exit()
    #0、计算每天的观测站点数量
    count_daily_valid_station_observations(DEFAULT_STATION_OBS_CSV,start_year=2016,end_year=2025,
        warm_month_start=1,
        warm_month_end=12,
        out_csv_path='newout/valid_station_daily_counts_2016_2025.csv')
    exit()
    # 1. 找到有有效观测的站点并绘制分布图
    passed = find_stations_with_valid_observation_ratio(
        #csv_path='',
        start_date="2016-01-01",
        end_date="2025-12-31",
        valid_ratio_threshold=0.95,
        months=[5, 6, 7, 8, 9],
    )
    plot_path = plot_station_distribution(
        passed,
        out_png="newout/valid_station_distribution_95pct_5to9.png",
    )
    print("站点分布图已输出:", plot_path)
    exit()


    
    
    #2、先筛选有效站点，再统计每天站点数量
    passed, daily = daily_valid_station_counts_after_ratio_screen(
    csv_path=DEFAULT_STATION_OBS_CSV,
    screen_start_date="2016-01-01",
    screen_end_date="2024-12-31",
    valid_ratio_threshold=0.95,
    screen_months=[5, 6, 7, 8, 9],
    count_start_year=2016,
    count_end_year=2024,
    warm_month_start=5,
    warm_month_end=9,
    filtered_obs_csv_path="out/station_obs_screened_95pct_5to9.csv",  # 可改为 None 跳过写出
    daily_counts_csv_path="out/valid_station_daily_counts_screened.csv",
)
    exit()
    
    