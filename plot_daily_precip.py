from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"


def plot_daily_precip_distribution_from_merged_nc(
    nc_path: str | Path = "newout/daily_precip_tps_500m/prep_daily_2016_2025.nc",
    target_date: str = "2024-09-18",
    out_png_path: str | Path | None = None,
    var_name: str = "prep",
    cmap: str = "Blues",
    title: str | None = None,
    vmax_percentile: float = 99.0,
    overlay_province_boundary: bool = True,
    province_shp_path: str | None = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    """
    从合并后的逐日降水 NetCDF 中读取任意一天，并输出降水空间分布图。

    `overlay_province_boundary=False` 时不叠加浙江省边界。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc_path = Path(nc_path)
    day = pd.Timestamp(target_date).normalize()

    with nc.Dataset(nc_path, "r") as ds:
        if var_name not in ds.variables:
            raise ValueError(f"{nc_path} 不包含变量 {var_name!r}")
        if "time" not in ds.variables or "lat" not in ds.variables or "lon" not in ds.variables:
            raise ValueError(f"{nc_path} 缺少 time/lat/lon 坐标变量")

        time_var = ds.variables["time"]
        time_units = getattr(time_var, "units", None)
        time_calendar = getattr(time_var, "calendar", "standard")
        if time_units is None:
            raise ValueError(f"{nc_path} 的 time 变量缺少 units 属性")
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                nc.num2date(
                    time_var[:],
                    units=time_units,
                    calendar=time_calendar,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
            )
        ).normalize()

        matches = np.flatnonzero(dates == day)
        if len(matches) == 0:
            available_range = f"{dates.min().date()} 到 {dates.max().date()}" if len(dates) else "空"
            raise ValueError(f"{nc_path} 中没有日期 {day.date()}，可用日期范围：{available_range}")
        time_index = int(matches[0])

        lats = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lons = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        prep = ds.variables[var_name][time_index, :, :]
        if np.ma.isMaskedArray(prep):
            prep = prep.filled(np.nan)
        prep = np.asarray(prep, dtype=np.float64)

    if prep.shape != (len(lats), len(lons)):
        raise ValueError(f"{nc_path} 中 {var_name!r} 形状与 lat/lon 尺寸不一致")
    valid = prep[np.isfinite(prep)]
    if len(valid) == 0:
        raise ValueError(f"{day.date()} 没有有效降水像元，无法绘图。")

    vmax = float(np.nanpercentile(valid, vmax_percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(valid))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    if out_png_path is None:
        out_png_path = nc_path.with_name(f"{nc_path.stem}_{day.strftime('%Y%m%d')}.png")
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lon2d, lat2d = np.meshgrid(lons, lats)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.6), dpi=180, constrained_layout=True)
    im = ax.pcolormesh(
        lon2d,
        lat2d,
        prep,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    if overlay_province_boundary and province_shp_path is not None:
        try:
            import geopandas as gpd

            gdf = gpd.read_file(province_shp_path)
            if "NAME" in gdf.columns:
                gdf = gdf[gdf["NAME"] == "浙江省"]
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            if not gdf.empty:
                gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=5)
        except Exception as e:
            print(f"[warn] overlay province boundary failed: {e}")
    ax.set_title(title or f"Daily precipitation {day.strftime('%Y-%m-%d')}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(lons.min()), float(lons.max()))
    ax.set_ylim(float(lats.min()), float(lats.max()))
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("precipitation (mm/day)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] {day.date()} 降水分布图已输出 -> {out_path}")
    return str(out_path)


if __name__ == "__main__":
     #绘制逐日降水nc文件的分布图
    plot_daily_precip_distribution_from_merged_nc(
        nc_path="newout/daily_precip_tps_500m/prep_daily_2016_2025.nc",
        target_date="2021-07-25",
        out_png_path="newout/plot_daily_precip_tps_500m/prep_daily_20210725.png",
        overlay_province_boundary=False,
    )
    exit()