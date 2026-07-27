from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


DEFAULT_DAILY_PRECIP_NC = "newout/daily_precip_tps_500m/prep_daily_2016_2025.nc"
DEFAULT_R99P_NC = "newout/r95p_typhoon_contribution/r99p_typhoon_contribution_2016_2025_1to12.nc"
DEFAULT_OUT_NC = "newout/usri/hazard_r99p_prob_sdii50_2016_2025.nc"
DEFAULT_HAZARD_NC = DEFAULT_OUT_NC
DEFAULT_GIS_DATA_DIR = "/nas/zsai/python_projects/prect500-idw-gis-data"
DEFAULT_EXTRACT_DIR = "/nas/zsai/python_projects/prect500-idw-gis-data"
DEFAULT_EXPOSURE_OUT_NC = "newout/usri/exposure_population_gdp_landcover_500m.nc"
DEFAULT_EXPOSURE_NC = DEFAULT_EXPOSURE_OUT_NC
DEFAULT_USRI_OUT_NC = "newout/usri/usri_2016_2025_500m.nc"
DEFAULT_USRI_NC = DEFAULT_USRI_OUT_NC
DEFAULT_ATTRIBUTION_OUT_NC = "newout/usri/usri_driver_type_500m.nc"
DEFAULT_DEM_NC = "static/dem_zhejiang_500m_merged.nc"
DEFAULT_STATION_OBS_CSV = "/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv"
DEFAULT_DEM_STATION_OUT_PNG = "newout/usri/figs/dem_station_distribution_500m.png"
DEFAULT_LANDCOVER_TIF = None
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"
DEFAULT_NORMALIZE_QMIN = 0.0
DEFAULT_NORMALIZE_QMAX = 99.5
DEFAULT_CROPLAND_CODES = (10,)
DEFAULT_BUILTUP_CODES = (80,)


def _read_2d_metric_nc(
    nc_path: str | Path,
    var_name: str,
    lat_var: str = "lat",
    lon_var: str = "lon",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with nc.Dataset(path, "r") as ds:
        for name in (lat_var, lon_var, var_name):
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")
        lats = np.asarray(ds.variables[lat_var][:], dtype=np.float32)
        lons = np.asarray(ds.variables[lon_var][:], dtype=np.float32)
        data = ds.variables[var_name][:]
        if np.ma.isMaskedArray(data):
            data = data.filled(np.nan)
        data = np.asarray(data, dtype=np.float32)
    if data.shape != (len(lats), len(lons)):
        raise ValueError(f"{path} 中 {var_name!r} 形状与 lat/lon 尺寸不一致: {data.shape}")
    return lats, lons, data


def _minmax_normalize(
    data: np.ndarray,
    valid_mask: np.ndarray | None = None,
    q_min: float = DEFAULT_NORMALIZE_QMIN,
    q_max: float = DEFAULT_NORMALIZE_QMAX,
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if valid_mask is None:
        mask = np.isfinite(arr)
    else:
        mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if not mask.any():
        return out

    vmin = float(np.nanpercentile(arr[mask], q_min))
    vmax = float(np.nanpercentile(arr[mask], q_max))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return out
    if np.isclose(vmax, vmin):
        out[mask] = 0.0
        return out

    clipped = np.clip(arr[mask], vmin, vmax)
    out[mask] = ((clipped - vmin) / (vmax - vmin)).astype(np.float32)
    return out


def _check_same_grid(
    lats_a: np.ndarray,
    lons_a: np.ndarray,
    lats_b: np.ndarray,
    lons_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> None:
    if lats_a.shape != lats_b.shape or lons_a.shape != lons_b.shape:
        raise ValueError(f"{label_a} 与 {label_b} 的 lat/lon 尺寸不一致")
    if not np.array_equal(lats_a, lats_b) or not np.array_equal(lons_a, lons_b):
        raise ValueError(f"{label_a} 与 {label_b} 的 lat/lon 坐标不一致")


def compute_prob_and_sdii50_from_merged_precip(
    daily_precip_nc: str | Path = DEFAULT_DAILY_PRECIP_NC,
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    storm_threshold_mm: float = 50.0,
    time_chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从合并后的逐日降水 nc 计算暴雨发生概率 Prob 和暴雨日平均强度 SDII50。
    """
    path = Path(daily_precip_nc)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with nc.Dataset(path, "r") as ds:
        for name in (lat_var, lon_var, precip_var):
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")

        lats = np.asarray(ds.variables[lat_var][:], dtype=np.float32)
        lons = np.asarray(ds.variables[lon_var][:], dtype=np.float32)
        precip = ds.variables[precip_var]
        if precip.ndim != 3:
            raise ValueError(f"{path} 中 {precip_var!r} 应为 3D(time,y,x)，实际维度为 {precip.ndim}")
        if precip.shape[1:] != (len(lats), len(lons)):
            raise ValueError(f"{path} 中 {precip_var!r} 形状与 lat/lon 尺寸不一致: {precip.shape}")

        n_time = precip.shape[0]
        valid_day_count = np.zeros((len(lats), len(lons)), dtype=np.int32)
        storm_day_count = np.zeros((len(lats), len(lons)), dtype=np.int32)
        storm_precip_sum = np.zeros((len(lats), len(lons)), dtype=np.float64)

        for start in range(0, n_time, time_chunk_size):
            stop = min(start + time_chunk_size, n_time)
            block = precip[start:stop, :, :]
            if np.ma.isMaskedArray(block):
                block = block.filled(np.nan)
            block = np.asarray(block, dtype=np.float32)

            finite_mask = np.isfinite(block)
            storm_mask = finite_mask & (block >= storm_threshold_mm)
            valid_day_count += finite_mask.sum(axis=0).astype(np.int32)
            storm_day_count += storm_mask.sum(axis=0).astype(np.int32)
            storm_precip_sum += np.where(storm_mask, block, 0.0).sum(axis=0, dtype=np.float64)

            print(f"[usri] 计算 Prob/SDII50: {stop}/{n_time} 天")

    prob = np.divide(
        storm_day_count,
        valid_day_count,
        out=np.full(valid_day_count.shape, np.nan, dtype=np.float32),
        where=valid_day_count > 0,
    ).astype(np.float32)
    sdii50 = np.divide(
        storm_precip_sum,
        storm_day_count,
        out=np.full(storm_day_count.shape, np.nan, dtype=np.float64),
        where=storm_day_count > 0,
    ).astype(np.float32)

    return lats, lons, prob, sdii50, valid_day_count, storm_day_count


def write_hazard_nc(
    out_nc_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    r99p: np.ndarray,
    prob: np.ndarray,
    sdii50: np.ndarray,
    norm_r99p: np.ndarray,
    norm_prob: np.ndarray,
    norm_sdii50: np.ndarray,
    hazard: np.ndarray,
    valid_day_count: np.ndarray,
    storm_day_count: np.ndarray,
    weights: tuple[float, float, float],
) -> str:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)

        def put_var(name: str, data: np.ndarray, long_name: str, units: str = "1") -> None:
            var = ds.createVariable(
                name,
                "f4",
                ("y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            var[:] = np.asarray(data, dtype=np.float32)
            var.long_name = long_name
            var.units = units

        put_var("r99p", r99p, "99th percentile wet-day precipitation threshold", "mm/day")
        put_var("prob", prob, "probability of daily precipitation >= 50 mm", "1")
        put_var("sdii50", sdii50, "mean precipitation intensity on days with daily precipitation >= 50 mm", "mm/day")
        put_var("norm_r99p", norm_r99p, "normalized R99p", "1")
        put_var("norm_prob", norm_prob, "normalized Prob", "1")
        put_var("norm_sdii50", norm_sdii50, "normalized SDII50", "1")
        put_var("hazard", hazard, "USRI hazard component from R99p, Prob and SDII50", "1")

        vvalid = ds.createVariable("valid_day_count", "i4", ("y", "x"), zlib=True, complevel=4, fill_value=np.int32(-9999))
        vstorm = ds.createVariable("storm_day_count", "i4", ("y", "x"), zlib=True, complevel=4, fill_value=np.int32(-9999))
        vvalid[:] = valid_day_count.astype(np.int32)
        vstorm[:] = storm_day_count.astype(np.int32)
        vvalid.long_name = "valid daily precipitation day count"
        vstorm.long_name = "daily precipitation >= 50 mm day count"
        vvalid.units = "days"
        vstorm.units = "days"

        ds.description = "USRI Hazard component using equal weights for R99p, Prob and SDII50"
        ds.normalization_method = "robust min-max with percentile clipping"
        ds.normalization_q_min = float(DEFAULT_NORMALIZE_QMIN)
        ds.normalization_q_max = float(DEFAULT_NORMALIZE_QMAX)
        ds.r99p_weight = float(weights[0])
        ds.prob_weight = float(weights[1])
        ds.sdii50_weight = float(weights[2])

    print(f"[usri] Hazard nc 输出 -> {out_path}")
    return str(out_path)


def compute_hazard_index(
    daily_precip_nc: str | Path = DEFAULT_DAILY_PRECIP_NC,
    r99p_nc: str | Path = DEFAULT_R99P_NC,
    out_nc_path: str | Path = DEFAULT_OUT_NC,
    r99p_var: str = "r99p_threshold",
    precip_var: str = "prep",
    storm_threshold_mm: float = 50.0,
    time_chunk_size: int = 32,
    weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
) -> str:
    """
    计算 USRI 的 Hazard 部分：
    Hazard = 1/3 * Norm(R99p) + 1/3 * Norm(Prob) + 1/3 * Norm(SDII50)
    """
    lats, lons, prob, sdii50, valid_day_count, storm_day_count = compute_prob_and_sdii50_from_merged_precip(
        daily_precip_nc=daily_precip_nc,
        precip_var=precip_var,
        storm_threshold_mm=storm_threshold_mm,
        time_chunk_size=time_chunk_size,
    )
    r99p_lats, r99p_lons, r99p = _read_2d_metric_nc(r99p_nc, r99p_var)
    _check_same_grid(lats, lons, r99p_lats, r99p_lons, "逐日降水文件", "R99p 文件")

    province_valid_mask = (valid_day_count > 0) & np.isfinite(r99p)
    norm_r99p = _minmax_normalize(r99p, province_valid_mask)
    norm_prob = _minmax_normalize(prob, province_valid_mask)
    norm_sdii50 = _minmax_normalize(sdii50, province_valid_mask)

    hazard = np.full(r99p.shape, np.nan, dtype=np.float32)
    hazard[province_valid_mask] = (
        weights[0] * np.nan_to_num(norm_r99p[province_valid_mask], nan=0.0)
        + weights[1] * np.nan_to_num(norm_prob[province_valid_mask], nan=0.0)
        + weights[2] * np.nan_to_num(norm_sdii50[province_valid_mask], nan=0.0)
    ).astype(np.float32)

    return write_hazard_nc(
        out_nc_path=out_nc_path,
        lats=lats,
        lons=lons,
        r99p=r99p,
        prob=prob,
        sdii50=sdii50,
        norm_r99p=norm_r99p,
        norm_prob=norm_prob,
        norm_sdii50=norm_sdii50,
        hazard=hazard,
        valid_day_count=valid_day_count,
        storm_day_count=storm_day_count,
        weights=weights,
    )


def _read_grid_from_nc(
    nc_path: str | Path,
    lat_var: str = "lat",
    lon_var: str = "lon",
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with nc.Dataset(path, "r") as ds:
        for name in (lat_var, lon_var):
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")
        lats = np.asarray(ds.variables[lat_var][:], dtype=np.float32)
        lons = np.asarray(ds.variables[lon_var][:], dtype=np.float32)
    return lats, lons


def _target_transform_from_latlon(lats: np.ndarray, lons: np.ndarray):
    xres = float(np.nanmedian(np.abs(np.diff(lons))))
    yres = float(np.nanmedian(np.abs(np.diff(lats))))
    west = float(np.nanmin(lons)) - xres / 2.0
    north = float(np.nanmax(lats)) + yres / 2.0
    return from_origin(west, north, xres, yres)


def _ensure_archive_extracted(archive_path: str | Path, extract_root: str | Path) -> Path:
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"压缩包不存在: {archive}")
    out_dir = Path(extract_root) / archive.stem
    if out_dir.exists() and any(out_dir.rglob("*")):
        print(f"[usri] 已解压，跳过: {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[usri] 解压: {archive} -> {out_dir}")
    if archive.suffix.lower() == ".rar":
        cmd = ["bsdtar", "-xf", str(archive), "-C", str(out_dir)]
    else:
        cmd = ["7z", "x", "-y", f"-o{out_dir}", str(archive)]
    subprocess.run(cmd, check=True)
    return out_dir


def _find_first_file(base_dir: str | Path, patterns: tuple[str, ...], exclude_keywords: tuple[str, ...] = ()) -> Path:
    base = Path(base_dir)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(base.rglob(pattern))
    matches = [
        path
        for path in sorted(set(matches))
        if not any(keyword.lower() in path.name.lower() for keyword in exclude_keywords)
    ]
    if not matches:
        raise FileNotFoundError(f"在 {base} 中没有找到匹配文件: {patterns}")
    return matches[0]


def _find_arcgrid_dir(base_dir: str | Path, dir_name: str = "gdp2020") -> Path:
    base = Path(base_dir)
    for path in sorted(base.rglob(dir_name)):
        if path.is_dir() and (path / "w001001.adf").exists():
            return path
    raise FileNotFoundError(f"在 {base} 中没有找到 ArcInfo Grid 目录 {dir_name!r}")


def _build_province_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    province_name: str = "浙江省",
) -> np.ndarray:
    import geopandas as gpd

    shp_path = Path(province_shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"省界 shp 不存在: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == province_name]
    if gdf.empty:
        raise ValueError(f"在 {shp_path} 中未找到 {province_name} 边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    geom = geom.buffer(1e-6)
    lon2d, lat2d = np.meshgrid(lons, lats)
    try:
        from shapely import contains_xy

        mask = contains_xy(geom, lon2d, lat2d)
    except Exception:
        from shapely.vectorized import contains

        mask = contains(geom, lon2d, lat2d)
    return np.asarray(mask, dtype=bool)


def _load_province_boundary(
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    province_name: str = "浙江省",
):
    import geopandas as gpd

    shp_path = Path(province_shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"省界 shp 不存在: {shp_path}")

    gdf = gpd.read_file(shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == province_name]
    if gdf.empty:
        raise ValueError(f"在 {shp_path} 中未找到 {province_name} 边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _configure_matplotlib_for_map():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt


def _read_station_locations(
    station_csv: str | Path,
    station_id_col: str = "stationId",
    lat_col: str = "lat",
    lon_col: str = "lng",
    chunksize: int = 500_000,
):
    import pandas as pd

    path = Path(station_csv)
    if not path.exists():
        raise FileNotFoundError(f"站点观测 CSV 不存在: {path}")

    usecols = [station_id_col, lat_col, lon_col]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={station_id_col: str}, chunksize=chunksize):
        chunk = chunk.dropna(subset=[lat_col, lon_col])
        chunk = chunk.drop_duplicates(subset=[station_id_col])
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"{path} 中没有读到有效站点经纬度。")

    stations = pd.concat(chunks, ignore_index=True)
    stations = stations.drop_duplicates(subset=[station_id_col]).reset_index(drop=True)
    stations = stations.rename(columns={lat_col: "lat", lon_col: "lon", station_id_col: "station_id"})
    return stations[["station_id", "lat", "lon"]]


def plot_dem_with_station_distribution(
    dem_nc_path: str | Path = DEFAULT_DEM_NC,
    station_csv: str | Path = DEFAULT_STATION_OBS_CSV,
    out_png: str | Path = DEFAULT_DEM_STATION_OUT_PNG,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    dem_var: str = "dem",
    lat_var: str = "lat",
    lon_var: str = "lon",
    station_id_col: str = "stationId",
    station_lat_col: str = "lat",
    station_lon_col: str = "lng",
) -> str:
    """
    绘制浙江省 500 m DEM 空间分布，并叠加降水站点位置和浙江省边界。
    """
    plt = _configure_matplotlib_for_map()
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    lats, lons, dem = _read_2d_metric_nc(dem_nc_path, dem_var, lat_var=lat_var, lon_var=lon_var)
    stations = _read_station_locations(
        station_csv,
        station_id_col=station_id_col,
        lat_col=station_lat_col,
        lon_col=station_lon_col,
    )
    boundary = _load_province_boundary(province_shp_path)

    lon2d, lat2d = np.meshgrid(lons, lats)
    dem_cmap = LinearSegmentedColormap.from_list(
        "zhejiang_dem",
        ["#5f9f1a", "#a9cf6a", "#f2e9a3", "#d69034", "#8c510a", "#f7f3d0"],
        N=256,
    )
    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=220, constrained_layout=True)
    mesh = ax.pcolormesh(lon2d, lat2d, dem, shading="auto", cmap=dem_cmap)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=5)
    ax.scatter(
        stations["lon"],
        stations["lat"],
        s=4,
        c="black",
        edgecolors="none",
        linewidths=0,
        alpha=0.85,
        label=f"Stations (n={len(stations)})",
        zorder=6,
    )

    minx, miny, maxx, maxy = boundary.total_bounds
    xpad = max((maxx - minx) * 0.04, 0.08)
    ypad = max((maxy - miny) * 0.04, 0.08)
    ax.set_xlim(max(minx - xpad, float(np.nanmin(lons))), min(maxx + xpad, float(np.nanmax(lons))))
    ax.set_ylim(max(miny - ypad, float(np.nanmin(lats))), min(maxy + ypad, float(np.nanmax(lats))))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)
    ax.set_title("浙江省 500 m DEM 与降水站点分布")
    ax.legend(loc="lower left", frameon=True, framealpha=0.85, fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Elevation (m)")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[usri] DEM 与站点分布图输出 -> {out_path}")
    print(f"[usri] 站点数量 -> {len(stations)}")
    return str(out_path)


def _reproject_continuous_to_target(
    raster_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    resampling: Resampling = Resampling.average,
) -> np.ndarray:
    path = str(raster_path)
    dst = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    dst_transform = _target_transform_from_latlon(lats, lons)

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} 缺少 CRS，无法重投影")
        data = src.read(1, masked=True)
        nodata = -3.4e38
        src_data = np.asarray(data.filled(nodata), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=resampling,
        )

    dst[~np.isfinite(dst)] = np.nan
    dst[dst < 0] = np.nan
    return dst.astype(np.float32)


def _reproject_landcover_fraction_to_target(
    landcover_paths: list[Path],
    lats: np.ndarray,
    lons: np.ndarray,
    class_codes: tuple[int, ...],
) -> np.ndarray:
    dst_transform = _target_transform_from_latlon(lats, lons)
    weighted_fraction_sum = np.zeros((len(lats), len(lons)), dtype=np.float64)
    coverage_sum = np.zeros((len(lats), len(lons)), dtype=np.float64)
    nodata = -9999.0

    for path in landcover_paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path} 缺少 CRS，无法重投影")
            data = src.read(1, masked=True)
            valid = ~np.ma.getmaskarray(data)
            arr = np.asarray(data.filled(-9999), dtype=np.int32)
            class_mask = np.isin(arr, np.asarray(class_codes, dtype=np.int32)) & valid

            src_fraction = np.where(valid, class_mask.astype(np.float32), nodata).astype(np.float32)
            src_coverage = np.where(valid, 1.0, nodata).astype(np.float32)
            dst_fraction = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
            dst_coverage = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

            reproject(
                source=src_fraction,
                destination=dst_fraction,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            reproject(
                source=src_coverage,
                destination=dst_coverage,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )

            valid_dst = np.isfinite(dst_fraction) & np.isfinite(dst_coverage) & (dst_coverage > 0)
            weighted_fraction_sum[valid_dst] += dst_fraction[valid_dst] * dst_coverage[valid_dst]
            coverage_sum[valid_dst] += dst_coverage[valid_dst]
            print(f"[usri] 地表覆盖占比重采样完成: {path.name}")

    fraction = np.divide(
        weighted_fraction_sum,
        coverage_sum,
        out=np.full(weighted_fraction_sum.shape, np.nan, dtype=np.float64),
        where=coverage_sum > 0,
    ).astype(np.float32)
    fraction[(fraction < 0) | (fraction > 1)] = np.nan
    return fraction


def _write_exposure_nc(
    out_nc_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    pop_density: np.ndarray,
    gdp_density: np.ndarray,
    cropland_ratio: np.ndarray,
    builtup_ratio: np.ndarray,
    norm_pop_density: np.ndarray,
    norm_gdp_density: np.ndarray,
    norm_cropland_ratio: np.ndarray,
    norm_builtup_ratio: np.ndarray,
    exposure: np.ndarray,
    weights: tuple[float, float, float, float],
) -> str:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))
        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)

        def put_var(name: str, data: np.ndarray, long_name: str, units: str = "1") -> None:
            var = ds.createVariable(name, "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
            var[:] = np.asarray(data, dtype=np.float32)
            var.long_name = long_name
            var.units = units

        put_var("pop_density", pop_density, "population density proxy resampled to 500 m grid", "1")
        put_var("gdp_density", gdp_density, "GDP density proxy resampled to 500 m grid", "1")
        put_var("cropland_ratio", cropland_ratio, "cropland area fraction from GlobeLand30 class 10", "1")
        put_var("builtup_ratio", builtup_ratio, "built-up area fraction from GlobeLand30 class 80", "1")
        put_var("norm_pop_density", norm_pop_density, "normalized population density", "1")
        put_var("norm_gdp_density", norm_gdp_density, "normalized GDP density", "1")
        put_var("norm_cropland_ratio", norm_cropland_ratio, "normalized cropland ratio", "1")
        put_var("norm_builtup_ratio", norm_builtup_ratio, "normalized built-up ratio", "1")
        put_var("exposure", exposure, "USRI exposure component", "1")

        ds.description = "USRI Exposure component from population, GDP, cropland ratio and built-up ratio"
        ds.normalization_method = "robust min-max with percentile clipping"
        ds.normalization_q_min = float(DEFAULT_NORMALIZE_QMIN)
        ds.normalization_q_max = float(DEFAULT_NORMALIZE_QMAX)
        ds.population_gdp_transform_before_normalization = "log1p"
        ds.pop_density_weight = float(weights[0])
        ds.gdp_density_weight = float(weights[1])
        ds.cropland_ratio_weight = float(weights[2])
        ds.builtup_ratio_weight = float(weights[3])
        ds.landcover_source = "GlobeLand30"
        ds.landcover_cropland_codes = ",".join(str(code) for code in DEFAULT_CROPLAND_CODES)
        ds.landcover_builtup_codes = ",".join(str(code) for code in DEFAULT_BUILTUP_CODES)

    print(f"[usri] Exposure nc 输出 -> {out_path}")
    return str(out_path)


def compute_exposure_index(
    reference_grid_nc: str | Path = DEFAULT_DAILY_PRECIP_NC,
    gis_data_dir: str | Path = DEFAULT_GIS_DATA_DIR,
    extract_dir: str | Path = DEFAULT_EXTRACT_DIR,
    out_nc_path: str | Path = DEFAULT_EXPOSURE_OUT_NC,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    population_archive: str = "landscan-mosaic-china-v1-assets.zip",
    gdp_archive: str = "数据GDP2020年.rar",
    landcover_tif: str | Path | None = DEFAULT_LANDCOVER_TIF,
    landcover_archives: tuple[str, ...] = (
        "N50_25_2025LC030.rar",
        "N50_30_2025LC030.rar",
        "N51_25_2025LC030.rar",
        "N51_30_2025LC030.rar",
    ),
    cropland_codes: tuple[int, ...] = DEFAULT_CROPLAND_CODES,
    builtup_codes: tuple[int, ...] = DEFAULT_BUILTUP_CODES,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.15, 0.15),
) -> str:
    """
    计算 USRI 的 Exposure 部分：
    Exposure = 0.4*Norm(PopDensity) + 0.3*Norm(GDPDensity)
             + 0.15*Norm(CroplandRatio) + 0.15*Norm(BuiltupRatio)
    """
    lats, lons = _read_grid_from_nc(reference_grid_nc)
    province_mask = _build_province_mask(lats, lons, province_shp_path=province_shp_path)

    gis_dir = Path(gis_data_dir)
    extract_root = Path(extract_dir)
    try:
        population_tif = _find_first_file(
            gis_dir,
            ("landscan-mosaic-china-v1.tif",),
            exclude_keywords=("confidence", "colorized"),
        )
    except FileNotFoundError:
        population_dir = _ensure_archive_extracted(gis_dir / population_archive, extract_root)
        population_tif = _find_first_file(
            population_dir,
            ("*.tif", "*.tiff"),
            exclude_keywords=("confidence", "colorized"),
        )

    try:
        gdp_grid = _find_arcgrid_dir(gis_dir, "gdp2020")
    except FileNotFoundError:
        gdp_dir = _ensure_archive_extracted(gis_dir / gdp_archive, extract_root)
        gdp_grid = _find_arcgrid_dir(gdp_dir, "gdp2020")

    landcover_tifs: list[Path] = []
    if landcover_tif is not None and Path(landcover_tif).exists():
        landcover_tifs.append(Path(landcover_tif))
    else:
        landcover_dirs: list[Path] = []
        for archive_name in landcover_archives:
            archive_stem = Path(archive_name).stem
            extracted_dir = gis_dir / archive_stem
            if extracted_dir.exists():
                landcover_dirs.append(extracted_dir)
            else:
                landcover_dirs.append(_ensure_archive_extracted(gis_dir / archive_name, extract_root))
        for landcover_dir in landcover_dirs:
            landcover_tifs.append(_find_first_file(landcover_dir, ("*.tif", "*.tiff")))

    print(f"[usri] 人口栅格: {population_tif}")
    print(f"[usri] GDP 栅格: {gdp_grid}")
    print(f"[usri] 地表覆盖栅格数: {len(landcover_tifs)}")

    pop_density = _reproject_continuous_to_target(population_tif, lats, lons)
    gdp_density = _reproject_continuous_to_target(gdp_grid, lats, lons)
    cropland_ratio = _reproject_landcover_fraction_to_target(landcover_tifs, lats, lons, class_codes=cropland_codes)
    builtup_ratio = _reproject_landcover_fraction_to_target(landcover_tifs, lats, lons, class_codes=builtup_codes)

    for arr in (pop_density, gdp_density, cropland_ratio, builtup_ratio):
        arr[~province_mask] = np.nan

    exposure_valid_mask = province_mask & (
        np.isfinite(pop_density)
        | np.isfinite(gdp_density)
        | np.isfinite(cropland_ratio)
        | np.isfinite(builtup_ratio)
    )
    norm_pop_density = _minmax_normalize(np.log1p(pop_density), exposure_valid_mask)
    norm_gdp_density = _minmax_normalize(np.log1p(gdp_density), exposure_valid_mask)
    norm_cropland_ratio = _minmax_normalize(cropland_ratio, exposure_valid_mask)
    norm_builtup_ratio = _minmax_normalize(builtup_ratio, exposure_valid_mask)

    exposure = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    exposure[exposure_valid_mask] = (
        weights[0] * np.nan_to_num(norm_pop_density[exposure_valid_mask], nan=0.0)
        + weights[1] * np.nan_to_num(norm_gdp_density[exposure_valid_mask], nan=0.0)
        + weights[2] * np.nan_to_num(norm_cropland_ratio[exposure_valid_mask], nan=0.0)
        + weights[3] * np.nan_to_num(norm_builtup_ratio[exposure_valid_mask], nan=0.0)
    ).astype(np.float32)

    return _write_exposure_nc(
        out_nc_path=out_nc_path,
        lats=lats,
        lons=lons,
        pop_density=pop_density,
        gdp_density=gdp_density,
        cropland_ratio=cropland_ratio,
        builtup_ratio=builtup_ratio,
        norm_pop_density=norm_pop_density,
        norm_gdp_density=norm_gdp_density,
        norm_cropland_ratio=norm_cropland_ratio,
        norm_builtup_ratio=norm_builtup_ratio,
        exposure=exposure,
        weights=weights,
    )


def write_usri_nc(
    out_nc_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    hazard: np.ndarray,
    exposure: np.ndarray,
    usri: np.ndarray,
) -> str:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)

        def put_var(name: str, data: np.ndarray, long_name: str) -> None:
            var = ds.createVariable(
                name,
                "f4",
                ("y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            var[:] = np.asarray(data, dtype=np.float32)
            var.long_name = long_name
            var.units = "1"

        put_var("hazard", hazard, "USRI hazard component")
        put_var("exposure", exposure, "USRI exposure component")
        put_var("usri", usri, "urban storm rainfall risk index")
        ds.description = "USRI = Hazard x Exposure"

    print(f"[usri] USRI nc 输出 -> {out_path}")
    return str(out_path)


def compute_usri_index(
    hazard_nc: str | Path = DEFAULT_HAZARD_NC,
    exposure_nc: str | Path = DEFAULT_EXPOSURE_NC,
    out_nc_path: str | Path = DEFAULT_USRI_OUT_NC,
    hazard_var: str = "hazard",
    exposure_var: str = "exposure",
) -> str:
    """
    计算城市暴雨气候风险综合评估指数：
    USRI = Hazard × Exposure
    """
    hazard_lats, hazard_lons, hazard = _read_2d_metric_nc(hazard_nc, hazard_var)
    exposure_lats, exposure_lons, exposure = _read_2d_metric_nc(exposure_nc, exposure_var)
    _check_same_grid(hazard_lats, hazard_lons, exposure_lats, exposure_lons, "Hazard 文件", "Exposure 文件")

    valid_mask = np.isfinite(hazard) & np.isfinite(exposure)
    usri = np.full(hazard.shape, np.nan, dtype=np.float32)
    usri[valid_mask] = (hazard[valid_mask] * exposure[valid_mask]).astype(np.float32)

    return write_usri_nc(
        out_nc_path=out_nc_path,
        lats=hazard_lats,
        lons=hazard_lons,
        hazard=hazard,
        exposure=exposure,
        usri=usri,
    )


def _safe_contribution(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator_arr = np.asarray(numerator, dtype=np.float32)
    denominator_arr = np.asarray(denominator, dtype=np.float32)
    out = np.full(numerator_arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(numerator_arr) & np.isfinite(denominator_arr) & (denominator_arr > 0)
    out[valid_mask] = (numerator_arr[valid_mask] / denominator_arr[valid_mask]).astype(np.float32)
    return out


def write_usri_attribution_nc(
    out_nc_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    hazard: np.ndarray,
    exposure: np.ndarray,
    usri: np.ndarray,
    driver_type: np.ndarray,
    r99p_contrib: np.ndarray,
    prob_contrib: np.ndarray,
    sdii50_contrib: np.ndarray,
    pop_contrib: np.ndarray,
    gdp_contrib: np.ndarray,
    cropland_contrib: np.ndarray,
    builtup_contrib: np.ndarray,
    hazard_threshold: float,
    exposure_threshold: float,
    high_percentile: float,
) -> str:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)

        def put_float_var(name: str, data: np.ndarray, long_name: str, units: str = "1") -> None:
            var = ds.createVariable(
                name,
                "f4",
                ("y", "x"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            var[:] = np.asarray(data, dtype=np.float32)
            var.long_name = long_name
            var.units = units

        put_float_var("hazard", hazard, "USRI hazard component")
        put_float_var("exposure", exposure, "USRI exposure component")
        put_float_var("usri", usri, "urban storm rainfall risk index")
        put_float_var("r99p_contrib", r99p_contrib, "R99p contribution rate to Hazard")
        put_float_var("prob_contrib", prob_contrib, "Prob contribution rate to Hazard")
        put_float_var("sdii50_contrib", sdii50_contrib, "SDII50 contribution rate to Hazard")
        put_float_var("pop_contrib", pop_contrib, "population density contribution rate to Exposure")
        put_float_var("gdp_contrib", gdp_contrib, "GDP density contribution rate to Exposure")
        put_float_var("cropland_contrib", cropland_contrib, "cropland ratio contribution rate to Exposure")
        put_float_var("builtup_contrib", builtup_contrib, "built-up ratio contribution rate to Exposure")

        vdriver = ds.createVariable(
            "driver_type",
            "i2",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.int16(-1),
        )
        vdriver[:] = np.asarray(driver_type, dtype=np.int16)
        vdriver.long_name = "dominant driver type from Hazard-Exposure matrix"
        vdriver.units = "1"
        vdriver.flag_values = "0, 1, 2, 3"
        vdriver.flag_meanings = "general hazard_dominated exposure_dominated dual_high"

        ds.description = "USRI attribution using Hazard-Exposure driver matrix and component contribution rates"
        ds.driver_type_0 = "一般风险区"
        ds.driver_type_1 = "危险性主导型"
        ds.driver_type_2 = "暴露度主导型"
        ds.driver_type_3 = "双高驱动型"
        ds.high_percentile = float(high_percentile)
        ds.hazard_high_threshold = float(hazard_threshold)
        ds.exposure_high_threshold = float(exposure_threshold)

    print(f"[usri] USRI 归因 nc 输出 -> {out_path}")
    print(f"[usri] Hazard 高值阈值(P{high_percentile:g}) -> {hazard_threshold:.6f}")
    print(f"[usri] Exposure 高值阈值(P{high_percentile:g}) -> {exposure_threshold:.6f}")
    return str(out_path)


def compute_usri_attribution(
    hazard_nc: str | Path = DEFAULT_HAZARD_NC,
    exposure_nc: str | Path = DEFAULT_EXPOSURE_NC,
    usri_nc: str | Path = DEFAULT_USRI_NC,
    out_nc_path: str | Path = DEFAULT_ATTRIBUTION_OUT_NC,
    high_percentile: float = 75.0,
    hazard_weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    exposure_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.15, 0.15),
) -> str:
    """
    计算 USRI 归因结果：
    1) 基于 Hazard/Exposure 高低组合划分主导类型；
    2) 计算 Hazard 和 Exposure 内部各因子的贡献率。
    """
    lats, lons, hazard = _read_2d_metric_nc(hazard_nc, "hazard")
    _, _, norm_r99p = _read_2d_metric_nc(hazard_nc, "norm_r99p")
    _, _, norm_prob = _read_2d_metric_nc(hazard_nc, "norm_prob")
    _, _, norm_sdii50 = _read_2d_metric_nc(hazard_nc, "norm_sdii50")

    exposure_lats, exposure_lons, exposure = _read_2d_metric_nc(exposure_nc, "exposure")
    _check_same_grid(lats, lons, exposure_lats, exposure_lons, "Hazard 文件", "Exposure 文件")
    _, _, norm_pop_density = _read_2d_metric_nc(exposure_nc, "norm_pop_density")
    _, _, norm_gdp_density = _read_2d_metric_nc(exposure_nc, "norm_gdp_density")
    _, _, norm_cropland_ratio = _read_2d_metric_nc(exposure_nc, "norm_cropland_ratio")
    _, _, norm_builtup_ratio = _read_2d_metric_nc(exposure_nc, "norm_builtup_ratio")

    usri_lats, usri_lons, usri = _read_2d_metric_nc(usri_nc, "usri")
    _check_same_grid(lats, lons, usri_lats, usri_lons, "Hazard 文件", "USRI 文件")

    valid_mask = np.isfinite(hazard) & np.isfinite(exposure) & np.isfinite(usri)
    if not valid_mask.any():
        raise ValueError("没有有效格点，无法计算 USRI 归因。")

    hazard_threshold = float(np.nanpercentile(hazard[valid_mask], high_percentile))
    exposure_threshold = float(np.nanpercentile(exposure[valid_mask], high_percentile))
    hazard_high = hazard >= hazard_threshold
    exposure_high = exposure >= exposure_threshold

    driver_type = np.full(hazard.shape, -1, dtype=np.int16)
    driver_type[valid_mask] = 0
    driver_type[valid_mask & hazard_high & ~exposure_high] = 1
    driver_type[valid_mask & ~hazard_high & exposure_high] = 2
    driver_type[valid_mask & hazard_high & exposure_high] = 3

    r99p_contrib = _safe_contribution(hazard_weights[0] * norm_r99p, hazard)
    prob_contrib = _safe_contribution(hazard_weights[1] * norm_prob, hazard)
    sdii50_contrib = _safe_contribution(hazard_weights[2] * norm_sdii50, hazard)
    pop_contrib = _safe_contribution(exposure_weights[0] * norm_pop_density, exposure)
    gdp_contrib = _safe_contribution(exposure_weights[1] * norm_gdp_density, exposure)
    cropland_contrib = _safe_contribution(exposure_weights[2] * norm_cropland_ratio, exposure)
    builtup_contrib = _safe_contribution(exposure_weights[3] * norm_builtup_ratio, exposure)

    return write_usri_attribution_nc(
        out_nc_path=out_nc_path,
        lats=lats,
        lons=lons,
        hazard=hazard,
        exposure=exposure,
        usri=usri,
        driver_type=driver_type,
        r99p_contrib=r99p_contrib,
        prob_contrib=prob_contrib,
        sdii50_contrib=sdii50_contrib,
        pop_contrib=pop_contrib,
        gdp_contrib=gdp_contrib,
        cropland_contrib=cropland_contrib,
        builtup_contrib=builtup_contrib,
        hazard_threshold=hazard_threshold,
        exposure_threshold=exposure_threshold,
        high_percentile=high_percentile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 USRI Hazard、Exposure 或综合 USRI 指标。")
    parser.add_argument(
        "--mode",
        choices=("hazard", "exposure", "usri", "attribution", "dem-stations"),
        default="hazard",
        help="计算或绘图模式",
    )
    parser.add_argument("--daily-precip-nc", default=DEFAULT_DAILY_PRECIP_NC, help="合并后的逐日降水 nc")
    parser.add_argument("--r99p-nc", default=DEFAULT_R99P_NC, help="包含 r99p_threshold 的 nc")
    parser.add_argument("--out-nc", default=DEFAULT_OUT_NC, help="输出 Hazard nc")
    parser.add_argument("--r99p-var", default="r99p_threshold", help="R99p 阈值变量名")
    parser.add_argument("--precip-var", default="prep", help="逐日降水变量名")
    parser.add_argument("--storm-threshold-mm", type=float, default=50.0, help="暴雨日阈值，默认 50 mm")
    parser.add_argument("--time-chunk-size", type=int, default=32, help="逐日降水按时间分块读取大小")
    parser.add_argument("--reference-grid-nc", default=DEFAULT_DAILY_PRECIP_NC, help="Exposure 使用的目标 500 m 网格 nc")
    parser.add_argument("--gis-data-dir", default=DEFAULT_GIS_DATA_DIR, help="人口/GDP/地表覆盖压缩包所在目录")
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR, help="压缩包解压输出目录")
    parser.add_argument("--exposure-out-nc", default=DEFAULT_EXPOSURE_OUT_NC, help="输出 Exposure nc")
    parser.add_argument("--hazard-nc", default=DEFAULT_HAZARD_NC, help="USRI 计算使用的 Hazard nc")
    parser.add_argument("--exposure-nc", default=DEFAULT_EXPOSURE_NC, help="USRI 计算使用的 Exposure nc")
    parser.add_argument("--usri-out-nc", default=DEFAULT_USRI_OUT_NC, help="输出 USRI nc")
    parser.add_argument("--usri-nc", default=DEFAULT_USRI_NC, help="归因分析使用的 USRI nc")
    parser.add_argument("--attribution-out-nc", default=DEFAULT_ATTRIBUTION_OUT_NC, help="输出 USRI 归因 nc")
    parser.add_argument("--high-percentile", type=float, default=75.0, help="Hazard/Exposure 高值阈值百分位")
    parser.add_argument("--dem-nc", default=DEFAULT_DEM_NC, help="DEM nc 文件")
    parser.add_argument("--station-csv", default=DEFAULT_STATION_OBS_CSV, help="站点观测 CSV 文件")
    parser.add_argument("--dem-station-out-png", default=DEFAULT_DEM_STATION_OUT_PNG, help="输出 DEM 与站点分布图")
    parser.add_argument("--landcover-tif", default=DEFAULT_LANDCOVER_TIF, help="可选土地覆盖 tif；默认不指定，使用原 GlobeLand30 数据")
    parser.add_argument("--province-shp", default=DEFAULT_PROVINCE_SHP_PATH, help="浙江省边界 shp")
    args = parser.parse_args()

    if args.mode == "hazard":
        compute_hazard_index(
            daily_precip_nc=args.daily_precip_nc,
            r99p_nc=args.r99p_nc,
            out_nc_path=args.out_nc,
            r99p_var=args.r99p_var,
            precip_var=args.precip_var,
            storm_threshold_mm=args.storm_threshold_mm,
            time_chunk_size=args.time_chunk_size,
        )
    elif args.mode == "exposure":
        compute_exposure_index(
            reference_grid_nc=args.reference_grid_nc,
            gis_data_dir=args.gis_data_dir,
            extract_dir=args.extract_dir,
            out_nc_path=args.exposure_out_nc,
            province_shp_path=args.province_shp,
            landcover_tif=args.landcover_tif,
        )
    elif args.mode == "usri":
        compute_usri_index(
            hazard_nc=args.hazard_nc,
            exposure_nc=args.exposure_nc,
            out_nc_path=args.usri_out_nc,
        )
    elif args.mode == "attribution":
        compute_usri_attribution(
            hazard_nc=args.hazard_nc,
            exposure_nc=args.exposure_nc,
            usri_nc=args.usri_nc,
            out_nc_path=args.attribution_out_nc,
            high_percentile=args.high_percentile,
        )
    else:
        plot_dem_with_station_distribution(
            dem_nc_path=args.dem_nc,
            station_csv=args.station_csv,
            out_png=args.dem_station_out_png,
            province_shp_path=args.province_shp,
        )


if __name__ == "__main__":
    main()
   