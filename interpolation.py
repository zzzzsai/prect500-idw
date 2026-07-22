from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
#obs_screened = "out/station_obs_screened_95pct_5to9.csv"
#valid_dates = "out/valid_station_daily_counts_screened.csv"
obs_screened = "/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv"
valid_dates = "newout/valid_station_daily_counts_2016_2025.csv"
dem_nc = "static/dem_zhejiang_500m_merged.nc"
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"

_WORKER_CONTEXT: dict[str, object] = {}



def load_target_dates(
    valid_dates_csv: str,
    min_station_count: int = 500,
    date_col: str = "date",
    count_col: str = "valid_station_count",
) -> pd.DatetimeIndex:
    df = pd.read_csv(valid_dates_csv, usecols=[date_col, count_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[df[count_col] >= min_station_count].copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.DatetimeIndex(df[date_col].dt.normalize().unique())
    print(
        f"[interp] 从 {valid_dates_csv} 中筛出 {len(dates)} 个日期，"
        f"要求 {count_col} >= {min_station_count}"
    )
    return dates


def load_dem_grid(
    dem_nc_path: str,
    lat_name: str = "lat",
    lon_name: str = "lon",
    dem_name: str = "dem",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(dem_nc_path, "r") as ds:
        lats = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        lons = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        dem = np.asarray(ds.variables[dem_name][:], dtype=np.float64)
    if lats.ndim != 1 or lons.ndim != 1 or dem.ndim != 2:
        raise ValueError("DEM NetCDF 必须包含 lat(1D), lon(1D), dem(2D) 结构。")
    if dem.shape != (len(lats), len(lons)):
        raise ValueError("DEM 形状与 lat/lon 尺寸不匹配。")
    return lats, lons, dem


def build_dem_sampler(
    lats: np.ndarray,
    lons: np.ndarray,
    dem: np.ndarray,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (lats, lons),
        dem,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def sample_dem_at_points(
    sampler: RegularGridInterpolator,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    pts = np.column_stack([lats, lons])
    elev = sampler(pts)
    elev = np.asarray(elev, dtype=np.float64)
    elev[~np.isfinite(elev)] = np.nan
    return elev


def geographic_xyz(
    lons: np.ndarray,
    lats: np.ndarray,
    elev_m: np.ndarray,
    ref_lat: float,
) -> np.ndarray:
    lat_km = lats * 110.574
    lon_km = lons * (111.320 * np.cos(np.deg2rad(ref_lat)))
    elev_km = elev_m / 1000.0
    return np.column_stack([lon_km, lat_km, elev_km]).astype(np.float64)


def build_province_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    province_shp_path: str = DEFAULT_PROVINCE_SHP_PATH,
    province_name: str = "浙江省",
) -> np.ndarray:
    """
    根据省界 shp 在目标规则网格上生成布尔掩膜。

    返回二维 mask，省界内(含边界附近)为 True，外部为 False。
    """
    import geopandas as gpd

    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == province_name]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到 {province_name} 边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    # 给边界一个极小缓冲，避免刚好落在边界上的格点被误判为省外。
    geom = geom.buffer(1e-6)

    lon2d, lat2d = np.meshgrid(lons, lats)
    try:
        from shapely import contains_xy

        mask = contains_xy(geom, lon2d, lat2d)
    except Exception:
        from shapely.vectorized import contains

        mask = contains(geom, lon2d, lat2d)
    return np.asarray(mask, dtype=bool)


def load_observations_for_dates(
    obs_csv: str,
    target_dates: pd.DatetimeIndex,
    sampler: RegularGridInterpolator,
    station_col: str = "stationId",
    date_col: str = "datetime",
    lat_col: str = "lat",
    lon_col: str = "lng",
    value_col: str = "r2020",
    chunksize: int = 300_000,
) -> pd.DataFrame:
    target_set = set(target_dates.normalize())
    usecols = [station_col, date_col, lat_col, lon_col, value_col]
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        obs_csv,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk[lat_col] = pd.to_numeric(chunk[lat_col], errors="coerce")
        chunk[lon_col] = pd.to_numeric(chunk[lon_col], errors="coerce")
        chunk = chunk.dropna(subset=[date_col, lat_col, lon_col, value_col])
        chunk["date"] = chunk[date_col].dt.normalize()
        chunk = chunk[chunk["date"].isin(target_set)].copy()
        if chunk.empty:
            continue
        chunk = chunk[np.isfinite(chunk[value_col]) & (chunk[value_col] >= 0)].copy()
        if chunk.empty:
            continue
        chunk["elev"] = sample_dem_at_points(
            sampler,
            chunk[lat_col].to_numpy(),
            chunk[lon_col].to_numpy(),
        )
        chunk = chunk[np.isfinite(chunk["elev"])].copy()
        if chunk.empty:
            continue
        parts.append(chunk[[station_col, "date", lat_col, lon_col, "elev", value_col]].copy())

    if not parts:
        raise ValueError("筛选后没有可用于插值的观测记录。")

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=[station_col, "date"], keep="last").reset_index(drop=True)
    print(f"[interp] 载入可插值观测 {len(out):,} 条，覆盖 {out['date'].nunique()} 天")
    return out


def fit_tps_for_day(
    day_obs: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    elev_col: str,
    value_col: str,
    ref_lat: float,
    smoothing: float = 0.0,
    neighbors: int | None = 256,
) -> tuple[RBFInterpolator, pd.DataFrame]:
    collapsed = (
        day_obs.groupby([lon_col, lat_col, elev_col], as_index=False)[value_col]
        .mean()
        .sort_values([lon_col, lat_col, elev_col])
        .reset_index(drop=True)
    )
    xyz = geographic_xyz(
        collapsed[lon_col].to_numpy(),
        collapsed[lat_col].to_numpy(),
        collapsed[elev_col].to_numpy(),
        ref_lat=ref_lat,
    )
    values = collapsed[value_col].to_numpy(dtype=np.float64)
    model = RBFInterpolator(
        xyz,
        values,
        kernel="thin_plate_spline",
        degree=1,
        smoothing=smoothing,
        neighbors=neighbors,
    )
    return model, collapsed


def write_daily_output_nc(
    out_nc_path: str,
    date: pd.Timestamp,
    lats: np.ndarray,
    lons: np.ndarray,
) -> nc.Dataset:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = nc.Dataset(out_path, "w")
    ds.createDimension("time", 1)
    ds.createDimension("y", len(lats))
    ds.createDimension("x", len(lons))

    vtime = ds.createVariable("time", "i4", ("time",))
    vtime.units = "days since 1970-01-01 00:00:00"
    vtime.calendar = "standard"
    vtime[:] = nc.date2num([date.to_pydatetime()], units=vtime.units, calendar=vtime.calendar)

    vlat = ds.createVariable("lat", "f4", ("y",))
    vlon = ds.createVariable("lon", "f4", ("x",))
    vlat[:] = lats.astype(np.float32)
    vlon[:] = lons.astype(np.float32)

    vprep = ds.createVariable(
        "prep",
        "f4",
        ("time", "y", "x"),
        zlib=True,
        complevel=4,
        chunksizes=(1, min(len(lats), 200), min(len(lons), 200)),
        fill_value=np.float32(np.nan),
    )
    vprep.long_name = "daily precipitation interpolated by thin plate spline"
    vprep.units = "mm/day"
    return ds


def _init_interp_worker(
    lats: np.ndarray,
    lons: np.ndarray,
    grid_xyz: np.ndarray,
    province_mask: np.ndarray | None,
    ref_lat: float,
    out_dir: str,
    lon_col: str,
    lat_col: str,
    value_col: str,
    smoothing: float,
    neighbors: int | None,
    eval_batch_size: int,
    clip_negative_to_zero: bool,
) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "lats": lats,
            "lons": lons,
            "grid_xyz": grid_xyz,
            "province_mask": province_mask,
            "ref_lat": ref_lat,
            "out_dir": out_dir,
            "lon_col": lon_col,
            "lat_col": lat_col,
            "value_col": value_col,
            "smoothing": smoothing,
            "neighbors": neighbors,
            "eval_batch_size": eval_batch_size,
            "clip_negative_to_zero": clip_negative_to_zero,
        }
    )


def _interpolate_one_day(task: tuple[int, int, pd.Timestamp, pd.DataFrame]) -> str:
    i, total, day, day_obs = task
    ctx = _WORKER_CONTEXT
    day = pd.Timestamp(day).normalize()

    candidate_count = len(day_obs)
    if candidate_count == 0:
        return f"[interp] {day.date()} 无有效观测，跳过"

    lon_col = ctx["lon_col"]
    lat_col = ctx["lat_col"]
    value_col = ctx["value_col"]

    approx_station_count = day_obs[[lon_col, lat_col]].drop_duplicates().shape[0]
    if approx_station_count < 4:
        return f"[interp] {day.date()} 有效独立站点少于 4 个，跳过"

    model, collapsed = fit_tps_for_day(
        day_obs,
        lon_col=lon_col,
        lat_col=lat_col,
        elev_col="elev",
        value_col=value_col,
        ref_lat=ctx["ref_lat"],
        smoothing=ctx["smoothing"],
        neighbors=ctx["neighbors"],
    )
    if len(collapsed) < 4:
        return f"[interp] {day.date()} 有效独立站点少于 4 个，跳过"

    grid_xyz = ctx["grid_xyz"]
    grid_values = np.empty(grid_xyz.shape[0], dtype=np.float32)
    for j in range(0, grid_xyz.shape[0], ctx["eval_batch_size"]):
        k = min(j + ctx["eval_batch_size"], grid_xyz.shape[0])
        pred = model(grid_xyz[j:k]).astype(np.float32)
        if ctx["clip_negative_to_zero"]:
            pred = np.maximum(pred, 0.0)
        grid_values[j:k] = pred

    out_nc_path = Path(ctx["out_dir"]) / f"prep_{day.strftime('%Y%m%d')}.nc"
    ds_out = write_daily_output_nc(str(out_nc_path), day, ctx["lats"], ctx["lons"])
    try:
        grid_2d = grid_values.reshape(len(ctx["lats"]), len(ctx["lons"]))
        province_mask = ctx.get("province_mask")
        if province_mask is not None:
            grid_2d = grid_2d.copy()
            grid_2d[~province_mask] = np.nan
        ds_out.variables["prep"][0, :, :] = grid_2d
    finally:
        ds_out.close()

    return (
        f"[interp] {i}/{total} {day.date()} 完成，"
        f"候选观测数={candidate_count}，独立站点={len(collapsed)} -> {out_nc_path.name}"
    )


def interpolate_daily_precip_to_dem_grid(
    obs_csv: str = obs_screened,
    valid_dates_csv: str = valid_dates,
    dem_nc_path: str = dem_nc,
    out_dir: str = "out/daily_precip_tps_500m",
    min_station_count: int = 500,
    num_workers: int | None = None,
    interp_start_date: str | None = None,
    interp_end_date: str | None = None,
    station_col: str = "stationId",
    obs_date_col: str = "datetime",
    valid_date_col: str = "date",
    lat_col: str = "lat",
    lon_col: str = "lng",
    value_col: str = "r2020",
    valid_count_col: str = "valid_station_count",
    chunksize: int = 300_000,
    smoothing: float = 0.0,
    neighbors: int | None = 256,
    eval_batch_size: int = 200_000,
    max_dates: int | None = None,
    clip_negative_to_zero: bool = True,
    apply_province_mask: bool = True,
    province_shp_path: str = DEFAULT_PROVINCE_SHP_PATH,
    skip_existing: bool = True,
) -> str:
    """
    按天执行基于 DEM 协变量的薄板样条插值，并将每个日期写成一个 NetCDF 文件。

    可通过 `interp_start_date` / `interp_end_date` 限定实际插值日期范围，
    例如只处理 `2016-05-01` 到 `2016-06-01`。

    说明：
    - 这里实现的是 Python/Scipy 版的 ANUSPLIN 思路：以 `(lon, lat, elev)` 为自变量，
      使用 thin-plate spline 在 500m DEM 网格上逐日插值。
    - `neighbors=None` 表示全局薄板样条；若日期很多、站点很多，建议保留默认局地邻域近似，
      否则计算会明显变慢。
    - `num_workers` 控制按日期并行的进程数；`None` 时自动取不超过日期数和 CPU 数的较小值。
    - 若 `apply_province_mask=True`，则会按浙江省边界将省外插值结果置为 NaN。
    - 若 `skip_existing=True`，则已存在 `prep_YYYYMMDD.nc` 的日期会自动跳过。
    """
    dates = load_target_dates(
        valid_dates_csv,
        min_station_count=min_station_count,
        date_col=valid_date_col,
        count_col=valid_count_col,
    )
    if interp_start_date is not None:
        start_ts = pd.Timestamp(interp_start_date).normalize()
        dates = dates[dates >= start_ts]
    if interp_end_date is not None:
        end_ts = pd.Timestamp(interp_end_date).normalize()
        dates = dates[dates <= end_ts]
    if interp_start_date is not None or interp_end_date is not None:
        print(
            f"[interp] 日期范围筛选后剩余 {len(dates)} 天，"
            f"范围=({interp_start_date or '-inf'} ~ {interp_end_date or '+inf'})"
        )
    if max_dates is not None:
        dates = dates[:max_dates]
        print(f"[interp] 仅处理前 {len(dates)} 个日期用于试运行")
    if len(dates) == 0:
        raise ValueError("没有满足站点数阈值的日期。")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        pending_dates = []
        skipped_existing = 0
        for day in dates:
            out_nc_path = out_dir_path / f"prep_{pd.Timestamp(day).strftime('%Y%m%d')}.nc"
            if out_nc_path.exists():
                skipped_existing += 1
                print(f"[interp] {pd.Timestamp(day).date()} 已存在 {out_nc_path.name}，跳过")
            else:
                pending_dates.append(day)
        if skipped_existing:
            print(f"[interp] 已跳过 {skipped_existing} 天已存在的 nc 文件")
        dates = pd.DatetimeIndex(pending_dates)
        if len(dates) == 0:
            print(f"[interp] 所有目标日期的 nc 文件均已存在，无需插值 -> {out_dir_path}")
            return str(out_dir_path)

    lats, lons, dem = load_dem_grid(dem_nc_path)
    dem = dem.astype(np.float64)
    ref_lat = float(np.nanmean(lats))
    sampler = build_dem_sampler(lats, lons, dem)
    province_mask = None
    if apply_province_mask:
        province_mask = build_province_mask(lats, lons, province_shp_path=province_shp_path)
        print(
            f"[interp] 已生成浙江省边界掩膜，省内格点数={int(province_mask.sum())}/"
            f"{province_mask.size}"
        )

    obs = load_observations_for_dates(
        obs_csv,
        dates,
        sampler,
        station_col=station_col,
        date_col=obs_date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        value_col=value_col,
        chunksize=chunksize,
    )

    lon2d, lat2d = np.meshgrid(lons, lats)
    grid_xyz = geographic_xyz(lon2d.ravel(), lat2d.ravel(), dem.ravel(), ref_lat=ref_lat)

    date_to_indices = obs.groupby("date", sort=False).indices
    tasks: list[tuple[int, int, pd.Timestamp, pd.DataFrame]] = []
    total = len(dates)
    for i, day in enumerate(dates, start=1):
        idx = date_to_indices.get(day)
        if idx is None:
            print(f"[interp] {day.date()} 无有效观测，跳过")
            continue
        day_obs = obs.iloc[idx].copy()
        print(f"[interp] {day.date()} 当天候选观测数={len(day_obs)}")
        tasks.append((i, total, day, day_obs))

    if not tasks:
        raise ValueError("日期筛选后没有可执行的插值任务。")

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, min(cpu_count, len(tasks)))
    else:
        num_workers = max(1, min(int(num_workers), len(tasks)))

    print(f"[interp] 按日期并行插值，worker 数={num_workers}，任务数={len(tasks)}")

    _init_interp_worker(
        lats=lats,
        lons=lons,
        grid_xyz=grid_xyz,
        province_mask=province_mask,
        ref_lat=ref_lat,
        out_dir=str(out_dir_path),
        lon_col=lon_col,
        lat_col=lat_col,
        value_col=value_col,
        smoothing=smoothing,
        neighbors=neighbors,
        eval_batch_size=eval_batch_size,
        clip_negative_to_zero=clip_negative_to_zero,
    )

    if num_workers == 1:
        for task in tasks:
            print(_interpolate_one_day(task))
    else:
        try:
            mp_context = mp.get_context("fork")
        except ValueError:
            mp_context = mp.get_context()
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_context,
            initializer=_init_interp_worker,
            initargs=(
                lats,
                lons,
                grid_xyz,
                province_mask,
                ref_lat,
                str(out_dir_path),
                lon_col,
                lat_col,
                value_col,
                smoothing,
                neighbors,
                eval_batch_size,
                clip_negative_to_zero,
            ),
        ) as executor:
            futures = [executor.submit(_interpolate_one_day, task) for task in tasks]
            for future in as_completed(futures):
                print(future.result())

    print(f"[interp] 插值完成 -> {out_dir_path}")
    return str(out_dir_path)


def _date_from_daily_precip_path(
    nc_path: str | Path,
    file_prefix: str = "prep_",
    date_format: str = "%Y%m%d",
) -> pd.Timestamp | None:
    stem = Path(nc_path).stem
    if not stem.startswith(file_prefix):
        return None
    date_text = stem[len(file_prefix):]
    try:
        return pd.Timestamp(pd.to_datetime(date_text, format=date_format)).normalize()
    except ValueError:
        return None


def merge_daily_precip_nc_files(
    daily_nc_dir: str | Path = "newout/daily_precip_tps_500m",
    out_nc_path: str | Path = "newout/daily_precip_tps_500m/prep_daily_2016_2025.nc",
    start_date: str = "2016-01-01",
    end_date: str = "2025-12-31",
    daily_nc_glob: str = "prep_*.nc",
    var_name: str = "prep",
    file_prefix: str = "prep_",
    require_complete: bool = False,
    compression_level: int = 4,
) -> str:
    """
    将逐日降水 NetCDF(`prep_YYYYMMDD.nc`)按时间合并为一个 NetCDF 文件。

    默认合并 2016-2025 年范围内已经存在的有效日期文件；若
    `require_complete=True`，则要求日期范围内每一天的文件都存在。
    """
    daily_dir = Path(daily_nc_dir)
    out_path = Path(out_nc_path)
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("start_date 不能晚于 end_date。")

    dated_files: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(daily_dir.glob(daily_nc_glob)):
        if path.resolve() == out_path.resolve():
            continue
        day = _date_from_daily_precip_path(path, file_prefix=file_prefix)
        if day is None or day < start_ts or day > end_ts:
            continue
        dated_files.append((day, path))

    if not dated_files:
        raise ValueError(f"在 {daily_dir} 中没有找到 {start_ts.date()} 到 {end_ts.date()} 的逐日 nc 文件。")

    dates = pd.DatetimeIndex([day for day, _ in dated_files])
    if dates.has_duplicates:
        duplicate_dates = dates[dates.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"发现重复日期文件：{duplicate_dates}")

    dated_files = sorted(dated_files, key=lambda item: item[0])
    dates = pd.DatetimeIndex([day for day, _ in dated_files])
    if require_complete:
        expected_dates = pd.date_range(start_ts, end_ts, freq="D")
        missing_dates = expected_dates.difference(dates)
        if len(missing_dates) > 0:
            sample = ", ".join(missing_dates[:10].strftime("%Y-%m-%d"))
            suffix = " ..." if len(missing_dates) > 10 else ""
            raise ValueError(f"缺少 {len(missing_dates)} 天逐日 nc 文件：{sample}{suffix}")

    first_path = dated_files[0][1]
    with nc.Dataset(first_path, "r") as first_ds:
        if var_name not in first_ds.variables:
            raise ValueError(f"{first_path} 不包含变量 {var_name!r}")
        if "lat" not in first_ds.variables or "lon" not in first_ds.variables:
            raise ValueError(f"{first_path} 缺少 lat/lon 坐标变量")
        lats = np.asarray(first_ds.variables["lat"][:], dtype=np.float32)
        lons = np.asarray(first_ds.variables["lon"][:], dtype=np.float32)
        first_var = first_ds.variables[var_name]
        var_attrs = {
            attr: first_var.getncattr(attr)
            for attr in first_var.ncattrs()
            if attr != "_FillValue"
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as out_ds:
        out_ds.createDimension("time", len(dated_files))
        out_ds.createDimension("y", len(lats))
        out_ds.createDimension("x", len(lons))

        vtime = out_ds.createVariable("time", "i4", ("time",))
        vtime.units = "days since 1970-01-01 00:00:00"
        vtime.calendar = "standard"
        vtime[:] = nc.date2num(
            [day.to_pydatetime() for day in dates],
            units=vtime.units,
            calendar=vtime.calendar,
        )

        vlat = out_ds.createVariable("lat", "f4", ("y",))
        vlon = out_ds.createVariable("lon", "f4", ("x",))
        vlat[:] = lats
        vlon[:] = lons

        vprep = out_ds.createVariable(
            var_name,
            "f4",
            ("time", "y", "x"),
            zlib=True,
            complevel=compression_level,
            chunksizes=(1, min(len(lats), 200), min(len(lons), 200)),
            fill_value=np.float32(np.nan),
        )
        for attr, value in var_attrs.items():
            vprep.setncattr(attr, value)

        out_ds.title = "daily precipitation merged from daily TPS interpolation outputs"
        out_ds.source_directory = str(daily_dir)
        out_ds.source_file_count = len(dated_files)
        out_ds.date_range = f"{dates[0].date()} to {dates[-1].date()}"

        for i, (day, path) in enumerate(dated_files):
            with nc.Dataset(path, "r") as src_ds:
                if var_name not in src_ds.variables:
                    raise ValueError(f"{path} 不包含变量 {var_name!r}")
                src_lats = np.asarray(src_ds.variables["lat"][:], dtype=np.float32)
                src_lons = np.asarray(src_ds.variables["lon"][:], dtype=np.float32)
                if src_lats.shape != lats.shape or src_lons.shape != lons.shape:
                    raise ValueError(f"{path} 的 lat/lon 尺寸与首个文件不一致。")
                if not np.allclose(src_lats, lats, equal_nan=True) or not np.allclose(src_lons, lons, equal_nan=True):
                    raise ValueError(f"{path} 的 lat/lon 坐标与首个文件不一致。")

                prep = src_ds.variables[var_name][:]
                if np.ma.isMaskedArray(prep):
                    prep = prep.filled(np.nan)
                prep = np.asarray(prep, dtype=np.float32)
                if prep.ndim == 3:
                    if prep.shape[0] != 1:
                        raise ValueError(f"{path} 中 {var_name!r} 的 time 维度不是 1。")
                    prep = prep[0, :, :]
                elif prep.ndim != 2:
                    raise ValueError(f"{path} 中 {var_name!r} 维度应为 2D 或 3D(time,y,x)，实际为 {prep.shape}")
                if prep.shape != (len(lats), len(lons)):
                    raise ValueError(f"{path} 中 {var_name!r} 形状与 lat/lon 尺寸不一致。")
                vprep[i, :, :] = prep

            if (i + 1) % 100 == 0 or i == len(dated_files) - 1:
                print(f"[merge] {i + 1}/{len(dated_files)} {day.date()} -> {out_path}")

    print(f"[merge] 合并完成：{len(dated_files)} 天 -> {out_path}")
    return str(out_path)


def plot_daily_precip_distribution_from_merged_nc(
    nc_path: str | Path = "newout/daily_precip_tps_500m/prep_daily_2016_2025.nc",
    target_date: str = "2024-09-18",
    out_png_path: str | Path | None = None,
    var_name: str = "prep",
    cmap: str = "Blues",
    title: str | None = None,
    vmax_percentile: float = 99.0,
    province_shp_path: str | None = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    """
    从合并后的逐日降水 NetCDF 中读取任意一天，并输出降水空间分布图。
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
    if province_shp_path is not None:
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


def plot_precip_comparison_from_two_nc(
    nc_path_a: str,
    nc_path_b: str,
    out_png_path: str = "out/compare_two_nc_precip.png",
    out_diff_png_path: str | None = None,
    out_stats_csv_path: str | None = None,
    var_name: str = "prep",
    cmap: str = "Blues",
    diff_cmap: str = "RdBu_r",
    title_a: str | None = None,
    title_b: str | None = None,
) -> dict[str, str]:
    """
    输入两个降水 NetCDF，自动输出：
    1) 并排对比图；2) 差值图(A-B)；3) 统计指标(打印并写 CSV)。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _read_nc_2d(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with nc.Dataset(path, "r") as ds:
            if var_name not in ds.variables:
                raise ValueError(f"{path} 不包含变量 {var_name!r}")
            if "lat" not in ds.variables or "lon" not in ds.variables:
                raise ValueError(f"{path} 缺少 lat/lon 坐标变量")
            lats_ = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lons_ = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            prep_ = np.asarray(ds.variables[var_name][:], dtype=np.float64)

        if prep_.ndim == 3:
            prep_ = prep_[0, :, :]
        elif prep_.ndim != 2:
            raise ValueError(f"{path} 中 {var_name!r} 维度应为 2D 或 3D(time,y,x)，实际为 {prep_.shape}")

        if prep_.shape != (len(lats_), len(lons_)):
            raise ValueError(f"{path} 中 {var_name!r} 形状与 lat/lon 尺寸不一致")
        return lats_, lons_, prep_

    lats_a, lons_a, prep_a = _read_nc_2d(nc_path_a)
    lats_b, lons_b, prep_b = _read_nc_2d(nc_path_b)

    if len(lats_a) != len(lats_b) or len(lons_a) != len(lons_b):
        raise ValueError("两个 nc 网格尺寸不同，无法直接对比。")
    if not np.allclose(lats_a, lats_b, rtol=0.0, atol=1e-8) or not np.allclose(lons_a, lons_b, rtol=0.0, atol=1e-8):
        raise ValueError("两个 nc 的网格坐标不同，无法直接对比。")

    a_valid = prep_a[np.isfinite(prep_a)]
    b_valid = prep_b[np.isfinite(prep_b)]
    if len(a_valid) == 0 and len(b_valid) == 0:
        raise ValueError("两个 nc 都没有有效降水值，无法绘图。")
    combined = np.concatenate([x for x in (a_valid, b_valid) if len(x) > 0])
    vmin = 0.0
    vmax = float(np.nanpercentile(combined, 99.0))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(combined))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    extent_a = [float(lons_a.min()), float(lons_a.max()), float(lats_a.min()), float(lats_a.max())]
    extent_b = [float(lons_b.min()), float(lons_b.max()), float(lats_b.min()), float(lats_b.max())]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=180, constrained_layout=True)
    im_a = axes[0].imshow(
        prep_a,
        origin="lower",
        extent=extent_a,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[0].set_title(title_a or Path(nc_path_a).name)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    axes[1].imshow(
        prep_b,
        origin="lower",
        extent=extent_b,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axes[1].set_title(title_b or Path(nc_path_b).name)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    cbar = fig.colorbar(im_a, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("precipitation (mm/day)")

    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    diff = prep_a - prep_b
    valid_mask = np.isfinite(diff)
    valid_diff = diff[valid_mask]
    if len(valid_diff) == 0:
        raise ValueError("两个 nc 没有重叠有效像元，无法生成差值图和统计指标。")

    max_abs = float(np.nanpercentile(np.abs(valid_diff), 99.0))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = float(np.nanmax(np.abs(valid_diff)))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    if out_diff_png_path is None:
        out_diff_png_path = str(out_path.with_name(f"{out_path.stem}_diff{out_path.suffix}"))
    diff_path = Path(out_diff_png_path)
    diff_path.parent.mkdir(parents=True, exist_ok=True)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6.8, 5.2), dpi=180, constrained_layout=True)
    im2 = ax2.imshow(
        diff,
        origin="lower",
        extent=extent_a,
        cmap=diff_cmap,
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto",
    )
    ax2.set_title(f"Difference: {(title_a or Path(nc_path_a).name)} - {(title_b or Path(nc_path_b).name)}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.9, pad=0.02)
    cbar2.set_label("precipitation difference (mm/day)")
    fig2.savefig(diff_path, bbox_inches="tight")
    plt.close(fig2)

    stats = {
        "nc_a": str(Path(nc_path_a)),
        "nc_b": str(Path(nc_path_b)),
        "valid_pixel_count_a": int(np.isfinite(prep_a).sum()),
        "valid_pixel_count_b": int(np.isfinite(prep_b).sum()),
        "overlap_valid_pixel_count": int(valid_mask.sum()),
        "mean_a": float(np.nanmean(prep_a)),
        "mean_b": float(np.nanmean(prep_b)),
        "mean_diff_a_minus_b": float(np.nanmean(valid_diff)),
        "mae": float(np.nanmean(np.abs(valid_diff))),
        "rmse": float(np.sqrt(np.nanmean(valid_diff ** 2))),
        "max_abs_diff": float(np.nanmax(np.abs(valid_diff))),
        "p95_abs_diff": float(np.nanpercentile(np.abs(valid_diff), 95.0)),
        "corr": float(np.corrcoef(prep_a[valid_mask], prep_b[valid_mask])[0, 1]) if int(valid_mask.sum()) >= 2 else np.nan,
    }
    stats_df = pd.DataFrame([stats])
    stats_long_df = pd.DataFrame(
        {
            "metric": list(stats.keys()),
            "value": list(stats.values()),
        }
    )

    if out_stats_csv_path is None:
        out_stats_csv_path = str(out_path.with_name(f"{out_path.stem}_stats.csv"))
    stats_path = Path(out_stats_csv_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_long_df.to_csv(stats_path, index=False)

    print(f"[interp] 两个 nc 降水并排图已输出 -> {out_path}")
    print(f"[interp] 两个 nc 降水差值图已输出 -> {diff_path}")
    print(f"[interp] 两个 nc 对比统计已输出 -> {stats_path}")
    print(stats_df.to_string(index=False))
    return {
        "compare_png": str(out_path),
        "diff_png": str(diff_path),
        "stats_csv": str(stats_path),
    }


if __name__ == "__main__":
    #todo： 留一交叉验证 RMSE / MAE 评估
    #3、绘制逐日降水nc文件的分布图
    plot_daily_precip_distribution_from_merged_nc(
        nc_path="newout/daily_precip_tps_500m/prep_daily_2016_2025.nc",
        target_date="2021-07-25",
        out_png_path="newout/plot_daily_precip_tps_500m/prep_daily_20210725.png",
    )
    exit()
    #2、合并逐日降水nc文件
    merge_daily_precip_nc_files(
        daily_nc_dir="newout/daily_precip_tps_500m",
        out_nc_path="newout/daily_precip_tps_500m/prep_daily_2016_2025.nc",
        start_date="2016-01-01",
        end_date="2025-12-31",
        daily_nc_glob="prep_*.nc",
        var_name="prep",
        file_prefix="prep_",
    )
    

    interpolate_daily_precip_to_dem_grid(
        obs_csv=obs_screened,
        valid_dates_csv=valid_dates,
        dem_nc_path=dem_nc,
        out_dir="newout/daily_precip_tps_500m",
        min_station_count=1000,
        num_workers=4,
        interp_start_date="2015-01-01",
        interp_end_date="2026-01-01",
        smoothing=0.0,
        neighbors=64,
        max_dates=None,
        apply_province_mask = True,
        skip_existing=True
    )
    exit()
    plot_precip_comparison_from_two_nc(
    nc_path_a="newout/daily_precip_tps_500m/prep_20240918.nc",
    nc_path_b="newout/daily_precip_tps_500m/prep_20160730.nc",
    out_png_path="newout/prep_20240918_compare.png",
    title_a="newout TPS",
    title_b="out TPS",
)
    exit()

    interpolate_daily_precip_to_dem_grid(
        obs_csv=obs_screened,
        valid_dates_csv=valid_dates,
        dem_nc_path=dem_nc,
        out_dir="newout/daily_precip_tps_500m",
        min_station_count=1000,
        num_workers=4,
        interp_start_date="2016-07-28",
        interp_end_date="2026-10-04",
        smoothing=0.0,
        neighbors=64,
        max_dates=None,
        apply_province_mask = True
    )
    exit()

    
    