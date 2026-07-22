from __future__ import annotations

from pathlib import Path
import re

import netCDF4 as nc
import numpy as np
import pandas as pd


DATE_PATTERN = re.compile(r"(\d{8})")
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"


def _extract_date_from_filename(path: str | Path) -> np.datetime64:
    match = DATE_PATTERN.search(Path(path).stem)
    if match is None:
        raise ValueError(f"无法从文件名提取日期: {path}")
    return np.datetime64(match.group(1), "D")


def _extract_yyyymmdd_from_filename(path: str | Path) -> str:
    match = DATE_PATTERN.search(Path(path).stem)
    if match is None:
        raise ValueError(f"无法从文件名提取日期: {path}")
    return match.group(1)


def _normalize_date_strings(dates: list[str] | set[str] | tuple[str, ...] | None) -> set[str] | None:
    if dates is None:
        return None

    normalized: set[str] = set()
    for date_str in dates:
        s = str(date_str).strip()
        if not s:
            continue
        s = s.replace("-", "")
        if len(s) != 8 or not s.isdigit():
            raise ValueError(f"日期格式不支持: {date_str}")
        normalized.add(s)
    return normalized


def _percentile_metric_label(percentile: float) -> str:
    percentile_str = f"{percentile:g}".replace(".", "p")
    return f"r{percentile_str}p"


def _metric_display_label(metric_label: str) -> str:
    return metric_label.upper().replace("P", "p")


def _read_daily_precip_nc(
    nc_path: str | Path,
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(nc_path, "r") as ds:
        lats = np.asarray(ds.variables[lat_var][:], dtype=np.float32)
        lons = np.asarray(ds.variables[lon_var][:], dtype=np.float32)
        prep = np.asarray(ds.variables[precip_var][0, :, :], dtype=np.float32)
    return lats, lons, prep


def _read_2d_metric_nc(
    nc_path: str | Path,
    var_name: str,
    lat_var: str = "lat",
    lon_var: str = "lon",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with nc.Dataset(nc_path, "r") as ds:
        lats = np.asarray(ds.variables[lat_var][:], dtype=np.float32)
        lons = np.asarray(ds.variables[lon_var][:], dtype=np.float32)
        data = np.asarray(ds.variables[var_name][:], dtype=np.float32)
    return lats, lons, data


def _write_annual_rx1day_nc(
    out_nc_path: str | Path,
    year: int,
    lats: np.ndarray,
    lons: np.ndarray,
    rx1day: np.ndarray,
) -> None:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vrx = ds.createVariable(
            "rx1day",
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)
        vrx[:] = rx1day.astype(np.float32)

        vrx.long_name = "annual maximum 1-day precipitation"
        vrx.units = "mm/day"
        ds.description = f"Rx1day for {year}"


def _write_annual_sdii_nc(
    out_nc_path: str | Path,
    year: int,
    lats: np.ndarray,
    lons: np.ndarray,
    sdii: np.ndarray,
) -> None:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vsdii = ds.createVariable(
            "sdii",
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)
        vsdii[:] = sdii.astype(np.float32)

        vsdii.long_name = "simple daily intensity index"
        vsdii.units = "mm/day"
        ds.description = f"SDII for {year}"


def _write_sdii_nc(
    out_nc_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    sdii: np.ndarray,
    description: str,
) -> None:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vsdii = ds.createVariable(
            "sdii",
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)
        vsdii[:] = sdii.astype(np.float32)

        vsdii.long_name = "simple daily intensity index"
        vsdii.units = "mm/day"
        ds.description = description


def _write_rx1day_attribution_nc(
    out_nc_path: str | Path,
    lats: np.ndarray,
    lons: np.ndarray,
    rx1day: np.ndarray,
    rx1day_date: np.ndarray,
    source_flag: np.ndarray,
    description: str,
) -> None:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vrx = ds.createVariable(
            "rx1day",
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        vdate = ds.createVariable(
            "rx1day_date",
            "i4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.int32(-9999),
        )
        vsrc = ds.createVariable(
            "rx1day_source_flag",
            "i1",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.int8(-1),
        )

        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)
        vrx[:] = rx1day.astype(np.float32)
        vdate[:] = rx1day_date.astype(np.int32)
        vsrc[:] = source_flag.astype(np.int8)

        vrx.long_name = "maximum 1-day precipitation during selected period"
        vrx.units = "mm/day"
        vdate.long_name = "date when rx1day occurred"
        vdate.units = "YYYYMMDD"
        vsrc.long_name = "source of rx1day extreme"
        vsrc.flag_values = np.array([0, 1, 2], dtype=np.int8)
        vsrc.flag_meanings = "non_typhoon typhoon both"
        ds.description = description


def _write_annual_threshold_metric_nc(
    out_nc_path: str | Path,
    year: int,
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    var_name: str,
    long_name: str,
    units: str,
) -> None:
    out_path = Path(out_nc_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(out_path, "w") as ds:
        ds.createDimension("y", len(lats))
        ds.createDimension("x", len(lons))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vvar = ds.createVariable(
            var_name,
            "f4",
            ("y", "x"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )

        vlat[:] = lats.astype(np.float32)
        vlon[:] = lons.astype(np.float32)
        vvar[:] = data.astype(np.float32)

        vvar.long_name = long_name
        vvar.units = units
        ds.description = f"{var_name} for {year}"


def _collect_daily_files_by_year(
    daily_nc_dir: str,
    daily_nc_glob: str,
    start_year: int | None = None,
    end_year: int | None = None,
    start_month: int | None = None,
    end_month: int | None = None,
    include_dates: list[str] | set[str] | tuple[str, ...] | None = None,
    exclude_dates: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[int, list[Path]]:
    daily_dir = Path(daily_nc_dir)
    nc_files = sorted(daily_dir.glob(daily_nc_glob))
    if not nc_files:
        raise FileNotFoundError(f"未在 {daily_dir} 下找到逐日降水文件: {daily_nc_glob}")

    include_date_set = _normalize_date_strings(include_dates)
    exclude_date_set = _normalize_date_strings(exclude_dates)
    files_by_year: dict[int, list[Path]] = {}
    for path in nc_files:
        yyyymmdd = _extract_yyyymmdd_from_filename(path)
        year = int(yyyymmdd[:4])
        month = int(yyyymmdd[4:6])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        if start_month is not None and end_month is not None:
            if start_month <= end_month:
                if not (start_month <= month <= end_month):
                    continue
            else:
                if not (month >= start_month or month <= end_month):
                    continue
        if include_date_set is not None and yyyymmdd not in include_date_set:
            continue
        if exclude_date_set is not None and yyyymmdd in exclude_date_set:
            continue
        files_by_year.setdefault(year, []).append(path)
    return files_by_year


def compute_annual_rx1day(
    daily_nc_dir: str = "out/daily_precip_tps_500m",
    out_dir: str = "out/rx1day",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = None,
    end_year: int | None = None,
) -> list[str]:
    """
    从逐日降水插值结果计算每年的 Rx1day，并按年输出到 `out_dir`。

    Rx1day 定义：某一年内逐像元日降水量的最大值。
    输入默认为 `interpolation.py` 生成的逐日单文件 NetCDF：
    `prep_YYYYMMDD.nc`。
    """
    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
    )

    if not files_by_year:
        raise ValueError("按给定年份范围筛选后，没有可用于计算 Rx1day 的逐日文件。")

    out_paths: list[str] = []
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for year in sorted(files_by_year):
        year_files = sorted(files_by_year[year])
        lats_ref: np.ndarray | None = None
        lons_ref: np.ndarray | None = None
        rx1day: np.ndarray | None = None

        for i, path in enumerate(year_files, start=1):
            lats, lons, prep = _read_daily_precip_nc(
                path,
                precip_var=precip_var,
                lat_var=lat_var,
                lon_var=lon_var,
            )
            if lats_ref is None:
                lats_ref = lats
                lons_ref = lons
                rx1day = prep.copy()
            else:
                if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                    raise ValueError(f"网格不一致，无法合并年度最大值: {path}")
                rx1day = np.fmax(rx1day, prep)

            if i == 1 or i == len(year_files) or i % 30 == 0:
                print(f"[hazard] {year} Rx1day 处理中: {i}/{len(year_files)}")

        assert lats_ref is not None and lons_ref is not None and rx1day is not None
        out_path = out_dir_path / f"rx1day_{year}.nc"
        _write_annual_rx1day_nc(out_path, year, lats_ref, lons_ref, rx1day)
        out_paths.append(str(out_path))
        print(f"[hazard] {year} Rx1day 完成 -> {out_path}")

    return out_paths


def compute_annual_sdii(
    daily_nc_dir: str = "out/daily_precip_tps_500m",
    out_dir: str = "out/sdii",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    wet_day_threshold_mm: float = 1.0,
    start_year: int | None = None,
    end_year: int | None = None,
    start_month: int | None = None,
    end_month: int | None = None,
    include_dates: list[str] | set[str] | tuple[str, ...] | None = None,
    exclude_dates: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """
    从逐日降水插值结果计算每年的 SDII，并按年输出到 `out_dir`。

    SDII 定义：某一年内逐像元湿日（RR >= 1mm）降水总量除以湿日数。
    """
    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        include_dates=include_dates,
        exclude_dates=exclude_dates,
    )

    if not files_by_year:
        raise ValueError("按给定年份范围筛选后，没有可用于计算 SDII 的逐日文件。")

    out_paths: list[str] = []
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for year in sorted(files_by_year):
        year_files = sorted(files_by_year[year])
        lats_ref: np.ndarray | None = None
        lons_ref: np.ndarray | None = None
        wet_sum: np.ndarray | None = None
        wet_count: np.ndarray | None = None

        for i, path in enumerate(year_files, start=1):
            lats, lons, prep = _read_daily_precip_nc(
                path,
                precip_var=precip_var,
                lat_var=lat_var,
                lon_var=lon_var,
            )
            wet_mask = np.isfinite(prep) & (prep >= wet_day_threshold_mm)
            if lats_ref is None:
                lats_ref = lats
                lons_ref = lons
                wet_sum = np.where(wet_mask, prep, 0.0).astype(np.float32)
                wet_count = wet_mask.astype(np.int32)
            else:
                if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                    raise ValueError(f"网格不一致，无法合并年度 SDII: {path}")
                wet_sum = wet_sum + np.where(wet_mask, prep, 0.0).astype(np.float32)
                wet_count = wet_count + wet_mask.astype(np.int32)

            if i == 1 or i == len(year_files) or i % 30 == 0:
                print(f"[hazard] {year} SDII 处理中: {i}/{len(year_files)}")

        assert lats_ref is not None and lons_ref is not None
        assert wet_sum is not None and wet_count is not None
        sdii = np.divide(
            wet_sum,
            wet_count,
            out=np.full(wet_sum.shape, np.nan, dtype=np.float32),
            where=wet_count > 0,
        )
        out_path = out_dir_path / f"sdii_{year}.nc"
        _write_annual_sdii_nc(out_path, year, lats_ref, lons_ref, sdii)
        out_paths.append(str(out_path))
        print(f"[hazard] {year} SDII 完成 -> {out_path}")

    return out_paths


def compute_rx1day_attribution(
    track_csv: str | Path,
    daily_nc_dir: str = "out/daily_precip_tps_500m",
    out_nc_path: str = "out/rx1day/rx1day_2016_2024_5to9_attribution.nc",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = 2016,
    end_year: int | None = 2024,
    start_month: int = 5,
    end_month: int = 9,
) -> tuple[str, list[str]]:
    """
    统计给定时期内逐像元 Rx1day 的归属来源。

    输出 NetCDF 包含：
    - `rx1day`: 该时期逐像元最大单日降水
    - `rx1day_date`: 最大值发生日期，格式 YYYYMMDD
    - `rx1day_source_flag`: 0=非台风日, 1=台风日, 2=台风/非台风并列极值
    """
    _, typhoon_days = get_typhoon_case_count_and_days(
        track_csv,
        start_month=start_month,
        end_month=end_month,
    )
    typhoon_day_set = _normalize_date_strings(typhoon_days) or set()

    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )
    all_files = [path for year in sorted(files_by_year) for path in sorted(files_by_year[year])]
    if not all_files:
        raise ValueError("按给定条件筛选后，没有可用于计算 Rx1day 归属的逐日文件。")

    lats_ref: np.ndarray | None = None
    lons_ref: np.ndarray | None = None
    rx1day: np.ndarray | None = None
    rx1day_date: np.ndarray | None = None
    source_flag: np.ndarray | None = None

    for i, path in enumerate(all_files, start=1):
        yyyymmdd = _extract_yyyymmdd_from_filename(path)
        current_source_flag = np.int8(1 if yyyymmdd in typhoon_day_set else 0)
        current_date_int = np.int32(int(yyyymmdd))

        lats, lons, prep = _read_daily_precip_nc(
            path,
            precip_var=precip_var,
            lat_var=lat_var,
            lon_var=lon_var,
        )
        finite_mask = np.isfinite(prep)
        if lats_ref is None:
            lats_ref = lats
            lons_ref = lons
            rx1day = prep.copy()
            rx1day_date = np.full(prep.shape, -9999, dtype=np.int32)
            rx1day_date[finite_mask] = current_date_int
            source_flag = np.full(prep.shape, -1, dtype=np.int8)
            source_flag[finite_mask] = current_source_flag
        else:
            if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                raise ValueError(f"网格不一致，无法合并 Rx1day 归属: {path}")

            assert rx1day is not None and rx1day_date is not None and source_flag is not None
            greater_mask = finite_mask & (~np.isfinite(rx1day) | (prep > rx1day))
            rx1day[greater_mask] = prep[greater_mask]
            rx1day_date[greater_mask] = current_date_int
            source_flag[greater_mask] = current_source_flag

            tie_mask = finite_mask & np.isfinite(rx1day) & np.isclose(prep, rx1day, rtol=0.0, atol=1e-6)
            mixed_tie_mask = tie_mask & (source_flag >= 0) & (source_flag != current_source_flag)
            source_flag[mixed_tie_mask] = np.int8(2)

        if i == 1 or i == len(all_files) or i % 30 == 0:
            print(f"[hazard] Rx1day 归属处理中: {i}/{len(all_files)}")

    assert lats_ref is not None and lons_ref is not None
    assert rx1day is not None and rx1day_date is not None and source_flag is not None

    _write_rx1day_attribution_nc(
        out_nc_path,
        lats_ref,
        lons_ref,
        rx1day,
        rx1day_date,
        source_flag,
        description=(
            f"Rx1day attribution for selected period "
            f"({start_year or 'all'}-{end_year or 'all'}, months {start_month}-{end_month})"
        ),
    )
    print(f"[hazard] Rx1day 归属完成 -> {out_nc_path}")
    return str(out_nc_path), typhoon_days


def compute_sdii_for_period(
    daily_nc_dir: str = "out/daily_precip_tps_500m",
    out_nc_path: str = "out/sdii/sdii_all.nc",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    wet_day_threshold_mm: float = 1.0,
    start_year: int | None = None,
    end_year: int | None = None,
    start_month: int | None = None,
    end_month: int | None = None,
    include_dates: list[str] | set[str] | tuple[str, ...] | None = None,
    exclude_dates: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str:
    """
    对给定时间范围内的全部日文件整体计算一张 SDII 结果，不区分年份。
    """
    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        include_dates=include_dates,
        exclude_dates=exclude_dates,
    )
    all_files = [path for year in sorted(files_by_year) for path in sorted(files_by_year[year])]
    print(f'用于计算的日期数量：{len(all_files)}')
    if not all_files:
        raise ValueError("按给定条件筛选后，没有可用于计算整体 SDII 的逐日文件。")

    lats_ref: np.ndarray | None = None
    lons_ref: np.ndarray | None = None
    wet_sum: np.ndarray | None = None
    wet_count: np.ndarray | None = None

    for i, path in enumerate(all_files, start=1):
        lats, lons, prep = _read_daily_precip_nc(
            path,
            precip_var=precip_var,
            lat_var=lat_var,
            lon_var=lon_var,
        )
        wet_mask = np.isfinite(prep) & (prep >= wet_day_threshold_mm)
        if lats_ref is None:
            lats_ref = lats
            lons_ref = lons
            wet_sum = np.where(wet_mask, prep, 0.0).astype(np.float32)
            wet_count = wet_mask.astype(np.int32)
        else:
            if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                raise ValueError(f"网格不一致，无法合并整体 SDII: {path}")
            wet_sum = wet_sum + np.where(wet_mask, prep, 0.0).astype(np.float32)
            wet_count = wet_count + wet_mask.astype(np.int32)

        if i == 1 or i == len(all_files) or i % 30 == 0:
            print(f"[hazard] 整体 SDII 处理中: {i}/{len(all_files)}")

    assert lats_ref is not None and lons_ref is not None
    assert wet_sum is not None and wet_count is not None
    sdii = np.divide(
        wet_sum,
        wet_count,
        out=np.full(wet_sum.shape, np.nan, dtype=np.float32),
        where=wet_count > 0,
    )
    _write_sdii_nc(
        out_nc_path,
        lats_ref,
        lons_ref,
        sdii,
        description=f"SDII for selected days (n_days={len(all_files)})",
    )
    print(f"[hazard] 整体 SDII 完成 -> {out_nc_path}")
    return str(out_nc_path)


def compute_annual_threshold_count_and_rate(
    threshold_mm: float,
    metric_name: str,
    daily_nc_dir: str = "out/daily_precip_tps_500m",
    out_count_dir: str | None = None,
    out_rate_dir: str | None = None,
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    计算年度阈值日数与标准化频率。

    - 阈值日数：逐像元统计全年 `prep >= threshold_mm` 的天数
    - 标准化频率：阈值日数 / 当年有效总天数
    """
    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
    )
    if not files_by_year:
        raise ValueError(f"按给定年份范围筛选后，没有可用于计算 {metric_name} 的逐日文件。")

    if out_count_dir is None:
        out_count_dir = f"out/{metric_name.lower()}"
    if out_rate_dir is None:
        out_rate_dir = f"out/{metric_name.lower()}_rate"

    out_count_dir_path = Path(out_count_dir)
    out_rate_dir_path = Path(out_rate_dir)
    out_count_dir_path.mkdir(parents=True, exist_ok=True)
    out_rate_dir_path.mkdir(parents=True, exist_ok=True)

    count_paths: list[str] = []
    rate_paths: list[str] = []

    for year in sorted(files_by_year):
        year_files = sorted(files_by_year[year])
        lats_ref: np.ndarray | None = None
        lons_ref: np.ndarray | None = None
        count = None

        for i, path in enumerate(year_files, start=1):
            lats, lons, prep = _read_daily_precip_nc(
                path,
                precip_var=precip_var,
                lat_var=lat_var,
                lon_var=lon_var,
            )
            exceed = (np.isfinite(prep) & (prep >= threshold_mm)).astype(np.int32)
            if lats_ref is None:
                lats_ref = lats
                lons_ref = lons
                count = exceed
            else:
                if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                    raise ValueError(f"网格不一致，无法合并年度 {metric_name}: {path}")
                count = count + exceed

            if i == 1 or i == len(year_files) or i % 30 == 0:
                print(f"[hazard] {year} {metric_name} 处理中: {i}/{len(year_files)}")

        assert lats_ref is not None and lons_ref is not None and count is not None
        valid_day_count = len(year_files)
        rate = count.astype(np.float32) / float(valid_day_count)

        count_path = out_count_dir_path / f"{metric_name.lower()}_{year}.nc"
        rate_path = out_rate_dir_path / f"{metric_name.lower()}_rate_{year}.nc"
        _write_annual_threshold_metric_nc(
            count_path,
            year,
            lats_ref,
            lons_ref,
            count.astype(np.float32),
            var_name=metric_name.lower(),
            long_name=f"annual count of days with precipitation >= {threshold_mm:g} mm",
            units="days",
        )
        _write_annual_threshold_metric_nc(
            rate_path,
            year,
            lats_ref,
            lons_ref,
            rate,
            var_name=f"{metric_name.lower()}_rate",
            long_name=f"annual standardized frequency of days with precipitation >= {threshold_mm:g} mm",
            units="1",
        )
        count_paths.append(str(count_path))
        rate_paths.append(str(rate_path))
        print(
            f"[hazard] {year} {metric_name} 完成 -> {count_path}; "
            f"rate -> {rate_path} (有效总天数={valid_day_count})"
        )

    return count_paths, rate_paths


def compute_annual_r10mm(**kwargs) -> tuple[list[str], list[str]]:
    return compute_annual_threshold_count_and_rate(
        threshold_mm=10.0,
        metric_name="R10mm",
        **kwargs,
    )


def compute_annual_r20mm(**kwargs) -> tuple[list[str], list[str]]:
    return compute_annual_threshold_count_and_rate(
        threshold_mm=20.0,
        metric_name="R20mm",
        **kwargs,
    )


def compute_annual_r50mm(**kwargs) -> tuple[list[str], list[str]]:
    return compute_annual_threshold_count_and_rate(
        threshold_mm=50.0,
        metric_name="R50mm",
        **kwargs,
    )


def get_typhoon_case_count_and_days(
    track_csv: str | Path,
    start_month: int = 5,
    end_month: int = 9,
) -> tuple[int, list[str]]:
    """
    读取台风路径文件，统计指定月份范围内的台风个数，并返回台风日列表。

    返回：
    - 台风个数：按 SID 去重统计
    - 台风日列表：按 YYYY-MM-DD 排序后的唯一日期
    """
    df = pd.read_csv(track_csv, skiprows=[1])
    df["time"] = pd.to_datetime(
        df["ISO_TIME"].astype(str).str.strip(),
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["time"]).copy()

    if start_month <= end_month:
        month_mask = (df["time"].dt.month >= start_month) & (df["time"].dt.month <= end_month)
    else:
        month_mask = (df["time"].dt.month >= start_month) | (df["time"].dt.month <= end_month)

    df = df.loc[month_mask].copy()

    typhoon_count = df["SID"].astype(str).str.strip().replace("", np.nan).dropna().nunique()
    typhoon_days = sorted(df["time"].dt.strftime("%Y-%m-%d").dropna().unique().tolist())
    return int(typhoon_count), typhoon_days


def compute_typhoon_vs_non_typhoon_sdii(
    track_csv: str | Path,
    daily_nc_dir: str = "out/daily_precip_tps_500m",
    typhoon_out_nc: str = "out/sdii_typhoon/sdii_all.nc",
    non_typhoon_out_nc: str = "out/sdii_non_typhoon/sdii_all.nc",
    start_month: int = 5,
    end_month: int = 9,
    wet_day_threshold_mm: float = 1.0,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[str, str, list[str]]:
    """
    分别计算台风日与非台风日的整体 SDII，不区分年份。

    - 台风日：由路径文件提取的台风日列表确定
    - 非台风日：逐日降水文件中属于分析月份、且不在台风日列表中的日期
    """
    _, typhoon_days = get_typhoon_case_count_and_days(
        track_csv,
        start_month=start_month,
        end_month=end_month,
    )
    print(len(typhoon_days))
    typhoon_sdii_path = compute_sdii_for_period(
        daily_nc_dir=daily_nc_dir,
        out_nc_path=typhoon_out_nc,
        wet_day_threshold_mm=wet_day_threshold_mm,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        include_dates=typhoon_days,
    )
    non_typhoon_sdii_path = compute_sdii_for_period(
        daily_nc_dir=daily_nc_dir,
        out_nc_path=non_typhoon_out_nc,
        wet_day_threshold_mm=wet_day_threshold_mm,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
        exclude_dates=typhoon_days,
    )
    return typhoon_sdii_path, non_typhoon_sdii_path, typhoon_days


def compute_typhoon_r95p_extreme_precip_contribution(
    track_csv: str | Path,
    daily_nc_dir: str = "newout/daily_precip_tps_500m",
    out_dir: str = "newout/r95p_typhoon_contribution",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = 2016,
    end_year: int | None = 2025,
    start_month: int = 5,
    end_month: int = 9,
    wet_day_threshold_mm: float = 1.0,
    percentile: float = 95.0,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    make_plots: bool = True,
) -> dict[str, str]:
    """
    计算台风日对极端降水百分位指标的超额贡献率。

    实现步骤：
    1. 提取湿日(RR >= 1mm)，逐格点计算指定百分位阈值图；
    2. 对所有分析日提取超过逐格点百分位阈值的极端降水量；
    3. 分别累积台风日极端降水量和全部极端降水量，计算
       Frac = 台风日极端降水量 / 全部极端降水量 * 100%。
    """
    metric_label = _percentile_metric_label(percentile)
    metric_display = _metric_display_label(metric_label)
    _, typhoon_days = get_typhoon_case_count_and_days(
        track_csv,
        start_month=start_month,
        end_month=end_month,
    )
    typhoon_day_set = _normalize_date_strings(typhoon_days) or set()

    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )
    all_files = [path for year in sorted(files_by_year) for path in sorted(files_by_year[year])]
    if not all_files:
        raise ValueError(f"按给定条件筛选后，没有可用于 {metric_display} 贡献分析的逐日文件。")

    lats_ref: np.ndarray | None = None
    lons_ref: np.ndarray | None = None
    daily_stack: list[np.ndarray] = []
    date_strings: list[str] = []

    for i, path in enumerate(all_files, start=1):
        lats, lons, prep = _read_daily_precip_nc(
            path,
            precip_var=precip_var,
            lat_var=lat_var,
            lon_var=lon_var,
        )
        if lats_ref is None:
            lats_ref = lats
            lons_ref = lons
        else:
            if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                raise ValueError(f"网格不一致，无法计算 {metric_display} 贡献: {path}")

        wet_prep = prep.astype(np.float32, copy=True)
        wet_prep[~np.isfinite(wet_prep) | (wet_prep < wet_day_threshold_mm)] = np.nan
        daily_stack.append(wet_prep)
        date_strings.append(_extract_yyyymmdd_from_filename(path))

        if i == 1 or i == len(all_files) or i % 30 == 0:
            print(f"[hazard] {metric_display} 阈值样本读取中: {i}/{len(all_files)}")

    assert lats_ref is not None and lons_ref is not None
    data = np.stack(daily_stack, axis=0)
    percentile_threshold = np.nanpercentile(data, percentile, axis=0).astype(np.float32)

    total_extreme_sum = np.zeros(percentile_threshold.shape, dtype=np.float32)
    typhoon_extreme_sum = np.zeros(percentile_threshold.shape, dtype=np.float32)
    total_extreme_count = np.zeros(percentile_threshold.shape, dtype=np.int32)
    typhoon_extreme_count = np.zeros(percentile_threshold.shape, dtype=np.int32)

    for i, (yyyymmdd, wet_prep) in enumerate(zip(date_strings, daily_stack), start=1):
        extreme_mask = np.isfinite(wet_prep) & np.isfinite(percentile_threshold) & (wet_prep > percentile_threshold)
        extreme_amount = np.where(extreme_mask, wet_prep, 0.0).astype(np.float32)
        total_extreme_sum += extreme_amount
        total_extreme_count += extreme_mask.astype(np.int32)

        if yyyymmdd in typhoon_day_set:
            typhoon_extreme_sum += extreme_amount
            typhoon_extreme_count += extreme_mask.astype(np.int32)

        if i == 1 or i == len(daily_stack) or i % 30 == 0:
            print(f"[hazard] {metric_display} 极端降水累积中: {i}/{len(daily_stack)}")

    contribution_frac = np.divide(
        typhoon_extreme_sum,
        total_extreme_sum,
        out=np.full(total_extreme_sum.shape, np.nan, dtype=np.float32),
        where=total_extreme_sum > 0,
    ).astype(np.float32) * 100.0

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    year_label = f"{start_year or 'all'}_{end_year or 'all'}"
    month_label = f"{start_month}to{end_month}"
    out_nc = out_dir_path / f"{metric_label}_typhoon_contribution_{year_label}_{month_label}.nc"
    out_csv = out_dir_path / f"{metric_label}_typhoon_contribution_summary_{year_label}_{month_label}.csv"

    with nc.Dataset(out_nc, "w") as ds:
        ds.createDimension("y", len(lats_ref))
        ds.createDimension("x", len(lons_ref))

        vlat = ds.createVariable("lat", "f4", ("y",))
        vlon = ds.createVariable("lon", "f4", ("x",))
        vthreshold = ds.createVariable(f"{metric_label}_threshold", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
        vtotal = ds.createVariable(f"total_{metric_label}_sum", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
        vtyphoon = ds.createVariable(f"typhoon_{metric_label}_sum", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
        vfrac = ds.createVariable(f"{metric_label}_frac", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.float32(np.nan))
        vtotal_count = ds.createVariable(f"total_{metric_label}_count", "i4", ("y", "x"), zlib=True, complevel=4, fill_value=np.int32(-9999))
        vtyphoon_count = ds.createVariable(f"typhoon_{metric_label}_count", "i4", ("y", "x"), zlib=True, complevel=4, fill_value=np.int32(-9999))

        vlat[:] = lats_ref.astype(np.float32)
        vlon[:] = lons_ref.astype(np.float32)
        vthreshold[:] = percentile_threshold
        vtotal[:] = np.where(total_extreme_sum > 0, total_extreme_sum, np.nan).astype(np.float32)
        vtyphoon[:] = np.where(typhoon_extreme_sum > 0, typhoon_extreme_sum, np.nan).astype(np.float32)
        vfrac[:] = contribution_frac
        vtotal_count[:] = total_extreme_count.astype(np.int32)
        vtyphoon_count[:] = typhoon_extreme_count.astype(np.int32)

        vthreshold.long_name = f"{percentile:g}th percentile threshold of wet-day precipitation"
        vthreshold.units = "mm/day"
        vtotal.long_name = f"total precipitation amount above local {metric_display} threshold"
        vtotal.units = "mm"
        vtyphoon.long_name = f"typhoon-day precipitation amount above local {metric_display} threshold"
        vtyphoon.units = "mm"
        vfrac.long_name = f"fraction of {metric_display} precipitation contributed by typhoon days"
        vfrac.units = "%"
        vtotal_count.long_name = f"count of all days above local {metric_display} threshold"
        vtotal_count.units = "days"
        vtyphoon_count.long_name = f"count of typhoon days above local {metric_display} threshold"
        vtyphoon_count.units = "days"
        ds.description = (
            f"Typhoon contribution to {metric_display} precipitation, years={year_label}, "
            f"months={month_label}, wet_day_threshold={wet_day_threshold_mm:g} mm, "
            f"n_days={len(all_files)}, n_typhoon_days={len(typhoon_day_set)}"
        )

    summary = {
        "start_year": start_year,
        "end_year": end_year,
        "start_month": start_month,
        "end_month": end_month,
        "wet_day_threshold_mm": wet_day_threshold_mm,
        "percentile": percentile,
        "analysis_day_count": len(all_files),
        "typhoon_day_count_from_track": len(typhoon_day_set),
        "typhoon_day_count_with_nc": sum(1 for d in date_strings if d in typhoon_day_set),
        f"domain_total_{metric_label}_sum_mm": float(np.nansum(total_extreme_sum)),
        f"domain_typhoon_{metric_label}_sum_mm": float(np.nansum(typhoon_extreme_sum)),
        f"domain_{metric_label}_frac_percent": (
            float(np.nansum(typhoon_extreme_sum) / np.nansum(total_extreme_sum) * 100.0)
            if np.nansum(total_extreme_sum) > 0
            else np.nan
        ),
        f"mean_grid_{metric_label}_frac_percent": float(np.nanmean(contribution_frac)),
        f"median_grid_{metric_label}_frac_percent": float(np.nanmedian(contribution_frac)),
        f"p90_grid_{metric_label}_frac_percent": float(np.nanpercentile(contribution_frac, 90.0)),
    }
    pd.DataFrame({"metric": list(summary.keys()), "value": list(summary.values())}).to_csv(out_csv, index=False)

    result = {
        f"{metric_label}_contribution_nc": str(out_nc),
        "summary_csv": str(out_csv),
    }
    if make_plots:
        threshold_png = out_dir_path / f"{metric_label}_threshold_{year_label}_{month_label}.png"
        frac_png = out_dir_path / f"{metric_label}_typhoon_frac_{year_label}_{month_label}.png"
        _plot_single_grid_map(
            lats_ref,
            lons_ref,
            percentile_threshold,
            threshold_png,
            title=f"{metric_display} Threshold",
            cbar_label="mm/day",
            province_shp_path=province_shp_path,
            cmap="YlGnBu",
        )
        _plot_single_grid_map(
            lats_ref,
            lons_ref,
            contribution_frac,
            frac_png,
            title=f"Typhoon Contribution to {metric_display} Precipitation",
            cbar_label="%",
            province_shp_path=province_shp_path,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=100.0,
        )
        result[f"{metric_label}_threshold_png"] = str(threshold_png)
        result[f"{metric_label}_frac_png"] = str(frac_png)

    print(f"[hazard] {metric_display} 台风贡献分析完成 -> {out_nc}")
    print(f"[hazard] 汇总指标 -> {out_csv}")
    print(pd.DataFrame({"metric": list(summary.keys()), "value": list(summary.values())}).to_string(index=False))
    return result


def _plot_single_grid_map(
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
    out_png: str | Path,
    title: str,
    cbar_label: str,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    class_boundaries: list[float] | None = None,
    class_colors: list[str] | None = None,
    class_tick_labels: list[str] | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lon2d, lat2d = np.meshgrid(lons, lats)
    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=180)
    ticks = None
    if class_boundaries is not None and class_colors is not None:
        if len(class_boundaries) != len(class_colors) + 1:
            raise ValueError("class_boundaries 的数量必须比 class_colors 多 1。")
        cmap_obj = ListedColormap(class_colors)
        norm = BoundaryNorm(class_boundaries, cmap_obj.N, clip=True)
        ticks = [
            (class_boundaries[i] + class_boundaries[i + 1]) / 2.0
            for i in range(len(class_colors))
        ]
        m = ax.pcolormesh(lon2d, lat2d, data, shading="auto", cmap=cmap_obj, norm=norm)
    else:
        m = ax.pcolormesh(lon2d, lat2d, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=5)

    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = max((maxx - minx) * 0.03, 0.1)
    ypad = max((maxy - miny) * 0.03, 0.1)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)
    cbar = plt.colorbar(m, ax=ax, shrink=0.88, pad=0.02, ticks=ticks)
    if ticks is not None and class_tick_labels is not None:
        if len(class_tick_labels) != len(ticks):
            raise ValueError("class_tick_labels 的数量必须与颜色分级数量一致。")
        cbar.ax.set_yticklabels(class_tick_labels)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_two_panel_grid_maps(
    panels: list[tuple[str, np.ndarray]],
    lats: np.ndarray,
    lons: np.ndarray,
    out_png: str | Path,
    cbar_label: str,
    class_boundaries: list[float],
    class_colors: list[str],
    class_tick_labels: list[str],
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if len(panels) != 2:
        raise ValueError("双子图绘图函数要求 panels 恰好包含 2 个元素。")
    if len(class_boundaries) != len(class_colors) + 1:
        raise ValueError("class_boundaries 的数量必须比 class_colors 多 1。")
    if len(class_tick_labels) != len(class_colors):
        raise ValueError("class_tick_labels 的数量必须与颜色分级数量一致。")

    lon2d, lat2d = np.meshgrid(lons, lats)
    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    cmap_obj = ListedColormap(class_colors)
    norm = BoundaryNorm(class_boundaries, cmap_obj.N, clip=True)
    ticks = [
        (class_boundaries[i] + class_boundaries[i + 1]) / 2.0
        for i in range(len(class_colors))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), dpi=180)
    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = max((maxx - minx) * 0.03, 0.1)
    ypad = max((maxy - miny) * 0.03, 0.1)
    m = None

    for ax, panel_label, (title, data) in zip(axes, ["(a)", "(b)"], panels):
        m = ax.pcolormesh(lon2d, lat2d, data, shading="auto", cmap=cmap_obj, norm=norm)
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=5)
        ax.set_xlim(minx - xpad, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
            zorder=10,
        )
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)

    assert m is not None
    cbar = fig.colorbar(m, ax=axes, shrink=0.86, pad=0.025, ticks=ticks)
    cbar.ax.set_yticklabels(class_tick_labels)
    cbar.set_label(cbar_label)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_r95p_threshold_and_typhoon_fraction_maps(
    contribution_nc: str | Path = "newout/r95p_typhoon_contribution/r95p_typhoon_contribution_2016_2025_5to9.nc",
    out_threshold_png: str | Path | None = None,
    out_frac_png: str | Path | None = None,
    threshold_var: str = "r95p_threshold",
    frac_var: str = "r95p_frac",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> dict[str, str]:
    """
    从 R95p/R99p 台风贡献分析结果 nc 中读取阈值和贡献率并分别绘图。
    """
    contribution_path = Path(contribution_nc)
    metric_label = threshold_var.removesuffix("_threshold")
    metric_display = _metric_display_label(metric_label)
    lats, lons, percentile_threshold = _read_2d_metric_nc(
        contribution_path,
        threshold_var,
    )
    _, _, contribution_frac = _read_2d_metric_nc(
        contribution_path,
        frac_var,
    )

    if out_threshold_png is None:
        out_threshold_png = contribution_path.with_name(f"{threshold_var}.png")
    if out_frac_png is None:
        out_frac_png = contribution_path.with_name(f"{frac_var}.png")

    threshold_png = _plot_single_grid_map(
        lats,
        lons,
        percentile_threshold,
        out_threshold_png,
        title=f"{metric_display} Threshold",
        cbar_label="mm/day",
        province_shp_path=province_shp_path,
        class_boundaries=[0.0, 40.0, 50.0, 75.0, 100.0, 1_000.0],
        class_colors=["#92C5DE", "#4393C3", "#2166AC", "#762A83", "#40004B"],#"#2166AC",
        class_tick_labels=["<40", "40-50", "50-75", "75-100", ">100"],
    )
    frac_png = _plot_single_grid_map(
        lats,
        lons,
        contribution_frac,
        out_frac_png,
        title=f"Typhoon Contribution to {metric_display} Precipitation",
        cbar_label="%",
        province_shp_path=province_shp_path,
        class_boundaries=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        class_colors=["#FFFFCC", "#FFEDA0", "#FEB24C", "#F03B20", "#BD0026"],
        class_tick_labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
    )

    print(f"[hazard] {metric_display} 阈值图输出 -> {threshold_png}")
    print(f"[hazard] {metric_display} 台风贡献率图输出 -> {frac_png}")
    return {
        f"{metric_label}_threshold_png": threshold_png,
        f"{metric_label}_frac_png": frac_png,
    }


def plot_r95p_r99p_threshold_and_typhoon_fraction_comparison_maps(
    r95p_contribution_nc: str | Path,
    r99p_contribution_nc: str | Path,
    out_threshold_png: str | Path,
    out_frac_png: str | Path,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> dict[str, str]:
    """
    将 R95p/R99p 的阈值分布绘制到一张双子图，并将台风贡献率绘制到另一张双子图。
    """
    lats95, lons95, r95p_threshold = _read_2d_metric_nc(r95p_contribution_nc, "r95p_threshold")
    _, _, r95p_frac = _read_2d_metric_nc(r95p_contribution_nc, "r95p_frac")
    lats99, lons99, r99p_threshold = _read_2d_metric_nc(r99p_contribution_nc, "r99p_threshold")
    _, _, r99p_frac = _read_2d_metric_nc(r99p_contribution_nc, "r99p_frac")

    if not np.array_equal(lats95, lats99) or not np.array_equal(lons95, lons99):
        raise ValueError("R95p 与 R99p 的网格不一致，无法绘制合并图。")

    threshold_png = _plot_two_panel_grid_maps(
        panels=[
            ("R95p Threshold", r95p_threshold),
            ("R99p Threshold", r99p_threshold),
        ],
        lats=lats95,
        lons=lons95,
        out_png=out_threshold_png,
        cbar_label="mm/day",
        province_shp_path=province_shp_path,
        class_boundaries=[0.0, 40.0, 50.0, 75.0, 100.0, 1_000.0],
        class_colors=["#92C5DE", "#4393C3", "#2166AC", "#762A83", "#40004B"],
        class_tick_labels=["<40", "40-50", "50-75", "75-100", ">100"],
    )
    frac_png = _plot_two_panel_grid_maps(
        panels=[
            ("R95p Typhoon Contribution", r95p_frac),
            ("R99p Typhoon Contribution", r99p_frac),
        ],
        lats=lats95,
        lons=lons95,
        out_png=out_frac_png,
        cbar_label="%",
        province_shp_path=province_shp_path,
        class_boundaries=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        class_colors=["#FFFFCC", "#FFEDA0", "#FEB24C", "#F03B20", "#BD0026"],
        class_tick_labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
    )

    print(f"[hazard] R95p/R99p 阈值合并图输出 -> {threshold_png}")
    print(f"[hazard] R95p/R99p 台风贡献率合并图输出 -> {frac_png}")
    return {
        "threshold_png": threshold_png,
        "frac_png": frac_png,
    }


def compute_typhoon_vs_non_typhoon_precip_pdf(
    track_csv: str | Path,
    daily_nc_dir: str = "newout/daily_precip_tps_500m",
    out_csv: str | Path = "newout/precip_pdf/typhoon_vs_non_typhoon_precip_pdf_2016_2025_5to9.csv",
    out_png: str | Path | None = "newout/precip_pdf/typhoon_vs_non_typhoon_precip_pdf_2016_2025_5to9.png",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = 2016,
    end_year: int | None = 2025,
    start_month: int = 5,
    end_month: int = 9,
    bins: list[float] | None = None,
    bin_labels: list[str] | None = None,
    wet_day_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    """
    计算台风日与非台风日降水概率密度分布的谱偏移。

    输出两个视角：
    - 频次概率分布：每个降水等级的有效降水像元次数 / 对应池子的有效降水像元总次数。
    - 水量贡献分布：每个降水等级的降水总量 / 对应池子的有效降水总量。
    """
    if bins is None:
        bins = [1.0, 10.0, 25.0, 50.0, 100.0, np.inf]
    if bin_labels is None:
        bin_labels = ["1-10 mm", "10-25 mm", "25-50 mm", "50-100 mm", ">=100 mm"]
    if len(bin_labels) != len(bins) - 1:
        raise ValueError("bin_labels 的数量必须比 bins 少 1。")

    _, typhoon_days = get_typhoon_case_count_and_days(
        track_csv,
        start_month=start_month,
        end_month=end_month,
    )
    typhoon_day_set = _normalize_date_strings(typhoon_days) or set()

    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )
    all_files = [path for year in sorted(files_by_year) for path in sorted(files_by_year[year])]
    if not all_files:
        raise ValueError("按给定条件筛选后，没有可用于降水 PDF 分析的逐日文件。")

    pools = {
        "Typhoon": {
            "counts": np.zeros(len(bin_labels), dtype=np.int64),
            "sums": np.zeros(len(bin_labels), dtype=np.float64),
            "day_count": 0,
        },
        "Non-typhoon": {
            "counts": np.zeros(len(bin_labels), dtype=np.int64),
            "sums": np.zeros(len(bin_labels), dtype=np.float64),
            "day_count": 0,
        },
    }

    lats_ref: np.ndarray | None = None
    lons_ref: np.ndarray | None = None
    typhoon_day_count_with_nc = 0

    for i, path in enumerate(all_files, start=1):
        yyyymmdd = _extract_yyyymmdd_from_filename(path)
        pool_name = "Typhoon" if yyyymmdd in typhoon_day_set else "Non-typhoon"
        if pool_name == "Typhoon":
            typhoon_day_count_with_nc += 1
        pools[pool_name]["day_count"] += 1

        lats, lons, prep = _read_daily_precip_nc(
            path,
            precip_var=precip_var,
            lat_var=lat_var,
            lon_var=lon_var,
        )
        if lats_ref is None:
            lats_ref = lats
            lons_ref = lons
        else:
            if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                raise ValueError(f"网格不一致，无法计算降水 PDF: {path}")

        values = prep[np.isfinite(prep) & (prep >= wet_day_threshold_mm)].astype(np.float64)
        if values.size:
            pools[pool_name]["counts"] += np.histogram(values, bins=bins)[0].astype(np.int64)
            pools[pool_name]["sums"] += np.histogram(values, bins=bins, weights=values)[0].astype(np.float64)

        if i == 1 or i == len(all_files) or i % 30 == 0:
            print(f"[hazard] 降水 PDF 统计中: {i}/{len(all_files)}")

    rows: list[dict[str, object]] = []
    for pool_name, pool in pools.items():
        counts = pool["counts"]
        sums = pool["sums"]
        total_count = int(counts.sum())
        total_sum = float(sums.sum())
        for idx, label in enumerate(bin_labels):
            lower = bins[idx]
            upper = bins[idx + 1]
            event_frequency = counts[idx] / total_count * 100.0 if total_count > 0 else np.nan
            volume_contribution = sums[idx] / total_sum * 100.0 if total_sum > 0 else np.nan
            rows.append(
                {
                    "pool": pool_name,
                    "bin": label,
                    "lower_mm": lower,
                    "upper_mm": upper,
                    "day_count": int(pool["day_count"]),
                    "wet_pixel_count_total": total_count,
                    "wet_pixel_count_bin": int(counts[idx]),
                    "event_frequency_percent": float(event_frequency),
                    "precip_sum_total_mm": total_sum,
                    "precip_sum_bin_mm": float(sums[idx]),
                    "volume_contribution_percent": float(volume_contribution),
                }
            )

    out_df = pd.DataFrame(rows)

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    if out_png is not None:
        _plot_typhoon_vs_non_typhoon_precip_pdf_bars(out_df, out_png)

    print(f"[hazard] 台风/非台风降水 PDF 统计完成 -> {out_csv_path}")
    print(
        f"[hazard] 分析日数={len(all_files)}，台风日={typhoon_day_count_with_nc}，"
        f"非台风日={len(all_files) - typhoon_day_count_with_nc}"
    )
    print(out_df.to_string(index=False))
    return out_df


def compute_typhoon_wet_non_typhoon_all_precip_pdf(
    track_csv: str | Path,
    daily_nc_dir: str = "newout/daily_precip_tps_500m",
    out_csv: str | Path = "newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.csv",
    out_png: str | Path | None = "newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.png",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = 2016,
    end_year: int | None = 2025,
    typhoon_start_month: int = 1,
    typhoon_end_month: int = 12,
    wet_season_start_month: int = 5,
    wet_season_end_month: int = 9,
    bins: list[float] | None = None,
    bin_labels: list[str] | None = None,
    wet_day_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    """
    统计三类样本的降水 PDF：
    1. 十年间所有台风日；
    2. 雨季(默认 5-9 月)非台风日；
    3. 十年间所有日期的全部样本。
    """
    if bins is None:
        bins = [1.0, 10.0, 25.0, 50.0, 100.0, np.inf]
    if bin_labels is None:
        bin_labels = ["1-10 mm", "10-25 mm", "25-50 mm", "50-100 mm", ">=100 mm"]
    if len(bin_labels) != len(bins) - 1:
        raise ValueError("bin_labels 的数量必须比 bins 少 1。")

    _, typhoon_days = get_typhoon_case_count_and_days(
        track_csv,
        start_month=typhoon_start_month,
        end_month=typhoon_end_month,
    )
    typhoon_day_set = _normalize_date_strings(typhoon_days) or set()

    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
    )
    all_files = [path for year in sorted(files_by_year) for path in sorted(files_by_year[year])]
    if not all_files:
        raise ValueError("按给定年份筛选后，没有可用于三类降水 PDF 分析的逐日文件。")

    pool_order = ["Typhoon", "Wet season non-typhoon", "All days"]
    pools = {
        pool_name: {
            "counts": np.zeros(len(bin_labels), dtype=np.int64),
            "sums": np.zeros(len(bin_labels), dtype=np.float64),
            "day_count": 0,
        }
        for pool_name in pool_order
    }

    lats_ref: np.ndarray | None = None
    lons_ref: np.ndarray | None = None

    for i, path in enumerate(all_files, start=1):
        yyyymmdd = _extract_yyyymmdd_from_filename(path)
        month = int(yyyymmdd[4:6])
        is_typhoon_day = yyyymmdd in typhoon_day_set
        if wet_season_start_month <= wet_season_end_month:
            is_wet_season = wet_season_start_month <= month <= wet_season_end_month
        else:
            is_wet_season = month >= wet_season_start_month or month <= wet_season_end_month

        target_pools = ["All days"]
        if is_typhoon_day:
            target_pools.append("Typhoon")
        elif is_wet_season:
            target_pools.append("Wet season non-typhoon")

        lats, lons, prep = _read_daily_precip_nc(
            path,
            precip_var=precip_var,
            lat_var=lat_var,
            lon_var=lon_var,
        )
        if lats_ref is None:
            lats_ref = lats
            lons_ref = lons
        else:
            if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                raise ValueError(f"网格不一致，无法计算三类降水 PDF: {path}")

        values = prep[np.isfinite(prep) & (prep >= wet_day_threshold_mm)].astype(np.float64)
        hist_counts = np.histogram(values, bins=bins)[0].astype(np.int64) if values.size else None
        hist_sums = np.histogram(values, bins=bins, weights=values)[0].astype(np.float64) if values.size else None
        for pool_name in target_pools:
            pools[pool_name]["day_count"] += 1
            if values.size:
                pools[pool_name]["counts"] += hist_counts
                pools[pool_name]["sums"] += hist_sums

        if i == 1 or i == len(all_files) or i % 30 == 0:
            print(f"[hazard] 三类降水 PDF 统计中: {i}/{len(all_files)}")

    rows: list[dict[str, object]] = []
    for pool_name in pool_order:
        pool = pools[pool_name]
        counts = pool["counts"]
        sums = pool["sums"]
        total_count = int(counts.sum())
        total_sum = float(sums.sum())
        for idx, label in enumerate(bin_labels):
            lower = bins[idx]
            upper = bins[idx + 1]
            event_frequency = counts[idx] / total_count * 100.0 if total_count > 0 else np.nan
            volume_contribution = sums[idx] / total_sum * 100.0 if total_sum > 0 else np.nan
            rows.append(
                {
                    "pool": pool_name,
                    "bin": label,
                    "lower_mm": lower,
                    "upper_mm": upper,
                    "day_count": int(pool["day_count"]),
                    "wet_pixel_count_total": total_count,
                    "wet_pixel_count_bin": int(counts[idx]),
                    "event_frequency_percent": float(event_frequency),
                    "precip_sum_total_mm": total_sum,
                    "precip_sum_bin_mm": float(sums[idx]),
                    "volume_contribution_percent": float(volume_contribution),
                }
            )

    out_df = pd.DataFrame(rows)

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    if out_png is not None:
        _plot_multi_pool_precip_pdf_bars(out_df, out_png, pool_order=pool_order)

    day_summary = ", ".join(f"{name}={int(pools[name]['day_count'])}" for name in pool_order)
    print(f"[hazard] 三类降水 PDF 统计完成 -> {out_csv_path}")
    print(f"[hazard] 分析日数={len(all_files)}，{day_summary}")
    print(out_df.to_string(index=False))
    return out_df


def plot_typhoon_wet_non_typhoon_all_precip_pdf_from_csv(
    csv_path: str | Path = "newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.csv",
    out_png: str | Path = "newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.png",
    pool_order: list[str] | None = None,
) -> str:
    """
    读取已统计好的三类降水 PDF CSV，并重新绘制频次/水量贡献双子图。
    """
    pdf_df = pd.read_csv(csv_path)
    required_cols = {
        "pool",
        "bin",
        "event_frequency_percent",
        "volume_contribution_percent",
    }
    missing_cols = required_cols - set(pdf_df.columns)
    if missing_cols:
        raise ValueError(f"PDF CSV 缺少必要字段: {sorted(missing_cols)}")

    if pool_order is None:
        default_order = ["Typhoon", "Wet season non-typhoon", "All days"]
        existing_pools = set(pdf_df["pool"].dropna().astype(str))
        pool_order = [pool for pool in default_order if pool in existing_pools]
        pool_order.extend(
            pool
            for pool in pdf_df["pool"].drop_duplicates().astype(str).tolist()
            if pool not in pool_order
        )

    return _plot_multi_pool_precip_pdf_bars(pdf_df, out_png, pool_order=pool_order)


def compute_typhoon_vs_non_typhoon_precip_ks_test(
    track_csv: str | Path,
    daily_nc_dir: str = "newout/daily_precip_tps_500m",
    out_csv: str | Path = "newout/precip_pdf/typhoon_vs_non_typhoon_precip_ks_2016_2025_5to9.csv",
    daily_nc_glob: str = "prep_*.nc",
    precip_var: str = "prep",
    lat_var: str = "lat",
    lon_var: str = "lon",
    start_year: int | None = 2016,
    end_year: int | None = 2025,
    start_month: int = 5,
    end_month: int = 9,
    wet_day_threshold_mm: float = 1.0,
    max_samples_per_pool: int = 500_000,
    random_state: int | None = 0,
    significance_level: float = 0.01,
) -> pd.DataFrame:
    """
    单独执行台风日与非台风日有效降水分布的双样本 Kolmogorov-Smirnov 检验。

    为避免一次性展开全部像元导致内存过大，每个池子最多保留
    `max_samples_per_pool` 个蓄水池抽样样本。
    """
    if max_samples_per_pool <= 0:
        raise ValueError("max_samples_per_pool 必须为正整数。")

    _, typhoon_days = get_typhoon_case_count_and_days(
        track_csv,
        start_month=start_month,
        end_month=end_month,
    )
    typhoon_day_set = _normalize_date_strings(typhoon_days) or set()

    files_by_year = _collect_daily_files_by_year(
        daily_nc_dir=daily_nc_dir,
        daily_nc_glob=daily_nc_glob,
        start_year=start_year,
        end_year=end_year,
        start_month=start_month,
        end_month=end_month,
    )
    all_files = [path for year in sorted(files_by_year) for path in sorted(files_by_year[year])]
    if not all_files:
        raise ValueError("按给定条件筛选后，没有可用于 K-S 检验的逐日文件。")

    rng = np.random.default_rng(random_state if random_state is not None else None)
    buffers = {
        "Typhoon": np.empty(0, dtype=np.float64),
        "Non-typhoon": np.empty(0, dtype=np.float64),
    }
    seen = {"Typhoon": 0, "Non-typhoon": 0}
    day_counts = {"Typhoon": 0, "Non-typhoon": 0}
    lats_ref: np.ndarray | None = None
    lons_ref: np.ndarray | None = None

    for i, path in enumerate(all_files, start=1):
        yyyymmdd = _extract_yyyymmdd_from_filename(path)
        pool_name = "Typhoon" if yyyymmdd in typhoon_day_set else "Non-typhoon"
        day_counts[pool_name] += 1

        lats, lons, prep = _read_daily_precip_nc(
            path,
            precip_var=precip_var,
            lat_var=lat_var,
            lon_var=lon_var,
        )
        if lats_ref is None:
            lats_ref = lats
            lons_ref = lons
        else:
            if not np.array_equal(lats_ref, lats) or not np.array_equal(lons_ref, lons):
                raise ValueError(f"网格不一致，无法计算 K-S 检验: {path}")

        values = prep[np.isfinite(prep) & (prep >= wet_day_threshold_mm)].astype(np.float64)
        if values.size:
            buffers[pool_name], seen[pool_name] = _reservoir_update_from_values(
                values,
                buffers[pool_name],
                seen[pool_name],
                max_keep=max_samples_per_pool,
                rng=rng,
            )

        if i == 1 or i == len(all_files) or i % 30 == 0:
            print(f"[hazard] K-S 抽样中: {i}/{len(all_files)}")

    typhoon_samples = buffers["Typhoon"]
    non_samples = buffers["Non-typhoon"]
    ks_stat = np.nan
    ks_p = np.nan
    conclusion = "skipped"

    if typhoon_samples.size >= 2 and non_samples.size >= 2:
        from scipy import stats

        try:
            ks_stat, ks_p = stats.ks_2samp(typhoon_samples, non_samples, method="auto")
        except TypeError:
            ks_stat, ks_p = stats.ks_2samp(typhoon_samples, non_samples)
        conclusion = (
            "reject_null_hypothesis"
            if float(ks_p) < float(significance_level)
            else "fail_to_reject_null_hypothesis"
        )
    else:
        print("[hazard] K-S 检验跳过：某一池子的有效降水样本不足。")

    out_df = pd.DataFrame(
        [
            {
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "ks_alpha": float(significance_level),
                "ks_conclusion": conclusion,
                "typhoon_day_count": int(day_counts["Typhoon"]),
                "non_typhoon_day_count": int(day_counts["Non-typhoon"]),
                "typhoon_sample_size": int(typhoon_samples.size),
                "non_typhoon_sample_size": int(non_samples.size),
                "typhoon_seen_total": int(seen["Typhoon"]),
                "non_typhoon_seen_total": int(seen["Non-typhoon"]),
                "max_samples_per_pool": int(max_samples_per_pool),
                "random_state": random_state,
                "wet_day_threshold_mm": float(wet_day_threshold_mm),
            }
        ]
    )

    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    print(f"[hazard] K-S 检验结果 -> {out_csv_path}")
    print(f"[hazard] K-S 检验: D={ks_stat:.4f}, p={ks_p:.4e}, alpha={significance_level:g}")
    if conclusion == "reject_null_hypothesis":
        print(f"[hazard] 拒绝原假设：两类降水分布存在显著差异 (p < {significance_level:g})。")
    elif conclusion == "fail_to_reject_null_hypothesis":
        print(f"[hazard] 未能拒绝原假设：抽样样本上未观察到显著分布差异 (p >= {significance_level:g})。")
    print(out_df.to_string(index=False))
    return out_df


def _reservoir_update_from_values(
    new_values: np.ndarray,
    buffer: np.ndarray,
    seen_total: int,
    max_keep: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    将一批新值并入蓄水池样本：最多保留 `max_keep` 条，近似均匀随机抽样。

    按批次更新，避免对每个像元做 Python 循环。
    """
    if new_values.size == 0:
        return buffer, seen_total

    new_values = new_values.astype(np.float64, copy=False)
    old_seen = int(seen_total)
    new_count = int(new_values.size)
    total_seen = old_seen + new_count

    if total_seen <= max_keep:
        return np.concatenate([buffer, new_values]), total_seen

    if old_seen < max_keep:
        combined = np.concatenate([buffer, new_values])
        choice = rng.choice(combined.size, size=max_keep, replace=False)
        return combined[choice], total_seen

    # 旧 buffer 是 old_seen 个样本中的 max_keep 个均匀样本。
    # 从 total_seen 个样本中抽 max_keep 个时，新批次应进入样本的数量服从超几何分布。
    new_keep = int(rng.hypergeometric(ngood=new_count, nbad=old_seen, nsample=max_keep))
    old_keep = max_keep - new_keep

    if old_keep > 0:
        old_choice = rng.choice(buffer.size, size=old_keep, replace=False)
        kept_old = buffer[old_choice]
    else:
        kept_old = np.empty(0, dtype=np.float64)

    if new_keep > 0:
        new_choice = rng.choice(new_count, size=new_keep, replace=False)
        kept_new = new_values[new_choice]
    else:
        kept_new = np.empty(0, dtype=np.float64)

    return np.concatenate([kept_old, kept_new]), total_seen


def _plot_typhoon_vs_non_typhoon_precip_pdf_bars(
    pdf_df: pd.DataFrame,
    out_png: str | Path,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = pdf_df[pdf_df["pool"].isin(["Typhoon", "Non-typhoon"])].copy()
    bins = plot_df["bin"].drop_duplicates().tolist()
    x = np.arange(len(bins))
    width = 0.36
    colors = {"Typhoon": "#D73027", "Non-typhoon": "#4575B4"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180, sharex=True)
    specs = [
        ("event_frequency_percent", "Event Frequency PDF", "Frequency (%)"),
        ("volume_contribution_percent", "Volume Contribution PDF", "Contribution (%)"),
    ]

    for ax, (value_col, title, ylabel) in zip(axes, specs):
        for offset, pool in [(-width / 2, "Typhoon"), (width / 2, "Non-typhoon")]:
            values = (
                plot_df[plot_df["pool"] == pool]
                .set_index("bin")
                .reindex(bins)[value_col]
                .to_numpy(dtype=float)
            )
            ax.bar(x + offset, values, width=width, label=pool, color=colors[pool], edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=25, ha="right")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.45)
        ax.legend(frameon=False)

    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[hazard] 台风/非台风降水 PDF 图输出 -> {out_path}")
    return str(out_path)


def _plot_multi_pool_precip_pdf_bars(
    pdf_df: pd.DataFrame,
    out_png: str | Path,
    pool_order: list[str] | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pool_order is None:
        pool_order = pdf_df["pool"].drop_duplicates().tolist()
    plot_df = pdf_df[pdf_df["pool"].isin(pool_order)].copy()
    bins = plot_df["bin"].drop_duplicates().tolist()
    x = np.arange(len(bins))
    bar_width = min(0.8 / max(len(pool_order), 1), 0.26)
    offsets = (np.arange(len(pool_order)) - (len(pool_order) - 1) / 2.0) * bar_width
    color_cycle = ["#D73027", "#4575B4", "#4DAF4A", "#984EA3", "#FF7F00"]
    color_by_pool = {pool: color_cycle[i % len(color_cycle)] for i, pool in enumerate(pool_order)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=180, sharex=True)
    specs = [
        ("event_frequency_percent", "Event Frequency PDF", "Frequency (%)"),
        ("volume_contribution_percent", "Volume Contribution PDF", "Contribution (%)"),
    ]

    for ax, panel_label, (value_col, title, ylabel) in zip(axes, ["(a)", "(b)"], specs):
        for offset, pool in zip(offsets, pool_order):
            values = (
                plot_df[plot_df["pool"] == pool]
                .set_index("bin")
                .reindex(bins)[value_col]
                .to_numpy(dtype=float)
            )
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=pool,
                color=color_by_pool[pool],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(title)
        ax.text(
            -0.08,
            1.05,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=25, ha="right")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.35, alpha=0.45)
        ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[hazard] 多组降水 PDF 图输出 -> {out_path}")
    return str(out_path)


def plot_typhoon_non_typhoon_sdii_comparison(
    typhoon_sdii_nc: str | Path,
    non_typhoon_sdii_nc: str | Path,
    out_png: str | Path,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    """
    绘制台风日与非台风日 SDII 对比图，并叠加浙江省边界。
    左图: 台风日 SDII
    中图: 非台风日 SDII
    右图: 差值(台风日 - 非台风日)
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lats1, lons1, sdii_typhoon = _read_2d_metric_nc(typhoon_sdii_nc, "sdii")
    lats2, lons2, sdii_non_typhoon = _read_2d_metric_nc(non_typhoon_sdii_nc, "sdii")
    if not np.array_equal(lats1, lats2) or not np.array_equal(lons1, lons2):
        raise ValueError("台风日与非台风日 SDII 网格不一致，无法直接对比绘图。")
    
    sdii_diff = sdii_typhoon - sdii_non_typhoon
    lon2d, lat2d = np.meshgrid(lons1, lats1)

    valid_main = np.concatenate(
        [sdii_typhoon[np.isfinite(sdii_typhoon)], sdii_non_typhoon[np.isfinite(sdii_non_typhoon)]]
    )
    if valid_main.size == 0:
        raise ValueError("SDII 数据全为缺测，无法绘图。")
    vmax = float(np.nanmax(valid_main))
    sdii_boundaries = [0.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0, max(120.0, np.ceil(vmax))]
    sdii_tick_centers = [
        7.5,
        17.5,
        22.5,
        27.5,
        40.0,
        75.0,
        (sdii_boundaries[-2] + sdii_boundaries[-1]) / 2.0,
    ]
    sdii_tick_labels = ["<15", "15-20", "20-25", "25-30", "30-50", "50-100", ">100"]
    sdii_cmap = ListedColormap(
        ["#f7fcf0", "#ccebc5", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cbe", "#084081"]
    )
    sdii_norm = BoundaryNorm(sdii_boundaries, sdii_cmap.N, clip=True)

    diff_cmap = ListedColormap(
        ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    )
    # 使用有限边界保证 colorbar 刻度与离散色块中心正确对应。
    # 小于 -5 和大于 50 的值都会分别归入首末两档。
    diff_boundaries = [-10.0, -5.0, 0.0, 10.0, 20.0, 30.0, 50.0, 60.0]
    diff_tick_centers = [-7.5, -2.5, 5.0, 15.0, 25.0, 40.0, 55.0]
    diff_tick_labels = ["<-5", "-5-0", "0-10", "10-20", "20-30", "30-50", ">50"]
    diff_norm = BoundaryNorm(diff_boundaries, diff_cmap.N, clip=True)

    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=180)
    panel_specs = [
        ("Typhoon-day SDII", sdii_typhoon, sdii_cmap, sdii_norm, True),
        ("Wet-season SDII on Non-typhoon Days", sdii_non_typhoon, sdii_cmap, sdii_norm, True),
        ("Difference", sdii_diff, diff_cmap, diff_norm, False),
    ]
    panel_labels = ["(a)", "(b)", "(c)"]

    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = max((maxx - minx) * 0.03, 0.1)
    ypad = max((maxy - miny) * 0.03, 0.1)

    for ax, panel_label, (title, data, cmap, norm, is_sdii_panel) in zip(axes, panel_labels, panel_specs):
        if is_sdii_panel:
            m = ax.pcolormesh(
                lon2d,
                lat2d,
                data,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
        else:
            m = ax.pcolormesh(
                lon2d,
                lat2d,
                data,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=5)
        ax.set_xlim(minx - xpad, maxx + xpad)
        ax.set_ylim(miny - ypad, maxy + ypad)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
            zorder=10,
        )
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)
        if is_sdii_panel:
            cbar = plt.colorbar(m, ax=ax, shrink=0.85, pad=0.02, ticks=sdii_tick_centers)
            cbar.ax.set_yticklabels(sdii_tick_labels)
        else:
            cbar = plt.colorbar(m, ax=ax, shrink=0.85, pad=0.02, ticks=diff_tick_centers)
            cbar.ax.set_yticklabels(diff_tick_labels)
        cbar.set_label("mm/day")

    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_rx1day_attribution_map(
    attribution_nc: str | Path,
    out_png: str | Path,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    """
    绘制 Rx1day 极值归属图。

    - 红色: 台风日创造极值
    - 蓝色: 非台风日创造极值
    - 紫色: 台风日与非台风日并列极值
    """
    import matplotlib

    matplotlib.use("Agg")
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lats, lons, source_flag = _read_2d_metric_nc(attribution_nc, "rx1day_source_flag")
    _, _, rx1day = _read_2d_metric_nc(attribution_nc, "rx1day")
    source_flag = np.asarray(source_flag, dtype=np.float32)
    source_flag[~np.isfinite(rx1day)] = np.nan
    lon2d, lat2d = np.meshgrid(lons, lats)

    present_flags = np.unique(source_flag[np.isfinite(source_flag) & np.isfinite(rx1day)])
    present_sorted = np.sort(present_flags.astype(np.float32)).tolist()

    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    color_by_flag = {
        0: "#2b8cbe",
        1: "#d7301f",
        2: "#984ea3",
    }
    label_by_flag = {
        0: "Non-typhoon-day",
        1: "Typhoon-day",
        2: "2",
    }

    if len(present_sorted) == 0:
        raise ValueError("没有可用于绘制的 rx1day_source_flag 有效像元。")

    cmap_colors = [color_by_flag[int(f)] for f in present_sorted]
    cmap = ListedColormap(cmap_colors)
    boundaries = [present_sorted[0] - 0.5]
    for i in range(len(present_sorted) - 1):
        boundaries.append((present_sorted[i] + present_sorted[i + 1]) / 2.0)
    boundaries.append(present_sorted[-1] + 0.5)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=180)
    m = ax.pcolormesh(lon2d, lat2d, source_flag, shading="auto", cmap=cmap, norm=norm)
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=5)

    minx, miny, maxx, maxy = gdf.total_bounds
    xpad = max((maxx - minx) * 0.03, 0.1)
    ypad = max((maxy - miny) * 0.03, 0.1)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Rx1day Attribution")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)

    tick_vals = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(len(present_sorted))]
    tick_labels = [label_by_flag[int(f)] for f in present_sorted]
    cbar = plt.colorbar(m, ax=ax, shrink=0.88, pad=0.02, ticks=tick_vals)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("rx1day_source_flag")

    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


if __name__ == "__main__":
    #4.3 读取已统计 CSV，绘制台风日、雨季非台风日、全日期样本的 PDF
    plot_typhoon_wet_non_typhoon_all_precip_pdf_from_csv(
        csv_path="newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.csv",
        out_png="newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.png",
    )
    exit()
   
    
    #1、台风天 和 非台风天 的 SDII 
    #1.1计算台风天 和 非台风天 的 SDII 
    typhoon_path, non_typhoon_path, typhoon_days = compute_typhoon_vs_non_typhoon_sdii(
    track_csv="newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t3h.csv",
    daily_nc_dir="newout/daily_precip_tps_500m",
    typhoon_out_nc="newout/sdii_typhoon/sdii_2016_2025_5to9_typhoon.nc",
    non_typhoon_out_nc="newout/sdii_non_typhoon/sdii_2016_2025_5to9_non_typhoon.nc",
    start_month=5,
    end_month=9,
)
    #1.2 sdii对比图
    plot_path = plot_typhoon_non_typhoon_sdii_comparison(
        "newout/sdii_typhoon/sdii_2016_2025_5to9_typhoon.nc",
         "newout/sdii_non_typhoon/sdii_2016_2025_5to9_non_typhoon.nc",
        out_png="newout/sdii_typhoon_vs_non_typhoon_2016_2025_5to9.png",
    )
    print("SDII 对比图输出:", plot_path)
    exit()

    #2、rx1day、发生日期、是否发生在台风日     
    attribution_nc, typhoon_days = compute_rx1day_attribution(
        track_csv="newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t3h.csv",
        daily_nc_dir="newout/daily_precip_tps_500m",
        out_nc_path="newout/rx1day/rx1day_2016_2025_1to12_attribution.nc",
        start_year=2016,
        end_year=2025,
        start_month=1,
        end_month=12,
    )
    plot_path = plot_rx1day_attribution_map(
        attribution_nc,
        out_png="newout/rx1day/rx1day_2016_2025_1to12_attribution.png",
    )
    print("归属结果文件:", attribution_nc)
    print("归属图:", plot_path)
    exit()
    #4.2 台风日、雨季非台风日、全日期样本的 PDF
    compute_typhoon_wet_non_typhoon_all_precip_pdf(
        track_csv="newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t3h.csv",
        daily_nc_dir="newout/daily_precip_tps_500m",
        out_csv="newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.csv",
        out_png="newout/precip_pdf/typhoon_wetseason_non_typhoon_all_precip_pdf_2016_2025.png",
        start_year=2016,
        end_year=2025,
        typhoon_start_month=1,
        typhoon_end_month=12,
        wet_season_start_month=5,
        wet_season_end_month=9,
        bins=[1.0, 10.0, 25.0, 50.0, 100.0, 250.0, np.inf],
        bin_labels=["1-10 mm", "10-25 mm", "25-50 mm", "50-100 mm", "100-250 mm", ">=250 mm"],
    )
    exit()
    #3、台风对于极端降水的贡献
    #3.1 分别计算 R95p/R99p
    contribution_dir = "newout/r95p_typhoon_contribution"
    for percentile in (95.0, 99.0):
        metric_label = _percentile_metric_label(percentile)
        '''
        compute_typhoon_r95p_extreme_precip_contribution(
            track_csv="newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t3h.csv",
            daily_nc_dir="newout/daily_precip_tps_500m",
            out_dir=contribution_dir,
            start_year=2016,
            end_year=2025,
            start_month=1,
            end_month=12,
            percentile=percentile,
            make_plots=False,
        )
        '''
        #3.2 分别绘制阈值图和台风贡献率图
        plot_r95p_threshold_and_typhoon_fraction_maps(
            contribution_nc=f"{contribution_dir}/{metric_label}_typhoon_contribution_2016_2025_1to12.nc",
            out_threshold_png=f"{contribution_dir}/{metric_label}_threshold_2016_2025_1to12.png",
            out_frac_png=f"{contribution_dir}/{metric_label}_typhoon_frac_2016_2025_1to12.png",
            threshold_var=f"{metric_label}_threshold",
            frac_var=f"{metric_label}_frac",
        )
    #3.3 将 R95p/R99p 阈值图、台风贡献率图分别合并为双子图
    plot_r95p_r99p_threshold_and_typhoon_fraction_comparison_maps(
        r95p_contribution_nc=f"{contribution_dir}/r95p_typhoon_contribution_2016_2025_1to12.nc",
        r99p_contribution_nc=f"{contribution_dir}/r99p_typhoon_contribution_2016_2025_1to12.nc",
        out_threshold_png=f"{contribution_dir}/r95p_r99p_threshold_2016_2025_1to12.png",
        out_frac_png=f"{contribution_dir}/r95p_r99p_typhoon_frac_2016_2025_1to12.png",
    )
    exit()
    #4、pdf分析
    #4.1各降水等级 频次和水量贡献的 PDF
    compute_typhoon_vs_non_typhoon_precip_pdf(
    track_csv="newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t6h.csv",
    daily_nc_dir="newout/daily_precip_tps_500m",
    out_csv="newout/precip_pdf/typhoon_vs_non_typhoon_precip_pdf_2016_2025_5to9.csv",
    out_png="newout/precip_pdf/typhoon_vs_non_typhoon_precip_pdf_2016_2025_5to9.png",
    start_year=2016,
    end_year=2025,
    start_month=5,
    end_month=9,
    bins=[1.0, 10.0, 25.0, 50.0, 100.0,250.0, np.inf],
    bin_labels=["1-10 mm", "10-25 mm", "25-50 mm", "50-100 mm","100-250 mm", ">=250 mm"],
)
   
    #4.3台风日和非台风日降水分布的 K-S 检验
    compute_typhoon_vs_non_typhoon_precip_ks_test(
        track_csv="newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t6h.csv",
        daily_nc_dir="newout/daily_precip_tps_500m",
        out_csv="newout/precip_pdf/typhoon_vs_non_typhoon_precip_ks_2016_2025_5to9.csv",
        start_year=2016,
        end_year=2025,
        start_month=5,
        end_month=9,
        max_samples_per_pool=500_000,
        random_state=0,
        significance_level=0.01,
    )
    exit()

    
    

    


    



    
    
    
    
    
    
    
    #0. 确定台风日，其中输入csv文件来自 typhoon_days.py 计算结果
    typhoon_count, typhoon_days = get_typhoon_case_count_and_days(
        "newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_t6h.csv",
        start_month=5,
        end_month=9,
    )
    print("2016-2025年5-9月份的台风个数:", typhoon_count)
    print("台风日列表:")
    print(typhoon_days)
    print("台风日总数:", len(typhoon_days))
    
    exit()
    

    
    
    
    






    #3.R10mm / R10mm_rate
    compute_annual_r10mm(
        daily_nc_dir="out/daily_precip_tps_500m",
        out_count_dir="out/r10mm",
        out_rate_dir="out/r10mm_rate",
    )
    #4.R20mm / R20mm_rate
    compute_annual_r20mm(
        daily_nc_dir="out/daily_precip_tps_500m",
        out_count_dir="out/r20mm",
        out_rate_dir="out/r20mm_rate",
    )
    #5.R50mm / R50mm_rate
    compute_annual_r50mm(
        daily_nc_dir="out/daily_precip_tps_500m",
        out_count_dir="out/r50mm",
        out_rate_dir="out/r50mm_rate",
    )

    #1.Rx1day
    compute_annual_rx1day(
        daily_nc_dir="out/daily_precip_tps_500m",
        out_dir="out/rx1day",
    )
