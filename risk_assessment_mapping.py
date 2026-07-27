from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np

'''
山脉名称	重心北纬 (°)	重心东经 (°)	主峰名称	主峰北纬 (°)	主峰东经 (°)	主峰海拔 (m)
天目山（西天目主脉）	30.352	119.426	仙人顶	30.3495	119.4240	1506
四明山（宁波片区主体）	29.668	121.087	青虎湾岗	29.6652	121.0847	979.1
四明山（整条山脉全域）	29.602	120.840	金钟山	29.5972	120.8364	1018
括苍山	28.812	120.924	米筛浪	28.8100	120.9221	1382.4
北雁荡山	28.371	121.052	百岗尖（西峰）	28.3697	121.0503	1056.5
洞宫山脉（浙南核心）	27.760	119.220	黄茅尖	27.8890	119.1917	1929
千里岗山脉	主体核心重心为 29.34°N、118.55°E，主峰磨心尖坐标为 29.372°N、118.610°E。
'''

DEFAULT_HAZARD_NC = "newout/usri/hazard_r99p_prob_sdii50_2016_2025.nc"
DEFAULT_EXPOSURE_NC = "newout/usri/exposure_population_gdp_landcover_500m.nc"
DEFAULT_USRI_NC = "newout/usri/usri_2016_2025_500m.nc"
DEFAULT_ATTRIBUTION_NC = "newout/usri/usri_driver_type_500m.nc"
DEFAULT_OUT_DIR = "newout/usri/figs"
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"
DEFAULT_PERCENTILE_LEVELS = (5, 20, 50, 80, 90, 95)
DEFAULT_RISK_LABELS = ("低", "中", "中高", "高")
DEFAULT_DRIVER_LABELS = ("一般风险区", "危险性主导", "暴露度主导", "双高驱动")

MAJOR_MOUNTAINS = [
    {"name": "括苍山", "peak": "米筛浪", "lat": 28.812, "lon": 120.924, "elev_m": 1382.4, "dx": 0.04, "dy": 0.05},
    {"name": "北雁荡山", "peak": "百岗尖", "lat": 28.371, "lon": 121.052, "elev_m": 1056.5, "dx": 0.04, "dy": 0.05},
    {"name": "天目山", "peak": "仙人顶", "lat": 30.352, "lon": 119.426, "elev_m": 1506.0, "dx": 0.04, "dy": 0.05},
    {"name": "四明山-宁波", "peak": "青虎湾岗", "lat": 29.668, "lon": 121.087, "elev_m": 979.1, "dx": 0.04, "dy": 0.08},
   {"name": "南雁荡山", "peak": "-", "lat": 27.593367, "lon": 120.288351, "elev_m": 999, "dx": 0.04, "dy": 0.08},
    {"name": "千里岗", "peak": "磨心尖", "lat": 29.34, "lon": 118.55, "elev_m": 1523, "dx": 0.04, "dy": 0.08},
    #{"name": "洞宫山脉-浙南", "peak": "黄茅尖", "lat": 27.760, "lon": 119.220, "elev_m": 1929.0, "dx": 0.04, "dy": 0.05},
]


def _configure_matplotlib():
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


def _read_hazard_nc(nc_path: str | Path = DEFAULT_HAZARD_NC) -> dict[str, np.ndarray]:
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    variables = [
        "lat",
        "lon",
        "r99p",
        "prob",
        "sdii50",
        "norm_r99p",
        "norm_prob",
        "norm_sdii50",
        "hazard",
    ]
    out: dict[str, np.ndarray] = {}
    with nc.Dataset(path, "r") as ds:
        for name in variables:
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")
            data = ds.variables[name][:]
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            out[name] = np.asarray(data, dtype=np.float32)
    return out


def _read_exposure_nc(nc_path: str | Path = DEFAULT_EXPOSURE_NC) -> dict[str, np.ndarray]:
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    variables = [
        "lat",
        "lon",
        "pop_density",
        "gdp_density",
        "cropland_ratio",
        "builtup_ratio",
        "norm_pop_density",
        "norm_gdp_density",
        "norm_cropland_ratio",
        "norm_builtup_ratio",
        "exposure",
    ]
    out: dict[str, np.ndarray] = {}
    with nc.Dataset(path, "r") as ds:
        for name in variables:
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")
            data = ds.variables[name][:]
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            out[name] = np.asarray(data, dtype=np.float32)
    return out


def _read_usri_nc(nc_path: str | Path = DEFAULT_USRI_NC) -> dict[str, np.ndarray]:
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    variables = ["lat", "lon", "hazard", "exposure", "usri"]
    out: dict[str, np.ndarray] = {}
    with nc.Dataset(path, "r") as ds:
        for name in variables:
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")
            data = ds.variables[name][:]
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            out[name] = np.asarray(data, dtype=np.float32)
    return out


def _read_attribution_nc(nc_path: str | Path = DEFAULT_ATTRIBUTION_NC) -> dict[str, np.ndarray]:
    path = Path(nc_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    variables = [
        "lat",
        "lon",
        "hazard",
        "exposure",
        "usri",
        "driver_type",
        "r99p_contrib",
        "prob_contrib",
        "sdii50_contrib",
        "pop_contrib",
        "gdp_contrib",
        "cropland_contrib",
        "builtup_contrib",
    ]
    out: dict[str, np.ndarray] = {}
    with nc.Dataset(path, "r") as ds:
        for name in variables:
            if name not in ds.variables:
                raise ValueError(f"{path} 缺少变量 {name!r}")
            data = ds.variables[name][:]
            if np.ma.isMaskedArray(data):
                fill_value = np.nan if name != "driver_type" else -1
                data = data.filled(fill_value)
            if name == "driver_type":
                out[name] = np.asarray(data, dtype=np.int16)
            else:
                out[name] = np.asarray(data, dtype=np.float32)
    return out


def _load_zhejiang_boundary(province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH):
    import geopandas as gpd

    gdf = gpd.read_file(province_shp_path)
    if "NAME" in gdf.columns:
        gdf = gdf[gdf["NAME"] == "浙江省"]
    if gdf.empty:
        raise ValueError(f"在 {province_shp_path} 中未找到浙江省边界")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _set_map_extent(ax, boundary_gdf, pad_ratio: float = 0.04) -> None:
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    xpad = max((maxx - minx) * pad_ratio, 0.08)
    ypad = max((maxy - miny) * pad_ratio, 0.08)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35)


def _plot_mountain_markers(ax) -> None:
    for item in MAJOR_MOUNTAINS:
        ax.scatter(
            item["lon"],
            item["lat"],
            marker="^",
            s=46,
            color="black",
            edgecolor="white",
            linewidth=0.5,
            zorder=8,
        )
        ax.text(
            item["lon"] + item.get("dx", 0.035),
            item["lat"] + item.get("dy", 0.035),
            item["name"],
            fontsize=7.5,
            color="black",
            ha="left",
            va="bottom",
            zorder=9,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )


def _percentile_boundaries(
    data: np.ndarray,
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> np.ndarray:
    valid = np.asarray(data, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        raise ValueError("没有有效数据，无法按百分位数分级。")

    boundaries = [float(np.nanmin(valid))]
    boundaries.extend(float(x) for x in np.nanpercentile(valid, percentile_levels))
    boundaries.append(float(np.nanmax(valid)))
    boundaries = np.asarray(boundaries, dtype=np.float64)
    boundaries = np.unique(boundaries)
    if boundaries.size < 2:
        eps = max(abs(float(boundaries[0])) * 1e-6, 1e-6)
        boundaries = np.asarray([boundaries[0] - eps, boundaries[0] + eps], dtype=np.float64)
    return boundaries


def _percentile_tick_labels(
    boundaries: np.ndarray,
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
    decimals: int | None = None,
) -> list[str]:
    def fmt(value: float) -> str:
        if decimals is not None:
            if decimals == 0:
                return f"{value:.0f}"
            return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        abs_value = abs(float(value))
        if abs_value >= 100:
            return f"{value:.0f}"
        if abs_value >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if abs_value >= 1:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value:.3f}".rstrip("0").rstrip(".")

    labels: list[str] = []
    for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if i == 0:
            labels.append(f"<{fmt(right)}")
        elif i == len(boundaries) - 2:
            labels.append(f">{fmt(left)}")
        else:
            labels.append(f"{fmt(left)}-{fmt(right)}")
    return labels


def _add_map_colorbar(fig, ax, mesh, ticks: np.ndarray, tick_labels: list[str], label: str):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(mesh, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label(label)
    return cbar


def _weighted_sse(
    prefix_w: np.ndarray,
    prefix_wx: np.ndarray,
    prefix_wx2: np.ndarray,
    start_idx: int | np.ndarray,
    end_idx: int,
) -> np.ndarray:
    weight = prefix_w[end_idx + 1] - prefix_w[start_idx]
    weighted_sum = prefix_wx[end_idx + 1] - prefix_wx[start_idx]
    weighted_sq_sum = prefix_wx2[end_idx + 1] - prefix_wx2[start_idx]
    return weighted_sq_sum - (weighted_sum * weighted_sum / np.maximum(weight, 1.0))


def _natural_break_boundaries(
    data: np.ndarray,
    n_classes: int = 4,
    max_bins: int = 1024,
) -> np.ndarray:
    """
    用 Jenks 自然断点法划分连续值。

    对大数组先压缩为直方图频率表，再在频率表上最小化类内方差，避免对所有格点做高开销运算。
    """
    valid = np.asarray(data, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        raise ValueError("没有有效 USRI 数据，无法进行风险分级。")

    data_min = float(np.nanmin(valid))
    data_max = float(np.nanmax(valid))
    if np.isclose(data_min, data_max):
        eps = max(abs(data_min) * 1e-6, 1e-6)
        return np.linspace(data_min - eps, data_max + eps, n_classes + 1)

    unique_values, unique_counts = np.unique(valid, return_counts=True)
    if unique_values.size <= max_bins:
        values = unique_values.astype(np.float64)
        counts = unique_counts.astype(np.float64)
    else:
        counts_int, edges = np.histogram(valid, bins=max_bins, range=(data_min, data_max))
        centers = (edges[:-1] + edges[1:]) / 2.0
        keep = counts_int > 0
        values = centers[keep].astype(np.float64)
        counts = counts_int[keep].astype(np.float64)

    n_bins = values.size
    if n_bins <= n_classes:
        boundaries = np.linspace(data_min, data_max, n_classes + 1)
        boundaries[0] = data_min
        boundaries[-1] = data_max
        return boundaries.astype(np.float64)

    prefix_w = np.concatenate(([0.0], np.cumsum(counts)))
    prefix_wx = np.concatenate(([0.0], np.cumsum(counts * values)))
    prefix_wx2 = np.concatenate(([0.0], np.cumsum(counts * values * values)))

    dp = np.full((n_classes + 1, n_bins), np.inf, dtype=np.float64)
    backtrack = np.full((n_classes + 1, n_bins), -1, dtype=np.int32)
    for end_idx in range(n_bins):
        dp[1, end_idx] = _weighted_sse(prefix_w, prefix_wx, prefix_wx2, 0, end_idx)

    for cls in range(2, n_classes + 1):
        for end_idx in range(cls - 1, n_bins):
            candidate_starts = np.arange(cls - 1, end_idx + 1)
            prev_ends = candidate_starts - 1
            costs = dp[cls - 1, prev_ends] + _weighted_sse(
                prefix_w,
                prefix_wx,
                prefix_wx2,
                candidate_starts,
                end_idx,
            )
            best_pos = int(np.argmin(costs))
            dp[cls, end_idx] = float(costs[best_pos])
            backtrack[cls, end_idx] = int(candidate_starts[best_pos])

    split_starts: list[int] = []
    end_idx = n_bins - 1
    for cls in range(n_classes, 1, -1):
        start_idx = int(backtrack[cls, end_idx])
        split_starts.append(start_idx)
        end_idx = start_idx - 1
    split_starts.reverse()

    boundaries = [data_min]
    for start_idx in split_starts:
        boundaries.append(float((values[start_idx - 1] + values[start_idx]) / 2.0))
    boundaries.append(data_max)
    boundaries_arr = np.asarray(boundaries, dtype=np.float64)
    boundaries_arr = np.maximum.accumulate(boundaries_arr)
    if np.unique(boundaries_arr).size != n_classes + 1:
        boundaries_arr = np.nanpercentile(valid, np.linspace(0, 100, n_classes + 1))
        boundaries_arr[0] = data_min
        boundaries_arr[-1] = data_max
    return boundaries_arr


def _classify_by_boundaries(data: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid_mask = np.isfinite(arr)
    out[valid_mask] = np.digitize(arr[valid_mask], boundaries[1:-1], right=True).astype(np.float32)
    return out


def plot_hazard_spatial_distribution(
    hazard_nc: str | Path = DEFAULT_HAZARD_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/hazard_spatial_distribution_2016_2025.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    cmap: str = "YlOrRd",
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> str:
    """
    绘制浙江省 500 m 暴雨致灾危险性 Hazard 综合空间分布图，并标注主要山脉位置。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_hazard_nc(hazard_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    hazard = data["hazard"]
    boundaries = _percentile_boundaries(hazard, percentile_levels)
    cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=220, constrained_layout=True)
    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        hazard,
        shading="auto",
        cmap=cmap_obj,
        norm=norm,
    )
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=6)
    _plot_mountain_markers(ax)
    _set_map_extent(ax, boundary)

    ax.set_title("浙江省500 m暴雨致灾危险性空间分布")
    ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
    _add_map_colorbar(
        fig,
        ax,
        mesh,
        ticks,
        _percentile_tick_labels(boundaries, percentile_levels, decimals=2),
        "Hazard index",
    )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] Hazard 综合空间分布图输出 -> {out_path}")
    return str(out_path)


def plot_hazard_factor_maps(
    hazard_nc: str | Path = DEFAULT_HAZARD_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/hazard_factors_r99p_sdii50_prob_2016_2025.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> str:
    """
    绘制 R99p、SDII50 和 Prob 三个危险性驱动因子的 3-in-1 空间分布图。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_hazard_nc(hazard_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    panels = [
        ("(a) R99p", data["r99p"], "mm/day", "viridis", 0),
        ("(b) SDII$_{\\geq50mm}$", data["sdii50"], "mm/day", "plasma", 0),
        ("(c) Prob", data["prob"] * 100.0, "%", "cividis", 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=220, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.9, wspace=0.42)
    for ax, (title, arr, cbar_label, cmap, decimals) in zip(axes, panels):
        boundaries = _percentile_boundaries(arr, percentile_levels)
        cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)
        mesh = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap_obj, norm=norm)
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        _set_map_extent(ax, boundary)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_title(title)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
        _add_map_colorbar(
            fig,
            ax,
            mesh,
            ticks,
            _percentile_tick_labels(boundaries, percentile_levels, decimals=decimals),
            cbar_label,
        )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] Hazard 因子拼图输出 -> {out_path}")
    return str(out_path)


def plot_normalized_hazard_factor_maps(
    hazard_nc: str | Path = DEFAULT_HAZARD_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/hazard_normalized_factors_r99p_sdii50_prob_2016_2025.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> str:
    """
    绘制归一化后的 R99p、SDII50 和 Prob 三个危险性因子的 3 子图空间分布。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_hazard_nc(hazard_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    panels = [
        ("(a) Norm(R99p)", data["norm_r99p"], "Normalized R99p", "viridis"),
        ("(b) Norm(SDII$_{\\geq50mm}$)", data["norm_sdii50"], "Normalized SDII", "plasma"),
        ("(c) Norm(Prob)", data["norm_prob"], "Normalized Prob", "cividis"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=220, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.9, wspace=0.42)
    for ax, (title, arr, cbar_label, cmap) in zip(axes, panels):
        boundaries = _percentile_boundaries(arr, percentile_levels)
        cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)
        mesh = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap_obj, norm=norm)
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        _set_map_extent(ax, boundary)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_title(title)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
        _add_map_colorbar(
            fig,
            ax,
            mesh,
            ticks,
            _percentile_tick_labels(boundaries, percentile_levels, decimals=2),
            cbar_label,
        )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] 归一化 Hazard 因子拼图输出 -> {out_path}")
    return str(out_path)


def plot_exposure_spatial_distribution(
    exposure_nc: str | Path = DEFAULT_EXPOSURE_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/exposure_spatial_distribution_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    cmap: str = "YlOrRd",
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> str:
    """
    绘制浙江省 500 m 承灾体暴露度 Exposure 综合空间分布图。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_exposure_nc(exposure_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    exposure = data["exposure"]
    boundaries = _percentile_boundaries(exposure, percentile_levels)
    cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=220, constrained_layout=True)
    mesh = ax.pcolormesh(lon2d, lat2d, exposure, shading="auto", cmap=cmap_obj, norm=norm)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=6)
    _set_map_extent(ax, boundary)
    ax.set_title("浙江省500 m承灾体暴露度空间分布")
    ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
    _add_map_colorbar(
        fig,
        ax,
        mesh,
        ticks,
        _percentile_tick_labels(boundaries, percentile_levels, decimals=2),
        "Exposure index",
    )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] Exposure 综合空间分布图输出 -> {out_path}")
    return str(out_path)


def plot_exposure_factor_maps(
    exposure_nc: str | Path = DEFAULT_EXPOSURE_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/exposure_factors_population_gdp_landcover_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> str:
    """
    绘制人口、GDP、耕地占比和人造地表占比 4 个暴露度驱动因子的空间分布图。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_exposure_nc(exposure_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    panels = [
        ("(a) Population", data["pop_density"], "persons/km2", "viridis", 1),
        ("(b) GDP", data["gdp_density"], "10000 yuan / km2", "plasma", 0),
        ("(c) Cropland", data["cropland_ratio"] * 100.0, "%", "YlGn", 1),
        ("(d) Built-up", data["builtup_ratio"] * 100.0, "%", "OrRd", 1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), dpi=220, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.94, wspace=0.35, hspace=0.25)
    axes_flat = axes.ravel()
    for idx, (ax, (title, arr, cbar_label, cmap, decimals)) in enumerate(zip(axes_flat, panels)):
        boundaries = _percentile_boundaries(arr, percentile_levels)
        cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)
        mesh = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap_obj, norm=norm)
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        _set_map_extent(ax, boundary)
        if idx % 2 == 1:
            ax.set_ylabel("")
        if idx < 2:
            ax.set_xlabel("")
        ax.set_title(title)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
        _add_map_colorbar(
            fig,
            ax,
            mesh,
            ticks,
            _percentile_tick_labels(boundaries, percentile_levels, decimals=decimals),
            cbar_label,
        )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] Exposure 因子拼图输出 -> {out_path}")
    return str(out_path)


def plot_normalized_exposure_factor_maps(
    exposure_nc: str | Path = DEFAULT_EXPOSURE_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/exposure_normalized_factors_population_gdp_landcover_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    percentile_levels: tuple[int | float, ...] = DEFAULT_PERCENTILE_LEVELS,
) -> str:
    """
    绘制归一化后人口、GDP、耕地占比和人造地表占比 4 个暴露度因子的空间分布图。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_exposure_nc(exposure_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    panels = [
        ("(a) Norm(Population)", data["norm_pop_density"], "Normalized population", "viridis"),
        ("(b) Norm(GDP)", data["norm_gdp_density"], "Normalized GDP", "plasma"),
        ("(c) Norm(Cropland)", data["norm_cropland_ratio"], "Normalized cropland", "YlGn"),
        ("(d) Norm(Built-up)", data["norm_builtup_ratio"], "Normalized built-up", "OrRd"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), dpi=220, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.94, wspace=0.35, hspace=0.25)
    axes_flat = axes.ravel()
    for idx, (ax, (title, arr, cbar_label, cmap)) in enumerate(zip(axes_flat, panels)):
        boundaries = _percentile_boundaries(arr, percentile_levels)
        cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)
        mesh = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap_obj, norm=norm)
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        _set_map_extent(ax, boundary)
        if idx % 2 == 1:
            ax.set_ylabel("")
        if idx < 2:
            ax.set_xlabel("")
        ax.set_title(title)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
        _add_map_colorbar(
            fig,
            ax,
            mesh,
            ticks,
            _percentile_tick_labels(boundaries, percentile_levels, decimals=2),
            cbar_label,
        )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] 归一化 Exposure 因子拼图输出 -> {out_path}")
    return str(out_path)


def plot_usri_risk_zoning_map(
    usri_nc: str | Path = DEFAULT_USRI_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/usri_risk_zoning_2016_2025_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    risk_labels: tuple[str, ...] = DEFAULT_RISK_LABELS,
) -> str:
    """
    按自然断点法将 USRI 划分为低、中、中高、高 4 级，并绘制风险分区图。
    """
    if len(risk_labels) != 4:
        raise ValueError("risk_labels 必须包含 4 个等级名称。")

    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    data = _read_usri_nc(usri_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    usri = data["usri"]
    jenks_boundaries = _natural_break_boundaries(usri, n_classes=4)
    risk_class = _classify_by_boundaries(usri, jenks_boundaries)

    cmap_obj = ListedColormap(["#2ca25f", "#fee08b", "#fdae61", "#d73027"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap_obj.N)

    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=220, constrained_layout=True)
    mesh = ax.pcolormesh(lon2d, lat2d, risk_class, shading="auto", cmap=cmap_obj, norm=norm)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=6)
    _set_map_extent(ax, boundary)

    ax.set_title("浙江省500 m城市暴雨气候风险分区")
    cbar = _add_map_colorbar(
        fig,
        ax,
        mesh,
        np.arange(4),
        list(risk_labels),
        "USRI risk level",
    )
    cbar.ax.tick_params(length=0)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] USRI 风险分区图输出 -> {out_path}")
    print(f"[risk-map] USRI 自然断点 -> {', '.join(f'{x:.6f}' for x in jenks_boundaries)}")
    return str(out_path)


def plot_usri_driver_type_map(
    attribution_nc: str | Path = DEFAULT_ATTRIBUTION_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/usri_driver_type_map_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
    driver_labels: tuple[str, ...] = DEFAULT_DRIVER_LABELS,
) -> str:
    """
    绘制危险性/暴露度主导类型空间分布图。
    """
    if len(driver_labels) != 4:
        raise ValueError("driver_labels 必须包含 4 个类型名称。")

    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    data = _read_attribution_nc(attribution_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    driver_type = data["driver_type"].astype(np.float32)
    driver_type[driver_type < 0] = np.nan

    cmap_obj = ListedColormap(["#d9d9d9", "#4575b4", "#fdae61", "#d73027"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap_obj.N)

    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=220, constrained_layout=True)
    mesh = ax.pcolormesh(lon2d, lat2d, driver_type, shading="auto", cmap=cmap_obj, norm=norm)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=6)
    _set_map_extent(ax, boundary)
    ax.set_title("浙江省 500 m 暴雨风险主导驱动类型")

    cbar = _add_map_colorbar(fig, ax, mesh, np.arange(4), list(driver_labels), "Driver type")
    cbar.ax.tick_params(length=0)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] USRI 主导类型图输出 -> {out_path}")
    return str(out_path)


def _contribution_tick_labels(boundaries: np.ndarray) -> list[str]:
    labels: list[str] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        labels.append(f"{left:.0f}-{right:.0f}%")
    return labels


def plot_hazard_factor_contribution_maps(
    attribution_nc: str | Path = DEFAULT_ATTRIBUTION_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/usri_hazard_factor_contribution_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    """
    绘制 R99p、Prob、SDII50 对 Hazard 的贡献率空间分布图。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_attribution_nc(attribution_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    panels = [
        ("(a) R99p contribution", data["r99p_contrib"] * 100.0, "viridis"),
        ("(b) Prob contribution", data["prob_contrib"] * 100.0, "cividis"),
        ("(c) SDII$_{\\geq50mm}$ contribution", data["sdii50_contrib"] * 100.0, "plasma"),
    ]
    boundaries = np.asarray([0, 20, 40, 60, 80, 100], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), dpi=220, constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.9, wspace=0.42)
    for ax, (title, arr, cmap) in zip(axes, panels):
        cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)
        mesh = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap_obj, norm=norm)
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        _set_map_extent(ax, boundary)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ax.set_title(title)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
        _add_map_colorbar(fig, ax, mesh, ticks, _contribution_tick_labels(boundaries), "Contribution")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] Hazard 因子贡献率图输出 -> {out_path}")
    return str(out_path)


def plot_exposure_factor_contribution_maps(
    attribution_nc: str | Path = DEFAULT_ATTRIBUTION_NC,
    out_png: str | Path = f"{DEFAULT_OUT_DIR}/usri_exposure_factor_contribution_500m.png",
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> str:
    """
    绘制人口、GDP、耕地和建成区对 Exposure 的贡献率空间分布图。
    """
    plt = _configure_matplotlib()
    from matplotlib.colors import BoundaryNorm

    data = _read_attribution_nc(attribution_nc)
    boundary = _load_zhejiang_boundary(province_shp_path)

    lats = data["lat"]
    lons = data["lon"]
    lon2d, lat2d = np.meshgrid(lons, lats)
    panels = [
        ("(a) Population contribution", data["pop_contrib"] * 100.0, "viridis"),
        ("(b) GDP contribution", data["gdp_contrib"] * 100.0, "plasma"),
        ("(c) Cropland contribution", data["cropland_contrib"] * 100.0, "YlGn"),
        ("(d) Built-up contribution", data["builtup_contrib"] * 100.0, "OrRd"),
    ]
    boundaries = np.asarray([0, 20, 40, 60, 80, 100], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), dpi=220, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.94, wspace=0.35, hspace=0.25)
    axes_flat = axes.ravel()
    for idx, (ax, (title, arr, cmap)) in enumerate(zip(axes_flat, panels)):
        cmap_obj = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)
        mesh = ax.pcolormesh(lon2d, lat2d, arr, shading="auto", cmap=cmap_obj, norm=norm)
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.75, zorder=5)
        _set_map_extent(ax, boundary)
        if idx % 2 == 1:
            ax.set_ylabel("")
        if idx < 2:
            ax.set_xlabel("")
        ax.set_title(title)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2.0
        _add_map_colorbar(fig, ax, mesh, ticks, _contribution_tick_labels(boundaries), "Contribution")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk-map] Exposure 因子贡献率图输出 -> {out_path}")
    return str(out_path)


def plot_hazard_maps(
    hazard_nc: str | Path = DEFAULT_HAZARD_NC,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> dict[str, str]:
    out_dir_path = Path(out_dir)
    hazard_png = plot_hazard_spatial_distribution(
        hazard_nc=hazard_nc,
        out_png=out_dir_path / "hazard_spatial_distribution_2016_2025.png",
        province_shp_path=province_shp_path,
    )
    factor_png = plot_hazard_factor_maps(
        hazard_nc=hazard_nc,
        out_png=out_dir_path / "hazard_factors_r99p_sdii50_prob_2016_2025.png",
        province_shp_path=province_shp_path,
    )
    norm_factor_png = plot_normalized_hazard_factor_maps(
        hazard_nc=hazard_nc,
        out_png=out_dir_path / "hazard_normalized_factors_r99p_sdii50_prob_2016_2025.png",
        province_shp_path=province_shp_path,
    )
    return {"hazard_png": hazard_png, "factor_png": factor_png, "norm_factor_png": norm_factor_png}


def plot_exposure_maps(
    exposure_nc: str | Path = DEFAULT_EXPOSURE_NC,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> dict[str, str]:
    out_dir_path = Path(out_dir)
    exposure_png = plot_exposure_spatial_distribution(
        exposure_nc=exposure_nc,
        out_png=out_dir_path / "exposure_spatial_distribution_500m.png",
        province_shp_path=province_shp_path,
    )
    factor_png = plot_exposure_factor_maps(
        exposure_nc=exposure_nc,
        out_png=out_dir_path / "exposure_factors_population_gdp_landcover_500m.png",
        province_shp_path=province_shp_path,
    )
    norm_factor_png = plot_normalized_exposure_factor_maps(
        exposure_nc=exposure_nc,
        out_png=out_dir_path / "exposure_normalized_factors_population_gdp_landcover_500m.png",
        province_shp_path=province_shp_path,
    )
    return {"exposure_png": exposure_png, "factor_png": factor_png, "norm_factor_png": norm_factor_png}


def plot_usri_maps(
    usri_nc: str | Path = DEFAULT_USRI_NC,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> dict[str, str]:
    out_dir_path = Path(out_dir)
    risk_zoning_png = plot_usri_risk_zoning_map(
        usri_nc=usri_nc,
        out_png=out_dir_path / "usri_risk_zoning_2016_2025_500m.png",
        province_shp_path=province_shp_path,
    )
    return {"risk_zoning_png": risk_zoning_png}


def plot_attribution_maps(
    attribution_nc: str | Path = DEFAULT_ATTRIBUTION_NC,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    province_shp_path: str | Path = DEFAULT_PROVINCE_SHP_PATH,
) -> dict[str, str]:
    out_dir_path = Path(out_dir)
    driver_type_png = plot_usri_driver_type_map(
        attribution_nc=attribution_nc,
        out_png=out_dir_path / "usri_driver_type_map_500m.png",
        province_shp_path=province_shp_path,
    )
    hazard_contribution_png = plot_hazard_factor_contribution_maps(
        attribution_nc=attribution_nc,
        out_png=out_dir_path / "usri_hazard_factor_contribution_500m.png",
        province_shp_path=province_shp_path,
    )
    exposure_contribution_png = plot_exposure_factor_contribution_maps(
        attribution_nc=attribution_nc,
        out_png=out_dir_path / "usri_exposure_factor_contribution_500m.png",
        province_shp_path=province_shp_path,
    )
    return {
        "driver_type_png": driver_type_png,
        "hazard_contribution_png": hazard_contribution_png,
        "exposure_contribution_png": exposure_contribution_png,
    }


def plot_risk_assessment_maps() -> dict[str, dict[str, str]]:
    return {
        "hazard": plot_hazard_maps(),
        "exposure": plot_exposure_maps(),
        "usri": plot_usri_maps(),
        "attribution": plot_attribution_maps(),
    }


if __name__ == "__main__":
    plot_risk_assessment_maps()
