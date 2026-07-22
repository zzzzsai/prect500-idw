import netCDF4 as nc
import numpy as np
import os
from glob import glob
import pandas as pd
from datetime import datetime
import pickle
import json
import math


try:
    import rasterio
    from rasterio.merge import merge as rio_merge
except Exception:
    rasterio = None





def read_prect_1km(nc_path: str):
    zj_lat_range = [27.0, 31.2]
    zj_lon_range = [118.0, 123.2]
    with nc.Dataset(nc_path, mode='r') as ds:
        if 'lat' not in ds.variables or 'lon' not in ds.variables or 'prep' not in ds.variables or 'time' not in ds.variables:
            raise KeyError('nc 文件缺少必要变量: lat/lon/prep/time')
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
        # 计算浙江范围索引
        lat_min, lat_max = min(zj_lat_range), max(zj_lat_range)
        lon_min, lon_max = min(zj_lon_range), max(zj_lon_range)

        if lats.ndim == 1 and lons.ndim == 1:
            lat_mask = (lats >= lat_min) & (lats <= lat_max)
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                raise ValueError('给定经纬度范围在坐标上没有匹配到索引')
            i0, i1 = int(lat_idx[0]), int(lat_idx[-1] + 1)
            j0, j1 = int(lon_idx[0]), int(lon_idx[-1] + 1)
            lats_sub = lats[i0:i1]
            lons_sub = lons[j0:j1]
            # 只按子区域切片读取，避免整幅加载
            prep = ds.variables['prep'][:, i0:i1, j0:j1]
            ij_bounds = (i0, i1, j0, j1)
       
        time = ds.variables['time'][:]
        return lats_sub, lons_sub, prep, time, ij_bounds

def save_prect_zj_nc(src_nc: str, out_nc: str):
    """
    读取全国 1km 日降水 NetCDF，裁剪到浙江范围并保存为新 NetCDF。
    输出变量：lat(y), lon(x), time(t), prep(t,y,x)
    """
    lats, lons, prep, time, _ = read_prect_1km(src_nc)  # prep: (t, y, x)
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    with nc.Dataset(src_nc, "r") as src, nc.Dataset(out_nc, "w") as dst:
        t_len, y_len, x_len = prep.shape
        # dims
        dst.createDimension("time", t_len)
        dst.createDimension("y", y_len)
        dst.createDimension("x", x_len)
        # coords
        vlat = dst.createVariable("lat", "f4", ("y",))
        vlon = dst.createVariable("lon", "f4", ("x",))
        vtime = dst.createVariable("time", time.dtype.str.replace('<','').replace('>',''), ("time",))
        vlat[:] = lats.astype("float32")
        vlon[:] = lons.astype("float32")
        vtime[:] = time
        # copy time attrs if exist
        if "time" in src.variables:
            for k, v in src.variables["time"].__dict__.items():
                setattr(vtime, k, v)
        # data
        vprep = dst.createVariable("prep", "f4", ("time", "y", "x"), zlib=True, complevel=4)
        vprep[:] = prep.astype("float32")
        # copy attrs if exist
        if "prep" in src.variables:
            for k, v in src.variables["prep"].__dict__.items():
                if k not in ("_FillValue", "scale_factor", "add_offset"):
                    setattr(vprep, k, v)
        vprep.long_name = getattr(vprep, "long_name", "Daily precipitation")
        vprep.units = getattr(vprep, "units", "mm/day")

def batch_crop_prect_zj(base_dir: str):
    """
    批量裁剪 /nas/zsai/prect_1km/China_1km_prep_*.nc 为 ZJ_1km_prep_*.nc
    """
    import re
    pattern = os.path.join(base_dir, "China_1km_prep_*.nc")
    files = sorted(glob(pattern))
    if not files:
        print("未找到输入文件：", pattern)
        return
    for fp in files:
        name = os.path.basename(fp)
        m = re.search(r"(\d{4})", name)
        year = m.group(1) if m else ""
        out_name = f"ZJ_1km_prep_{year}.nc" if year else name.replace("China_1km_", "ZJ_")
        out_fp = os.path.join(base_dir, out_name)
        if os.path.exists(out_fp):
            print("已存在，跳过：", out_fp)
            continue
        print("裁剪：", fp, "->", out_fp)
        save_prect_zj_nc(fp, out_fp)
        print("完成：", out_fp)

def read_landcover_zj(nc_path: str):
    """
    (117.5, 26.5, 123.5, 31.5)
    读取地表覆盖 NetCDF（包含纬度/经度与分类值），并裁剪到浙江范围。
    返回：lats_sub, lons_sub, lc_sub
    """
    zj_lat_range = [26.5, 31.5]
    zj_lon_range = [117.5, 123.5]
    with nc.Dataset(nc_path, mode='r') as ds:
        # 兼容不同变量命名
        lat_name = 'lat' if 'lat' in ds.variables else ('latitude' if 'latitude' in ds.variables else None)
        lon_name = 'lon' if 'lon' in ds.variables else ('longitude' if 'longitude' in ds.variables else None)
        if lat_name is None or lon_name is None:
            raise KeyError('nc 文件缺少必要坐标变量: lat/lon 或 latitude/longitude')
        lats = ds.variables[lat_name][:]
        lons = ds.variables[lon_name][:]
        # 数据变量名尝试
        cand = ['lccs_class', 'lc', 'landcover', 'band1']
        var = None
        for k in cand:
            if k in ds.variables:
                var = ds.variables[k]
                break
        if var is None:
            # 退一步：取第一个非坐标变量
            for k, v in ds.variables.items():
                if k not in [lat_name, lon_name]:
                    var = v
                    break
        if var is None:
            raise KeyError('未找到地表覆盖数据变量（如 lccs_class/lc/landcover）')

        # 浙江裁剪
        lat_min, lat_max = min(zj_lat_range), max(zj_lat_range)
        lon_min, lon_max = min(zj_lon_range), max(zj_lon_range)
        if lats.ndim == 1 and lons.ndim == 1:
            lat_mask = (lats >= lat_min) & (lats <= lat_max)
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]
            if lat_idx.size == 0 or lon_idx.size == 0:
                raise ValueError('给定经纬度范围在坐标上没有匹配到索引')
            i0, i1 = int(lat_idx[0]), int(lat_idx[-1] + 1)
            j0, j1 = int(lon_idx[0]), int(lon_idx[-1] + 1)
            lats_sub = lats[i0:i1]
            lons_sub = lons[j0:j1]
            # 形状兼容：(y,x) 或 (t,y,x)，优先尝试后两维空间
            if var.ndim == 2:
                lc_sub = var[i0:i1, j0:j1]
            elif var.ndim >= 3:
                lc_sub = var[0, i0:i1, j0:j1]
            else:
                raise ValueError('地表覆盖数据维度异常')
        else:
            # 2D 坐标（罕见），做包围盒切片
            lat2d = np.array(lats)
            lon2d = np.array(lons)
            mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
            rows, cols = np.where(mask)
            if rows.size == 0:
                raise ValueError('给定经纬度范围在 2D 坐标上没有匹配到索引')
            i0, i1 = int(rows.min()), int(rows.max() + 1)
            j0, j1 = int(cols.min()), int(cols.max() + 1)
            lats_sub = lat2d[i0:i1, j0:j1]
            lons_sub = lon2d[i0:i1, j0:j1]
            if var.ndim == 2:
                lc_sub = var[i0:i1, j0:j1]
            elif var.ndim >= 3:
                lc_sub = var[0, i0:i1, j0:j1]
            else:
                raise ValueError('地表覆盖数据维度异常')
    return lats_sub, lons_sub, np.array(lc_sub)

def save_landcover_zj_nc(src_nc: str, out_nc: str):
    """
    裁剪地表覆盖到浙江范围后，保存为新的 NetCDF 文件。
    变量：
      - lat, lon（1D 或 2D，取决于输入）
      - landcover(y,x)（int16）
    """
    lats, lons, lc = read_landcover_zj(src_nc)
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    with nc.Dataset(out_nc, "w") as ds_out:
        if np.ndim(lats) == 1 and np.ndim(lons) == 1:
            ny, nx = len(lats), len(lons)
            ds_out.createDimension("y", ny)
            ds_out.createDimension("x", nx)
            vlat = ds_out.createVariable("lat", "f4", ("y",))
            vlon = ds_out.createVariable("lon", "f4", ("x",))
            vlat[:] = lats.astype("float32")
            vlon[:] = lons.astype("float32")
            v = ds_out.createVariable("landcover", "i2", ("y", "x"), zlib=True, complevel=4, fill_value=-32768)
            v[:] = lc.astype("int16")
        else:
            ny, nx = lc.shape
            ds_out.createDimension("y", ny)
            ds_out.createDimension("x", nx)
            vlat = ds_out.createVariable("lat", "f4", ("y", "x"))
            vlon = ds_out.createVariable("lon", "f4", ("y", "x"))
            vlat[:] = lats.astype("float32")
            vlon[:] = lons.astype("float32")
            v = ds_out.createVariable("landcover", "i2", ("y", "x"), zlib=True, complevel=4, fill_value=-32768)
            v[:] = lc.astype("int16")
        v.long_name = "Land cover class (LCCS)"
        v.grid_mapping = "crs"
        crs = ds_out.createVariable("crs", "i4")
        crs.spatial_ref = "EPSG:4326"
    return (ny, nx), (lats.min(), lats.max(), lons.min(), lons.max())

def mosaic_dem_tiles_to_nc(
    dem_folder: str,
    out_nc: str,
    target_bounds: tuple = None,   # (min_lon, min_lat, max_lon, max_lat)
    target_res: float = None       # 像素分辨率（度），如 1/3600≈0.0002778（约30m）
):
    """
    读取 dem_folder 内的 GeoTIFF/HGT 瓦片并镶嵌到统一网格，缺失瓦片区域以 NaN 填充，输出 NetCDF。
    """
    if rasterio is None:
        raise RuntimeError("需要安装 rasterio：pip install rasterio")

    paths = sorted(glob(os.path.join(dem_folder, "*.tif")) + glob(os.path.join(dem_folder, "*.tiff")) + glob(os.path.join(dem_folder, "*.hgt")))
    if not paths:
        raise FileNotFoundError(f"在 {dem_folder} 未找到 *.tif/*.tiff/*.hgt")

    srcs = [rasterio.open(p) for p in paths]
    # 自动 bounds/res
    if target_bounds is None:
        minx = min(s.bounds.left for s in srcs)
        miny = min(s.bounds.bottom for s in srcs)
        maxx = max(s.bounds.right for s in srcs)
        maxy = max(s.bounds.top for s in srcs)
        target_bounds = (minx, miny, maxx, maxy)
    if target_res is None:
        tr = srcs[0].transform
        target_res = abs(tr.a)

    mosaic, transform = rio_merge(
        srcs,
        bounds=target_bounds,
        res=target_res,
        nodata=np.nan,
        dtype="float32",
        method="first",
    )
    for s in srcs:
        s.close()

    mosaic = mosaic[0]  # (H, W)
    H, W = mosaic.shape
    a, b, c, d, e, f = transform[:6]
    lons = c + a * (np.arange(W) + 0.5)
    lats = f + e * (np.arange(H) + 0.5)

    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    with nc.Dataset(out_nc, "w") as ds_out:
        ds_out.createDimension("y", H)
        ds_out.createDimension("x", W)
        vlat = ds_out.createVariable("lat", "f4", ("y",))
        vlon = ds_out.createVariable("lon", "f4", ("x",))
        vlat[:] = lats.astype("float32")
        vlon[:] = lons.astype("float32")
        vdem = ds_out.createVariable("dem", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.nan)
        vdem.long_name = "Digital Elevation Model"
        vdem.units = "m"
        vdem[:] = mosaic.astype("float32")
        crs = ds_out.createVariable("crs", "i4")
        crs.spatial_ref = "EPSG:4326"
    return (H, W), (lats.min(), lats.max(), lons.min(), lons.max())


def resample_gebco_nc_to_grid(
    gebco_nc: str,
    out_nc: str,
    target_bounds: tuple,  # (min_lon, min_lat, max_lon, max_lat)
    target_res: float,     # 像素分辨率（度），如 10.0/3600.0
    elev_var: str = "elevation",
) -> tuple:
    """
    将 GEBCO 等单幅 NetCDF 高程文件重采样到指定分辨率的规则网格，输出格式与 mosaic_dem_tiles_to_nc 一致。

    参数：
      - gebco_nc: 输入的 GEBCO/高程 NetCDF 路径（需含 1D lat, lon 及高程变量）
      - out_nc: 输出 NetCDF 路径
      - target_bounds: (min_lon, min_lat, max_lon, max_lat)
      - target_res: 目标分辨率（度）
      - elev_var: 高程变量名，默认 "elevation"

    输出 NetCDF 格式（与 mosaic_dem_tiles_to_nc 一致）：
      - 维度: y, x
      - 变量: lat(y), lon(x), dem(y,x), crs
      - dem: float32, long_name="Digital Elevation Model", units="m"

    返回: ((H, W), (lat_min, lat_max, lon_min, lon_max))
    """
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        raise RuntimeError("需要 scipy 以进行网格重采样：pip install scipy")

    min_lon, min_lat, max_lon, max_lat = target_bounds
    with nc.Dataset(gebco_nc, "r") as ds:
        lat_src = np.array(ds.variables["lat"][:], dtype=np.float64)
        lon_src = np.array(ds.variables["lon"][:], dtype=np.float64)
        elev = np.array(ds.variables[elev_var][:], dtype=np.float64)
        # 处理可能的 fill_value / 掩码
        fv = getattr(ds.variables[elev_var], "_FillValue", None)
        if fv is not None:
            elev = np.where(elev == fv, np.nan, elev)
        if hasattr(elev, "mask") and np.ma.is_masked(elev):
            elev = np.ma.filled(elev, np.nan)

    # 确保 lat 自南向北递增（RegularGridInterpolator 要求坐标递增）
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        elev = elev[::-1, :]

    # 目标网格（像素中心，与 mosaic_dem_tiles_to_nc 一致）
    nx = int(round((max_lon - min_lon) / target_res))
    ny = int(round((max_lat - min_lat) / target_res))
    lons_out = min_lon + target_res * (np.arange(nx) + 0.5)
    lats_out = min_lat + target_res * (np.arange(ny) + 0.5)

    # 2D 插值
    interp = RegularGridInterpolator(
        (lat_src, lon_src), elev, method="linear", bounds_error=False, fill_value=np.nan
    )
    lon2d, lat2d = np.meshgrid(lons_out, lats_out)
    pts = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    dem_out = interp(pts).reshape(ny, nx).astype(np.float32)

    os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
    with nc.Dataset(out_nc, "w") as ds_out:
        ds_out.createDimension("y", ny)
        ds_out.createDimension("x", nx)
        vlat = ds_out.createVariable("lat", "f4", ("y",))
        vlon = ds_out.createVariable("lon", "f4", ("x",))
        vlat[:] = lats_out.astype("float32")
        vlon[:] = lons_out.astype("float32")
        vdem = ds_out.createVariable("dem", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.nan)
        vdem.long_name = "Digital Elevation Model"
        vdem.units = "m"
        vdem[:] = dem_out
        crs = ds_out.createVariable("crs", "i4")
        crs.spatial_ref = "EPSG:4326"
    return (ny, nx), (lats_out.min(), lats_out.max(), lons_out.min(), lons_out.max())


def merge_dem_srtm_gebco(
    srtm_nc: str,
    gebco_nc: str,
    out_nc: str,
    srtm_var: str = "dem",
    gebco_var: str = "dem",
) -> tuple:
    """
    将两个同网格 DEM 合并：SRTM 优先，SRTM 为 NaN 时用 GEBCO 填补。

    要求两个 NC 文件网格一致（lat, lon 相同），格式为 lat(y), lon(x), dem(y,x)。

    参数：
      - srtm_nc: SRTM 来源 DEM 路径（可能存在 NaN）
      - gebco_nc: GEBCO 来源 DEM 路径（用作填补）
      - out_nc: 输出 NetCDF 路径
      - srtm_var, gebco_var: 两文件中的高程变量名，默认均为 "dem"

    返回: (shape, extent) 与 mosaic_dem_tiles_to_nc 一致
    """
    with nc.Dataset(srtm_nc, "r") as ds1, nc.Dataset(gebco_nc, "r") as ds2:
        lat1 = np.array(ds1.variables["lat"][:])
        lon1 = np.array(ds1.variables["lon"][:])
        dem1 = np.array(ds1.variables[srtm_var][:], dtype=np.float64)
        fv1 = getattr(ds1.variables[srtm_var], "_FillValue", None)
        lat2 = np.array(ds2.variables["lat"][:])
        lon2 = np.array(ds2.variables["lon"][:])
        dem2 = np.array(ds2.variables[gebco_var][:], dtype=np.float64)

    if dem1.shape != dem2.shape:
        raise ValueError(f"两 DEM 网格不一致: {dem1.shape} vs {dem2.shape}")
    # 处理 lat 顺序可能相反的情况（如 SRTM 降序、GEBCO 升序）
    if np.allclose(lat1, lat2[::-1]):
        dem2 = dem2[::-1, :]
        lat2 = lat2[::-1]
    elif not np.allclose(lat1, lat2):
        raise ValueError("两 DEM 的 lat 坐标不一致且无法通过翻转对齐")
    if not np.allclose(lon1, lon2):
        raise ValueError("两 DEM 的 lon 坐标不一致")

    if fv1 is not None:
        dem1 = np.where(dem1 == fv1, np.nan, dem1)
    if hasattr(dem1, "mask") and np.ma.is_masked(dem1):
        dem1 = np.ma.filled(dem1, np.nan)

    merged = np.where(np.isfinite(dem1), dem1, dem2).astype(np.float32)

    os.makedirs(os.path.dirname(out_nc) or ".", exist_ok=True)
    with nc.Dataset(out_nc, "w") as ds_out:
        ds_out.createDimension("y", merged.shape[0])
        ds_out.createDimension("x", merged.shape[1])
        vlat = ds_out.createVariable("lat", "f4", ("y",))
        vlon = ds_out.createVariable("lon", "f4", ("x",))
        vlat[:] = lat1.astype("float32")
        vlon[:] = lon1.astype("float32")
        vdem = ds_out.createVariable("dem", "f4", ("y", "x"), zlib=True, complevel=4, fill_value=np.nan)
        vdem.long_name = "Digital Elevation Model"
        vdem.units = "m"
        vdem[:] = merged
        crs = ds_out.createVariable("crs", "i4")
        crs.spatial_ref = "EPSG:4326"

    extent = (lat1.min(), lat1.max(), lon1.min(), lon1.max())
    return merged.shape, extent


def plot_dem_nc(
    nc_path: str,
    out_png: str = None,
    cmap: str = "viridis",
    province_shp_path: str | None = None,
):
    """
    读取 NetCDF DEM 文件（包含变量 lat(y), lon(x), dem(y,x)），绘制 DEM 分布图到 PNG。
    """
    with nc.Dataset(nc_path, "r") as ds:
        lats = ds.variables.get("lat")[:]
        lons = ds.variables.get("lon")[:]
        dem = ds.variables.get("dem")[:]
    # 简单绘图（不依赖地图投影）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon2d, lat2d = np.meshgrid(lons, lats)
    plt.figure(figsize=(8, 6), dpi=150)
    dem[dem<0]=0
    m = plt.pcolormesh(lon2d, lat2d, dem, shading="auto", cmap=cmap)
    plt.colorbar(m, label="Elevation (m)")
    if province_shp_path:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(province_shp_path)
            gdf=gdf[gdf['NAME']=='浙江省']
            #print(gdf)
            #exit()
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            gdf.boundary.plot(ax=plt.gca(), color="black", linewidth=0.8, zorder=5)
            minx, miny, maxx, maxy = gdf.total_bounds
            xpad = (maxx - minx) * 0.03
            ypad = (maxy - miny) * 0.03
            plt.xlim(minx - xpad, maxx + xpad)
            plt.ylim(miny - ypad, maxy + ypad)
        except Exception as e:
            print(f"[warn] overlay province boundary failed: {e}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"DEM: {os.path.basename(nc_path)}")
    plt.tight_layout()
    if out_png is None:
        out_png = os.path.splitext(nc_path)[0] + ".png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_landcover_nc(
    nc_path: str,
    out_png: str = None,
    var_name: str = "landcover",
    cmap_name: str = "tab20",
    classes_map: dict = None,
    title: str = None,
):
    """
    绘制 landcover 分类栅格（支持 1D/2D 经纬度坐标），使用离散色带与分类刻度。
    
    参数：
      - nc_path: NetCDF 文件路径（应包含 lat/lon 与 landcover 变量）
      - out_png: 输出 PNG 路径；None 则自动生成（同名 .png）
      - var_name: landcover 变量名（默认 "landcover"）
      - cmap_name: matplotlib 色带名（默认 "tab20"）
      - classes_map: 可选 {类别值: 标签字符串} 映射，用于颜色条刻度标签
      - title: 图标题；None 则自动生成
    """
    with nc.Dataset(nc_path, "r") as ds:
        lat_name, lon_name = _coord_names(ds)
        if lat_name is None or lon_name is None:
            raise KeyError("缺少经纬度坐标变量（lat/lon 或 latitude/longitude）")
        lats = np.array(ds.variables[lat_name][:])
        lons = np.array(ds.variables[lon_name][:])
        if var_name not in ds.variables:
            raise KeyError(f"未找到变量: {var_name}")
        v = ds.variables[var_name]
        arr = np.array(v[:])
        if arr.ndim >= 3:
            arr = arr[0, :, :]
        if arr.ndim != 2:
            raise ValueError(f"{var_name} 期望为 2D 或 3D（取第一层），当前为 {arr.ndim}D")
        fill_value = getattr(v, "_FillValue", None)
    # 掩码与分类值
    if fill_value is not None:
        arr = np.where(arr == fill_value, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    # 提取有效分类
    valid = np.isfinite(arr)
    if not np.any(valid):
        raise ValueError("无有效 landcover 分类数据可绘制")
    classes = np.unique(arr[valid].astype(int))
    classes = np.sort(classes)
    # 建立索引网格（将离散类别映射为 0..K-1）
    # 使用 searchsorted 要求 classes 升序
    arr_int = np.full_like(arr, fill_value=-1, dtype=np.int32)
    arr_flat = arr[valid].astype(int)
    idx_flat = np.searchsorted(classes, arr_flat)
    arr_int[valid] = idx_flat
    # 颜色与刻度
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    K = int(len(classes))
    base_cmap = plt.get_cmap(cmap_name, max(K, 3))
    cmap = ListedColormap(base_cmap.colors[:K])
    bounds = np.arange(-0.5, K + 0.5, 1.0)
    norm = BoundaryNorm(boundaries=bounds, ncolors=K)
    # 坐标
    if lats.ndim == 1 and lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    elif lats.ndim == 2 and lons.ndim == 2:
        lat2d, lon2d = lats, lons
    else:
        raise ValueError("经纬度坐标维度不受支持")
    # 绘制
    plt.figure(figsize=(10, 8), dpi=150)
    m = plt.pcolormesh(lon2d, lat2d, np.ma.masked_where(arr_int < 0, arr_int), shading="auto", cmap=cmap, norm=norm)
    cbar = plt.colorbar(m, ticks=np.arange(K))
    if classes_map is not None:
        ticklabels = [classes_map.get(int(c), str(int(c))) for c in classes]
    else:
        ticklabels = [str(int(c)) for c in classes]
    cbar.ax.set_yticklabels(ticklabels)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    if title is None:
        title = f"Landcover ({var_name}) - {os.path.basename(nc_path)}"
    plt.title(title)
    plt.tight_layout()
    if out_png is None:
        root, _ = os.path.splitext(nc_path)
        out_png = root + f"_{var_name}.png"
    os.makedirs(os.path.dirname(out_png) if os.path.dirname(out_png) else ".", exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"[plot_landcover_nc] 已保存: {out_png}")


def generate_landcover_index_map(nc_path: str, out_pkl: str) -> str:
    """
    从 landcover NetCDF 文件中读取所有有效类别值，生成「原始类别值 -> 连续索引」的映射，
    以便后续在 CNN / GNN 中将 landcover 当作类别特征做 Embedding。

    映射规则：
      - 先去掉填充值与 NaN；
      - 对剩余的 landcover 值取整（int），按升序去重得到 classes；
      - 按 classes 的顺序分配索引 0..K-1，得到 {原始值: 索引}。

    参数：
      - nc_path:  已重采样到 DEM 网格的 landcover NetCDF 路径（例如 landcover_on_dem300m.nc）
      - out_pkl:  输出映射 pkl 文件路径（例如 landcover_index_map.pkl）

    返回：
      - out_pkl 路径字符串
    """
    with nc.Dataset(nc_path, "r") as ds:
        # 尝试找到 landcover 变量
        cand_vars = ["landcover", "lccs_class", "lc", "band1"]
        var = None
        var_name = None
        for name in cand_vars:
            if name in ds.variables:
                var = ds.variables[name]
                var_name = name
                break
        if var is None:
            # 回退：选取第一个非坐标变量作为 landcover
            lat_name, lon_name = _coord_names(ds)
            coord_names = {lat_name, lon_name}
            for name, v in ds.variables.items():
                if name not in coord_names:
                    var = v
                    var_name = name
                    break
        if var is None:
            raise KeyError("未能在 NetCDF 中找到 landcover 变量")

        arr = np.array(var[:])
        # 兼容 (t,y,x) 或 (y,x)，优先取第一时间层
        if arr.ndim >= 3:
            arr = arr[0, ...]
        if arr.ndim != 2:
            raise ValueError(f"landcover 变量期望为 2D 或 3D，当前为 {arr.ndim}D")

        fill_value = getattr(var, "_FillValue", None)
        if fill_value is not None:
            arr = np.where(arr == fill_value, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, np.nan)  # 掩掉 NaN / inf
        valid_mask = np.isfinite(arr)
        if not np.any(valid_mask):
            raise ValueError("landcover 栅格中无有效分类值")

        classes = np.unique(arr[valid_mask].astype(int))
        classes = np.sort(classes)

    mapping = {int(c): int(i) for i, c in enumerate(classes)}
    print(mapping)

    out_dir = os.path.dirname(out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(mapping, f)

    print(
        f"[generate_landcover_index_map] 从 {nc_path} 读取变量 {var_name}，"
        f"得到 {len(classes)} 个类别，已写出映射到: {out_pkl}"
    )
    return out_pkl


def compute_scaling_factors_for_training(
    years: list[int],
    cmorph_pattern: str,
    dem_nc: str,
    wind_pattern: str,
    station_obs_parquet: str,
    out_json: str,
) -> dict:
    """
    基于「原始 nc / parquet 文件」在训练年份上计算模型输入的缩放系数，并写出 scaling.json。

    训练年份：
      - years: 例如 [2014, 2015, 2016, 2017, 2018, 2020, 2021]

    数据来源：
      - 降水格点：cmorph_pattern
          /nas/zsai/prect500/cmorph_pcp/merged-30min_8km_daily/
          CMORPH_V1.0_ADJ_8km-daily_ZJ_merged_{year}.nc
        用于计算 pcp_q99_grid。
      - DEM：dem_nc
          /home/zsai/python_projects/prect500-data/static/dem_zhejiang_10s.nc
        用于计算 dem_max。
      - 背景风：wind_pattern
          /home/zsai/python_projects/prect500-data/era5_100m_uv_0p1/
          {year}_uv100_hourly_117.5-123.5_26.5-31.5.nc
        用于计算 u_q99_abs / v_q99_abs。
      - 站点观测：station_obs_parquet
          /home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024.parquet
        用于计算 pcp_q99_station（基于 P_obs_t）。

    输出：
      - 一个 dict，形如：
          {
            "pcp_q99_grid": ...,
            "pcp_q99_station": ...,
            "u_q99_abs": ...,
            "v_q99_abs": ...,
            "dem_max": ...
          }
      - 同时写出 out_json（例如 /nas/zsai/prect500/samples/scaling.json），
        仅包含 "factors" 字段，后续可由 organize_samples.py 或训练脚本扩展。
    """
    years = sorted(set(int(y) for y in years))

    # ---------- 1) 降水格点 q99 ----------
    pcp_vals = []
    for y in years:
        nc_path = cmorph_pattern.format(year=y)
        if not os.path.exists(nc_path):
            print(f"[compute_scaling] 警告：未找到降水 nc 文件: {nc_path}，跳过该年")
            continue
        print(f"[compute_scaling] 读取格点降水: {nc_path}")
        with nc.Dataset(nc_path, "r") as ds:
            # 尝试识别降水变量名
            cand_vars = ["prep", "pcp", "precip", "cmorph", "cmorph_pcp"]
            var = None
            for name in cand_vars:
                if name in ds.variables:
                    var = ds.variables[name]
                    break
            if var is None:
                # 回退：选第一个非坐标变量
                lat_name, lon_name = _coord_names(ds)
                coord_names = {lat_name, lon_name}
                for name, v in ds.variables.items():
                    if name not in coord_names:
                        var = v
                        break
            if var is None:
                raise KeyError(f"在 {nc_path} 中未找到降水变量")

            arr = np.array(var[:], dtype=np.float32)
            fill_value = getattr(var, "_FillValue", None)
            if fill_value is not None:
                arr = np.where(arr == fill_value, np.nan, arr)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                print(f"[compute_scaling] 警告：{nc_path} 中无有效降水值")
                continue
            # 为避免内存峰值，随机抽样一部分值参与分位数估计
            if arr.size > 5_000_000:
                idx = np.random.choice(arr.size, size=5_000_000, replace=False)
                arr = arr[idx]
            pcp_vals.append(arr)

    if not pcp_vals:
        raise ValueError("在指定年份中未收集到任何格点降水用于计算 pcp_q99_grid")
    pcp_all = np.concatenate(pcp_vals)
    pcp_q99_grid = float(np.nanpercentile(pcp_all, 99.0))

    # ---------- 2) DEM 最大值 ----------
    if not os.path.exists(dem_nc):
        raise FileNotFoundError(f"DEM 文件不存在: {dem_nc}")
    with nc.Dataset(dem_nc, "r") as ds:
        if "dem" not in ds.variables:
            raise KeyError(f"DEM 文件 {dem_nc} 中未找到 dem 变量")
        dem_arr = np.array(ds.variables["dem"][:], dtype=np.float32)
    dem_arr = dem_arr[np.isfinite(dem_arr)]
    if dem_arr.size == 0:
        raise ValueError("DEM 文件中无有效高程值")
    dem_max = float(np.nanmax(dem_arr))

    # ---------- 3) 背景风 u/v 的 |x| q99 ----------
    u_vals = []
    v_vals = []
    for y in years:
        w_path = wind_pattern.format(year=y)
        if not os.path.exists(w_path):
            print(f"[compute_scaling] 警告：未找到风场 nc 文件: {w_path}，跳过该年")
            continue
        print(f"[compute_scaling] 读取背景风: {w_path}")
        with nc.Dataset(w_path, "r") as ds:
            u_name = "u100" if "u100" in ds.variables else ("u" if "u" in ds.variables else None)
            v_name = "v100" if "v100" in ds.variables else ("v" if "v" in ds.variables else None)
            if u_name is None or v_name is None:
                raise KeyError(f"风文件 {w_path} 缺少 u/v 变量（期望 u100/v100 或 u/v）")
            u = np.array(ds.variables[u_name][:], dtype=np.float32)
            v = np.array(ds.variables[v_name][:], dtype=np.float32)
            u = np.abs(u[np.isfinite(u)])
            v = np.abs(v[np.isfinite(v)])
            if u.size == 0 or v.size == 0:
                print(f"[compute_scaling] 警告：{w_path} 中 u/v 数组无有效值")
                continue
            if u.size > 5_000_000:
                idx = np.random.choice(u.size, size=5_000_000, replace=False)
                u = u[idx]
            if v.size > 5_000_000:
                idx = np.random.choice(v.size, size=5_000_000, replace=False)
                v = v[idx]
            u_vals.append(u)
            v_vals.append(v)

    if not u_vals or not v_vals:
        raise ValueError("在指定年份中未收集到任何风场数据用于计算 u_q99_abs / v_q99_abs")
    u_all = np.concatenate(u_vals)
    v_all = np.concatenate(v_vals)
    u_q99_abs = float(np.nanpercentile(u_all, 99.0))
    v_q99_abs = float(np.nanpercentile(v_all, 99.0))

    # ---------- 4) 站点观测降水 q99（pcp_q99_station） ----------
    if not os.path.exists(station_obs_parquet):
        raise FileNotFoundError(f"站点观测 parquet 不存在: {station_obs_parquet}")
    df_obs = pd.read_parquet(station_obs_parquet)
    # 统一日期列为 datetime
    if "datetime" in df_obs.columns:
        dates = pd.to_datetime(df_obs["datetime"])
    elif "date" in df_obs.columns:
        dates = pd.to_datetime(df_obs["date"])
    else:
        raise ValueError("站点观测文件需包含 datetime 或 date 列")
    df_obs["_year"] = dates.dt.year
    df_train = df_obs[df_obs["_year"].isin(years)]
    if "P_obs_t" not in df_train.columns:
        raise ValueError("站点观测文件中缺少 P_obs_t 列")
    vals = df_train["P_obs_t"].to_numpy(dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("在训练年份中未找到有效的站点观测降水 P_obs_t")
    pcp_q99_station = float(np.nanpercentile(vals, 99.0))

    factors = {
        "pcp_q99_grid": pcp_q99_grid,
        "pcp_q99_station": pcp_q99_station,
        "u_q99_abs": u_q99_abs,
        "v_q99_abs": v_q99_abs,
        "dem_max": dem_max,
    }

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"factors": factors}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[compute_scaling] 已写出缩放系数到: {out_json}")
    print(f"[compute_scaling] factors = {factors}")
    return factors


def estimate_tweedie_power_from_station_obs(
    station_obs_parquet: str,
    value_col: str = "P_obs_t",
    date_col: str = "date",
    sid_col: str = "sid",
    block_length_days: int = 30,
    min_mean: float = 0.0,
    min_var: float = 0.0,
    drop_zero_only_blocks: bool = True,
    out_json: str | None = None,
) -> dict:
    """
    基于方差-均值幂律，估计降水的 Tweedie 幂参数 p。

    思路参考 “Stop using root-mean-square error as a precipitation target!” 一文：
      1) 按站点将日降水序列划分为等长的时间块（如 30 天）；
      2) 剔除包含缺测的时间块，保证样本完整；
      3) 对所有有效块，计算 (E[Y], Var(Y))，在 log-log 空间中做 OLS 线性回归：
            log Var(Y) = a + p * log E[Y]
         其中斜率 p 即 Tweedie 的幂参数。

    参数
    ----
    station_obs_parquet : str
        站点观测日降水 parquet 路径，例如
        "/home/zsai/python_projects/prect500-data/cmorph-stations/"
        "station_obs_merged_2014-2024-train-and-test.parquet"。
    value_col : str
        日降水列名，默认 "P_obs_t"。
    date_col : str
        日期列名，默认 "date"（若实际为 "datetime" 会自动兼容）。
    sid_col : str
        站点 ID 列名，默认 "sid"。
    block_length_days : int
        时间块长度（天数），文中示例为 30。
    min_mean : float
        仅保留块内平均降水 > min_mean 的时间块（过滤全 0 或近似 0 的块）。
    min_var : float
        仅保留块内方差 > min_var 的时间块。
    drop_zero_only_blocks : bool
        若为 True，则对于“全为 0 或 NaN”的时间块直接丢弃。
    out_json : str | None
        若给定，则将 { "p": ..., "intercept": ..., ... } 写出到该 JSON 文件。

    返回
    ----
    dict，包含：
      - "p": Tweedie 幂参数（回归斜率）
      - "intercept": 截距 a
      - "num_blocks": 参与回归的时间块数量
      - "r2": 在线性回归中的 R^2
    """
    if not os.path.exists(station_obs_parquet):
        raise FileNotFoundError(station_obs_parquet)

    df = pd.read_parquet(station_obs_parquet)

    # 兼容 "datetime" 列
    if date_col not in df.columns:
        if "datetime" in df.columns:
            date_col = "datetime"
        else:
            raise ValueError(f"找不到日期列: {date_col} 或 datetime")
    if value_col not in df.columns:
        raise ValueError(f"找不到降水列: {value_col}")
    if sid_col not in df.columns:
        raise ValueError(f"找不到站点列: {sid_col}")

    df = df[[sid_col, date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([sid_col, date_col]).reset_index(drop=True)

    all_means: list[float] = []
    all_vars: list[float] = []

    # 按站点分别构造等长时间块
    for sid, g in df.groupby(sid_col):
        g = g.sort_values(date_col)
        # 以完整的日频时间轴为基础，便于检查缺测
        full_index = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
        s = g.set_index(date_col)[value_col].reindex(full_index)

        n = len(full_index)
        if n < block_length_days:
            continue

        # 逐块滑动（不重叠）
        for start in range(0, n, block_length_days):
            end = start + block_length_days
            if end > n:
                break
            block = s.iloc[start:end]

            # 步骤 2：剔除包含缺测的时间块
            if block.isna().any():
                continue

            vals = block.to_numpy(dtype=float)
            if drop_zero_only_blocks and np.all(vals == 0.0):
                continue

            m = float(np.mean(vals))
            v = float(np.var(vals, ddof=1))  # 与文中一致用样本方差

            if m <= min_mean or v <= min_var:
                continue

            all_means.append(m)
            all_vars.append(v)

    all_means = np.asarray(all_means, dtype=float)
    all_vars = np.asarray(all_vars, dtype=float)

    if all_means.size < 2:
        raise RuntimeError(
            f"有效时间块数量过少，无法进行回归：num_blocks={all_means.size}。"
        )

    x = np.log(all_means)
    y = np.log(all_vars)

    # 普通最小二乘：y = a + p * x
    p, a = np.polyfit(x, y, deg=1)

    # 计算 R^2 诊断
    y_pred = a + p * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result = {
        "p": float(p),
        "intercept": float(a),
        "num_blocks": int(all_means.size),
        "block_length_days": int(block_length_days),
        "min_mean": float(min_mean),
        "min_var": float(min_var),
        "r2": float(r2),
    }

    print(
        f"[estimate_tweedie_power_from_station_obs] "
        f"num_blocks={result['num_blocks']}, p≈{result['p']:.3f}, "
        f"intercept≈{result['intercept']:.3f}, R^2≈{result['r2']:.3f}"
    )

    if out_json is not None:
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[estimate_tweedie_power_from_station_obs] 已写出结果到: {out_json}")

    return result

def plot_prect_first_frame(
    nc_path: str,
    out_png: str = None,
    cmap: str = "Blues",
    var_name: str = "prep",
    frame_index: int = 0,
    target_time=None,
    plot_total: bool = False,
    out_png_total: str = None,
    stations_df: pd.DataFrame = None,
    station_lat_col: str = "lat",
    station_lon_col: str = "lon",
    station_mask_col: str = None,
    station_mask_value: bool = True,
    scatter_color: str = "red",
    scatter_marker: str = "x",
    scatter_size: float = 50,
    scatter_alpha: float = 0.8,
):
    """
    读取格点日降水 NetCDF 文件的第一帧（第一个时间步），绘制降水分布图。
    
    参数：
      - nc_path: NetCDF 文件路径，应包含变量 lat(y), lon(x), time(t), prep(t,y,x)
      - target_time: 目标时间（优先于 frame_index），可为字符串/datetime/np.datetime64/pandas.Timestamp
      - out_png: 输出 PNG 路径；若为 None 则自动生成
      - cmap: 颜色映射，默认 "Blues"
      - var_name: 降水变量名，默认 "prep"
      - frame_index: 备用：当未提供 target_time 时，用索引选择
      - plot_total: 是否绘制全年的累计降水（对 time 维求和）
      - out_png_total: 年累计图输出路径；若 None 且 plot_total=True，则自动生成
      - stations_df: 站点 DataFrame，包含经纬度列
      - station_lat_col: 站点纬度列名，默认 "lat"
      - station_lon_col: 站点经度列名，默认 "lon"（也可能是 "lng"）
      - station_mask_col: 用于筛选站点的列名（如 "hrlt_pcp_avg"），若为 None 则绘制所有站点
      - station_mask_value: 筛选条件，当 station_mask_col 为列名时，筛选该列值为 True/NaN 的站点
      - scatter_color: 散点颜色，默认 "red"
      - scatter_marker: 散点标记，默认 "x"
      - scatter_size: 散点大小，默认 50
      - scatter_alpha: 散点透明度，默认 0.8
    """
    with nc.Dataset(nc_path, "r") as ds:
        lats = ds.variables.get("lat")[:]
        lons = ds.variables.get("lon")[:]
        
        prep_var = ds.variables.get(var_name)
        prep = prep_var[:].filled(np.nan)
        time = ds.variables.get("time", None)
        
        # 若提供了目标时间，则将其转换为 time 轴上的索引（优先于 frame_index）
        chosen_time_text = ""
        if target_time is not None:
            if time is None:
                raise ValueError("文件中缺少 time 变量，无法按时间选择帧")
            # 解析目标时间
            try:
                tgt_dt = pd.to_datetime(target_time)
            except Exception:
                raise ValueError(f"无法解析 target_time: {target_time}")
            # 读取 time 的单位与日历，转换为 Python datetime
            t_units = getattr(time, "units", None)
            t_cal = getattr(time, "calendar", "standard")
            if t_units is None:
                raise ValueError("time 缺少 units 属性，无法进行时间转换")
            from netCDF4 import num2date
            t_vals = np.asarray(time[:], dtype=float)
            t_dt = num2date(t_vals, units=t_units, calendar=t_cal, only_use_cftime_datetimes=False)
            # 统一为 pandas 时间索引
            t_pd = pd.to_datetime(list(t_dt))
            # 对于日尺度文件，常以 00:00:00 表示当日；将比较做规范化匹配
            tgt_norm = pd.to_datetime(pd.Timestamp(tgt_dt).normalize())
            t_norm = t_pd.normalize()
            matches = np.where(t_norm.values == tgt_norm.to_datetime64())[0]
            if matches.size == 0:
                # 没有精确匹配，则选择最近的时间步
                deltas = np.abs((t_pd - tgt_dt).to_numpy(dtype="timedelta64[s]").astype("int64"))
                nearest = int(np.argmin(deltas))
                frame_index = nearest
                chosen_time_text = f"{t_pd[nearest].isoformat()}"
                print(f"[plot_prect_first_frame] 未找到精确时间，已选择最近时间步: idx={nearest}, time={chosen_time_text}")
            else:
                frame_index = int(matches[0])
                chosen_time_text = f"{t_pd[frame_index].isoformat()}"
        else:
            # 未提供目标时间，若存在 time 变量，提取对应索引用以展示
            if time is not None:
                try:
                    t_units = getattr(time, "units", None)
                    t_cal = getattr(time, "calendar", "standard")
                    if t_units is not None:
                        from netCDF4 import num2date
                        t_val = float(np.asarray(time[:])[frame_index])
                        dt = num2date(t_val, units=t_units, calendar=t_cal, only_use_cftime_datetimes=False)
                        chosen_time_text = pd.to_datetime(dt).isoformat()
                except Exception:
                    chosen_time_text = ""
        
        # 获取填充值信息
        fill_value = getattr(prep_var, "_FillValue", None)
        missing_value = getattr(prep_var, "missing_value", None)
        
        # 读取第一帧
        year_sum = None
        if prep.ndim == 3:
            prep_frame = prep[frame_index, :, :]
            if plot_total:
                valid_counts = np.sum(np.isfinite(prep), axis=0)
                year_sum = np.nansum(prep, axis=0)
                year_sum[valid_counts == 0] = np.nan
        elif prep.ndim == 2:
            prep_frame = prep
        else:
            raise ValueError(f"降水变量维度应为 2 或 3，当前为 {prep.ndim}")
        
        # 获取时间信息（如果有）
        time_info = ""
        if chosen_time_text:
            time_info = f" (Time: {chosen_time_text})"
    
    # 处理填充值和无效值
    prep_frame = np.array(prep_frame, dtype=np.float64)
    
    # 处理 NetCDF 填充值
    if fill_value is not None and np.isfinite(fill_value):
        prep_frame[np.isclose(prep_frame, float(fill_value), atol=1e-6)] = np.nan
    if missing_value is not None and np.isfinite(missing_value):
        prep_frame[np.isclose(prep_frame, float(missing_value), atol=1e-6)] = np.nan
    
    # 处理其他无效值
    prep_frame = np.where(np.isfinite(prep_frame), prep_frame, np.nan)
    prep_frame = np.where(prep_frame >= 0, prep_frame, np.nan)  # 屏蔽负值

    year_sum_proc = None
    if year_sum is not None:
        year_sum_proc = np.array(year_sum, dtype=np.float64)
        if fill_value is not None and np.isfinite(fill_value):
            year_sum_proc[np.isclose(year_sum_proc, float(fill_value), atol=1e-6)] = np.nan
        if missing_value is not None and np.isfinite(missing_value):
            year_sum_proc[np.isclose(year_sum_proc, float(missing_value), atol=1e-6)] = np.nan
        year_sum_proc = np.where(np.isfinite(year_sum_proc), year_sum_proc, np.nan)
        year_sum_proc = np.where(year_sum_proc >= 0, year_sum_proc, np.nan)
    
    # 诊断信息
    valid_mask = ~np.isnan(prep_frame)
    valid_count = np.sum(valid_mask)
    total_count = prep_frame.size
    print(f"数据统计: 总格点数={total_count}, 有效格点数={valid_count}, 缺失率={(1-valid_count/total_count)*100:.2f}%")
    if valid_count > 0:
        print(f"有效值范围: [{np.nanmin(prep_frame):.2f}, {np.nanmax(prep_frame):.2f}] mm/day")
    
    # 处理站点数据
    station_lons = None
    station_lats = None
    if stations_df is not None:
        # 确定经度列名（可能是 "lon" 或 "lng"）
        lon_col = station_lon_col
        if lon_col not in stations_df.columns and "lng" in stations_df.columns:
            lon_col = "lng"
        elif lon_col not in stations_df.columns and "lon" in stations_df.columns:
            lon_col = "lon"
        
        if station_lat_col not in stations_df.columns:
            raise ValueError(f"站点 DataFrame 中缺少纬度列: {station_lat_col}")
        if lon_col not in stations_df.columns:
            raise ValueError(f"站点 DataFrame 中缺少经度列: {station_lon_col} 或 lng/lon")
        
        # 筛选站点
        if station_mask_col is not None and station_mask_col in stations_df.columns:
            if station_mask_value is True:
                # 筛选值为 NaN 的站点（当 station_mask_value=True 时，通常用于筛选缺失值）
                mask = stations_df[station_mask_col].isna()
            elif station_mask_value is False:
                # 筛选值不为 NaN 的站点
                mask = ~stations_df[station_mask_col].isna()
            else:
                # 筛选值为指定值的站点
                mask = stations_df[station_mask_col] == station_mask_value
            stations_subset = stations_df[mask].copy()
        else:
            stations_subset = stations_df.copy()
        
        if len(stations_subset) > 0:
            station_lats = stations_subset[station_lat_col].values
            station_lons = stations_subset[lon_col].values
            print(f"将叠加绘制 {len(stations_subset)} 个站点")
        else:
            print("筛选后没有站点需要绘制")
    
    # 绘图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    lon2d, lat2d = np.meshgrid(lons, lats)
    
    
    # 创建掩码数组
    prep_plot = np.ma.masked_invalid(prep_frame)
    
    plt.figure(figsize=(10, 8), dpi=150)
    m = plt.pcolormesh(lon2d, lat2d, prep_plot, shading="auto", cmap=cmap, vmin=0)
    plt.colorbar(m, label="Precipitation (mm/day)")
    
    # 叠加站点散点
    if station_lons is not None and station_lats is not None:
        plt.scatter(station_lons, station_lats, c=scatter_color, marker=scatter_marker, 
                   s=scatter_size, alpha=scatter_alpha, edgecolors='black', linewidths=0.5,
                   label=f'Stations (n={len(station_lons)})', zorder=5)
        plt.legend(loc='upper right')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Daily Precipitation - Frame {frame_index}{time_info}\n{os.path.basename(nc_path)}")
    plt.tight_layout()
    
    if out_png is None:
        out_png = os.path.splitext(nc_path)[0] + f"_frame_{frame_index}.png"
    os.makedirs(os.path.dirname(out_png) if os.path.dirname(out_png) else ".", exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    print(f"已保存图像: {out_png}")

    if plot_total and year_sum_proc is not None:
        plt.figure(figsize=(10, 8), dpi=150)
        sum_plot = np.ma.masked_invalid(year_sum_proc)
        m = plt.pcolormesh(lon2d, lat2d, sum_plot, shading="auto", cmap=cmap, vmin=0)
        plt.colorbar(m, label="Accumulated Precipitation (mm/year)")
        
        # 叠加站点散点（年累计图也叠加）
        if station_lons is not None and station_lats is not None:
            plt.scatter(station_lons, station_lats, c=scatter_color, marker=scatter_marker, 
                       s=scatter_size, alpha=scatter_alpha, edgecolors='black', linewidths=0.5,
                       label=f'Stations (n={len(station_lons)})', zorder=5)
            plt.legend(loc='upper right')
        
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Annual Accumulated Precipitation\n{os.path.basename(nc_path)}")
        plt.tight_layout()

        if out_png_total is None:
            out_png_total = os.path.splitext(nc_path)[0] + "_total.png"
        os.makedirs(os.path.dirname(out_png_total) if os.path.dirname(out_png_total) else ".", exist_ok=True)
        plt.savefig(out_png_total)
        plt.close()
        print(f"已保存年累计图像: {out_png_total}")




def _coord_names(ds: nc.Dataset) -> tuple[str | None, str | None]:
    """从 NetCDF 猜测纬度/经度坐标变量名。"""
    cand_lat = ("lat", "latitude", "y")
    cand_lon = ("lon", "longitude", "x")
    lat_name = next((c for c in cand_lat if c in ds.variables), None)
    lon_name = next((c for c in cand_lon if c in ds.variables), None)
    return lat_name, lon_name


def _spacing_summary(arr: np.ndarray) -> tuple[float, float, float, bool, str]:
    """
    计算一维坐标的步长统计：(min, max, mean, is_constant, monotonic)
    - 对 2D 坐标（curvilinear）应先选取沿行/列的一条剖面再传入。
    """
    vec = np.asarray(arr).astype(float)
    if vec.ndim != 1:
        raise ValueError("spacing_summary 仅接受 1D 向量")
    d = np.diff(vec)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan, np.nan, False, "unknown"
    dmin, dmax, dmean = float(np.min(d)), float(np.max(d)), float(np.mean(d))
    is_const = np.allclose(d, d[0], rtol=1e-8, atol=1e-12)
    if np.all(d > 0):
        mono = "asc"
    elif np.all(d < 0):
        mono = "desc"
    else:
        mono = "non-monotonic"
    return dmin, dmax, dmean, bool(is_const), mono


def check_nc_resolution(nc_path: str) -> dict:
    """
    检查单个 NetCDF 的网格分辨率与坐标信息。
    返回字典：包含 grid_type、坐标名、shape、步长统计等，并打印关键信息。
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)
    out: dict = {"file": nc_path}
    with nc.Dataset(nc_path, "r") as ds:
        lat_name, lon_name = _coord_names(ds)
        out["lat_name"] = lat_name
        out["lon_name"] = lon_name
        if lat_name is None or lon_name is None:
            # 可能为投影网格，仅有 x/y
            out["grid_type"] = "projected-or-unknown"
            print(f"[check_nc_resolution] {os.path.basename(nc_path)}: 未找到经纬度坐标（可能为投影栅格）。")
            return out
        lat = np.array(ds.variables[lat_name][:])
        lon = np.array(ds.variables[lon_name][:])
        print(lat,lon)
        if lat.ndim == 1 and lon.ndim == 1:
            out["grid_type"] = "rectilinear-1D"
            ny, nx = len(lat), len(lon)
            out["ny"], out["nx"] = int(ny), int(nx)
            dlat = _spacing_summary(lat)
            dlon = _spacing_summary(lon)
            out["lat_spacing"] = {"min": dlat[0], "max": dlat[1], "mean": dlat[2], "constant": dlat[3], "monotonic": dlat[4]}
            out["lon_spacing"] = {"min": dlon[0], "max": dlon[1], "mean": dlon[2], "constant": dlon[3], "monotonic": dlon[4]}
            print(f"[{os.path.basename(nc_path)}] grid={out['grid_type']} shape=({ny},{nx}) "
                  f"Δlat≈{dlat[2]:.8f} (min={dlat[0]:.8f}, max={dlat[1]:.8f}, const={dlat[3]}), "
                  f"Δlon≈{dlon[2]:.8f} (min={dlon[0]:.8f}, max={dlon[1]:.8f}, const={dlon[3]})")
        elif lat.ndim == 2 and lon.ndim == 2:
            out["grid_type"] = "curvilinear-2D"
            H, W = lat.shape
            out["ny"], out["nx"] = int(H), int(W)
            # 沿列取一条纬度剖面、沿行取一条经度剖面评估“近似步长”
            dlat = _spacing_summary(lat[:, 0])
            dlon = _spacing_summary(lon[0, :])
            out["lat_spacing_profile"] = {"min": dlat[0], "max": dlat[1], "mean": dlat[2], "constant": dlat[3], "monotonic": dlat[4]}
            out["lon_spacing_profile"] = {"min": dlon[0], "max": dlon[1], "mean": dlon[2], "constant": dlon[3], "monotonic": dlon[4]}
            print(f"[{os.path.basename(nc_path)}] grid={out['grid_type']} shape=({H},{W}) "
                  f"Δlat(profile)≈{dlat[2]:.8f}, Δlon(profile)≈{dlon[2]:.8f}")
        else:
            out["grid_type"] = "unknown-shape"
            print(f"[check_nc_resolution] {os.path.basename(nc_path)}: 坐标维度异常 lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")
    return out


def check_dem_and_landcover_resolution(
    dem_nc: str = "/home/zsai/python_projects/prect500-data/static/dem_zhejiang_10s.nc",
    lc_nc: str = "/home/zsai/python_projects/prect500-data/static/landcover_zj300m.nc",
) -> tuple[dict, dict]:
    """
    检查 DEM 与 landcover 的网格分辨率与类型，返回两个结果字典，并打印摘要。
    """
    print("== 检查 DEM 分辨率与网格类型 ==")
    res_dem = check_nc_resolution(dem_nc)
    print("== 检查 Landcover 分辨率与网格类型 ==")
    res_lc = check_nc_resolution(lc_nc)
    # 是否共格网的粗判断（仅对 1D rectilinear 情况）
    same_grid_hint = False
    if res_dem.get("grid_type") == "rectilinear-1D" and res_lc.get("grid_type") == "rectilinear-1D":
        dlat_dem = res_dem["lat_spacing"]["mean"]
        dlon_dem = res_dem["lon_spacing"]["mean"]
        dlat_lc = res_lc["lat_spacing"]["mean"]
        dlon_lc = res_lc["lon_spacing"]["mean"]
        same_grid_hint = (
            np.isfinite(dlat_dem) and np.isfinite(dlat_lc) and
            np.isfinite(dlon_dem) and np.isfinite(dlon_lc) and
            np.isclose(dlat_dem, dlat_lc, rtol=1e-6, atol=1e-12) and
            np.isclose(dlon_dem, dlon_lc, rtol=1e-6, atol=1e-12)
        )
    print(f"是否可能共格网（基于 1D 坐标步长粗判）: {bool(same_grid_hint)}")
    return res_dem, res_lc


def _nearest_indices_1d(src_vec: np.ndarray, dst_vec: np.ndarray) -> np.ndarray:
    """
    为 dst_vec 中每个值在单调 src_vec 上找到最近邻索引（向量化，返回 shape=(len(dst_vec),)）。
    兼容升序/降序。src_vec 必须 1D 单调。
    """
    s = np.asarray(src_vec)
    d = np.asarray(dst_vec)
    asc = s[0] < s[-1]
    if asc:
        idx = np.searchsorted(s, d, side="left")
        idx = np.clip(idx, 1, len(s) - 1)
        left = idx - 1
        right = idx
    else:
        s_rev = s[::-1]
        idx_r = np.searchsorted(s_rev, d, side="left")
        idx_r = np.clip(idx_r, 1, len(s_rev) - 1)
        left = (len(s_rev) - idx_r) - 1
        right = (len(s_rev) - (idx_r - 1)) - 1
    choose_right = np.abs(s[right] - d) < np.abs(s[left] - d)
    return np.where(choose_right, right, left).astype(np.int64)


def resample_landcover_to_dem_grid(
    lc_nc: str,
    dem_nc: str,
    out_nc: str,
    lc_var_candidates: tuple = ("landcover", "lccs_class", "lc", "band1"),
) -> tuple[int, int]:
    """
    将 landcover NetCDF 重采样到 DEM NetCDF 的经纬度网格（最近邻），并写出新 NetCDF。
    - dem_nc 需包含 1D lat(y), lon(x)
    - lc_nc 支持 1D rectilinear（推荐）或 2D curvilinear（需 scipy.spatial.cKDTree）
    返回：(ny, nx) 目标分辨率。
    """
    if not os.path.exists(lc_nc):
        raise FileNotFoundError(lc_nc)
    if not os.path.exists(dem_nc):
        raise FileNotFoundError(dem_nc)
    # 读取 DEM 目标网格
    with nc.Dataset(dem_nc, "r") as dds:
        dlat = np.array(dds.variables["lat"][:])
        dlon = np.array(dds.variables["lon"][:])
    ny, nx = len(dlat), len(dlon)
    # 读取 landcover 源
    with nc.Dataset(lc_nc, "r") as lds:
        l_lat_name, l_lon_name = _coord_names(lds)
        if l_lat_name is None or l_lon_name is None:
            raise KeyError("landcover 文件缺少经纬度坐标")
        slat = np.array(lds.variables[l_lat_name][:])
        slon = np.array(lds.variables[l_lon_name][:])
        var = None
        for name in lc_var_candidates:
            if name in lds.variables:
                var = lds.variables[name]
                break
        if var is None:
            for k, v in lds.variables.items():
                if k not in (l_lat_name, l_lon_name):
                    var = v
                    break
        if var is None:
            raise KeyError("未找到 landcover 数据变量")
        if var.ndim == 2:
            src_grid = np.array(var[:])
        elif var.ndim >= 3:
            src_grid = np.array(var[0, :, :])
        else:
            raise ValueError("landcover 数据维度异常")
        fill_value = getattr(var, "_FillValue", None)
    # 最近邻重采样
    if slat.ndim == 1 and slon.ndim == 1:
        iy = _nearest_indices_1d(slat, dlat)           # (ny,)
        ix = _nearest_indices_1d(slon, dlon)           # (nx,)
        out_grid = src_grid[iy[:, None], ix[None, :]]  # (ny, nx)
    elif slat.ndim == 2 and slon.ndim == 2:
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:
            raise RuntimeError("需要 scipy 以在 2D 曲线网格上进行最近邻重采样：pip install scipy")
        pts_src = np.column_stack([slat.ravel(), slon.ravel()]).astype(np.float64)
        tree = cKDTree(pts_src)
        lon2d, lat2d = np.meshgrid(dlon, dlat)
        pts_dst = np.column_stack([lat2d.ravel(), lon2d.ravel()]).astype(np.float64)
        _, jj = tree.query(pts_dst, k=1, workers=-1)
        out_grid = src_grid.ravel()[jj].reshape(ny, nx)
    else:
        raise ValueError("landcover 坐标维度不受支持")
    # 清理与类型
    out_grid = np.array(out_grid)
    if fill_value is not None:
        out_grid = np.where(out_grid == fill_value, -32768, out_grid)
    out_grid = np.where(np.isfinite(out_grid), out_grid, -32768)
    out_grid = out_grid.astype(np.int16, copy=False)
    # 写出
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    with nc.Dataset(out_nc, "w") as ods:
        ods.createDimension("y", ny)
        ods.createDimension("x", nx)
        vlat = ods.createVariable("lat", "f4", ("y",))
        vlon = ods.createVariable("lon", "f4", ("x",))
        vlat[:] = dlat.astype("float32")
        vlon[:] = dlon.astype("float32")
        v = ods.createVariable("landcover", "i2", ("y", "x"), zlib=True, complevel=4, fill_value=-32768)
        v[:] = out_grid
        v.long_name = "Land cover class (LCCS), resampled to DEM grid"
        v.grid_mapping = "crs"
        crs = ods.createVariable("crs", "i4")
        crs.spatial_ref = "EPSG:4326"
    print(f"[resample_landcover_to_dem_grid] 写出: {out_nc}, shape=({ny},{nx})")
    return int(ny), int(nx)


def _nearest_index(vec: np.ndarray, val: float) -> int:
    """在可能升序或降序的 1D 向量 vec 中找最邻近 val 的索引"""
    if vec.ndim != 1:
        raise ValueError("vec 必须为 1D")
    asc = vec[0] < vec[-1]
    if asc:
        idx = np.searchsorted(vec, val)
    else:
        # 对降序向量，映射到升序索引空间
        idx = len(vec) - np.searchsorted(vec[::-1], val)
    idx = np.clip(idx, 1, len(vec) - 1)
    left = idx - 1
    right = idx
    if abs(vec[right] - val) < abs(vec[left] - val):
        return int(right)
    return int(left)

def _load_nc_grid(nc_path: str, var_name: str):
    """读取 NetCDF 中的 1D lat/lon 与变量网格 (y,x)。"""
    with nc.Dataset(nc_path, "r") as ds:
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
        grid = ds.variables[var_name][:]
    return np.array(lats), np.array(lons), np.array(grid)

def _sample_nearest(lats: np.ndarray, lons: np.ndarray, grid: np.ndarray, lat0: float, lon0: float) -> float:
    """最近邻采样 (y,x)"""
    iy = _nearest_index(lats, lat0)
    ix = _nearest_index(lons, lon0)
    return float(grid[iy, ix])


def compare_dem_at_points(
    dem_nc1: str,
    dem_nc2: str,
    lats: np.ndarray,
    lons: np.ndarray,
    var1: str = "dem",
    var2: str = "dem",
    label1: str = "dem1",
    label2: str = "dem2",
) -> pd.DataFrame:
    """
    在给定经纬度点位处比较两种来源 DEM 数据的差异（最近邻采样）。

    参数：
      - dem_nc1: 第一种 DEM NetCDF 路径（如 SRTM 镶嵌结果）
      - dem_nc2: 第二种 DEM NetCDF 路径（如 GEBCO 重采样结果）
      - lats: 纬度数组，shape=(N,)
      - lons: 经度数组，shape=(N,)
      - var1, var2: 两文件中的高程变量名，默认均为 "dem"
      - label1, label2: 输出列名，默认 "dem1", "dem2"

    返回：
      - DataFrame 含列: lat, lon, {label1}, {label2}, diff (dem1-dem2), abs_diff, 及 valid (两者均有效为 True)
    """
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    if lats.shape[0] != lons.shape[0]:
        raise ValueError("lats 与 lons 长度必须一致")

    lat1, lon1, grid1 = _load_nc_grid(dem_nc1, var1)
    lat2, lon2, grid2 = _load_nc_grid(dem_nc2, var2)

    v1 = np.array([_sample_nearest(lat1, lon1, grid1, la, lo) for la, lo in zip(lats, lons)])
    v2 = np.array([_sample_nearest(lat2, lon2, grid2, la, lo) for la, lo in zip(lats, lons)])

    valid = np.isfinite(v1) & np.isfinite(v2)
    diff = np.where(valid, v1.astype(np.float64) - v2.astype(np.float64), np.nan)
    abs_diff = np.where(valid, np.abs(diff), np.nan)

    df = pd.DataFrame({
        "lat": lats,
        "lon": lons,
        label1: v1,
        label2: v2,
        "diff": diff,
        "abs_diff": abs_diff,
        "valid": valid,
    })
    return df


def build_station_attributes(
    csv_glob: str,
    id_col: str,
    lat_col: str,
    lon_col: str,
    date_col: str,
    precip_col: str,
    dem_nc: str,
    lc_nc: str,
    out_parquet: str,
    years: list = [2014, 2015, 2016, 2017, 2018, 2019],
    encoding: str = "utf-8",
) -> str:
    """
    读取 2014-2019 年各 CSV 日降水，生成站点属性表：
      - 唯一站点（按 station_id 排序后编号 sid=1..N）
      - 最近格点 DEM 与 landcover 类型
      - 各年份日均降水列：avg_prect{year}
    保存为 parquet，并返回保存路径。
    """
    files = sorted(glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"未匹配到 CSV：{csv_glob}")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp, encoding=encoding, dtype={id_col: str})
        # 解析日期，并限制到目标年
        dt = pd.to_datetime(df[date_col])
        df["_date"] = dt
        df["_year"] = dt.dt.year
        df = df[df["_year"].isin(years)].copy()
        dfs.append(df[[id_col, lat_col, lon_col, precip_col, "_date", "_year"]])
    data = pd.concat(dfs, ignore_index=True)
    # 唯一站点列表（按 station_id 排序取第一条坐标）
    stations = (
        data.sort_values(by=[id_col], kind="stable")
        .drop_duplicates(subset=[id_col])
        .reset_index(drop=True)
        [[id_col, lat_col, lon_col]]
        .copy()
    )
    stations["sid"] = np.arange(1, len(stations) + 1, dtype=int)
    # DEM/LC 最近邻采样
    dem_lats, dem_lons, dem_grid = _load_nc_grid(dem_nc, "dem")
    # 可选：坡度与坡向（若已写入 DEM nc）
    slope_grid = None
    aspect_grid = None
    try:
        _, _, slope_grid = _load_nc_grid(dem_nc, "slope")
    except Exception:
        pass
    try:
        _, _, aspect_grid = _load_nc_grid(dem_nc, "aspect")
    except Exception:
        pass
    lc_lats, lc_lons, lc_grid = _load_nc_grid(lc_nc, "landcover")
    dem_vals = []
    lc_vals = []
    slope_vals = []
    aspect_vals = []
    for _, r in stations.iterrows():
        dem_vals.append(_sample_nearest(dem_lats, dem_lons, dem_grid, float(r[lat_col]), float(r[lon_col])))
        lc_vals.append(int(_sample_nearest(lc_lats, lc_lons, lc_grid, float(r[lat_col]), float(r[lon_col]))))
        if slope_grid is not None:
            slope_vals.append(_sample_nearest(dem_lats, dem_lons, slope_grid, float(r[lat_col]), float(r[lon_col])))
        if aspect_grid is not None:
            aspect_vals.append(_sample_nearest(dem_lats, dem_lons, aspect_grid, float(r[lat_col]), float(r[lon_col])))
    stations["dem"] = dem_vals
    stations["lc_type"] = lc_vals
    if slope_grid is not None:
        stations["slope"] = slope_vals
    if aspect_grid is not None:
        stations["aspect"] = aspect_vals
    # 年均降水（先合并 sid）
    data = data.merge(stations[[id_col, "sid"]], on=id_col, how="left")
    # 缺失降水置 NaN 后求均值
    data["_prec"] = pd.to_numeric(data[precip_col], errors="coerce")
    annual = data.groupby(["sid", "_year"])["_prec"].mean().unstack("_year")
    # 重命名列
    annual = annual.rename(columns={y: f"avg_prect{y}" for y in annual.columns})
    # 合并回 stations
    meta = stations.merge(annual, on="sid", how="left")
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    meta.to_parquet(out_parquet, index=False, engine="fastparquet")
    return out_parquet

def add_hrlt_pcp_avg(
    station_parquet_in: str,
    station_parquet_out: str = None,
    grid_pattern: str = "/nas/zsai/prect_1km/ZJ_1km_prep_{year}.nc",
    years=None,
    var_name: str = "prep",
    lat_name: str = "lat",
    lon_name: str = "lon",
    col_name: str = "hrlt_pcp_avg",
) -> str:
    """
    读取多年（默认 2014–2017）浙江 1km 日降水 NetCDF 文件，按最近邻为每个站点计算多年日降水的平均值，
    将结果追加为一列 'hrlt_pcp_avg' 到现有的站点属性 parquet。

    参数：
      - station_parquet_in: 输入站点属性 parquet，需包含 'lat' 与 'lon' 或 'lng'
      - station_parquet_out: 输出路径；若为 None 则覆盖输入文件
      - grid模式与 years: 指定逐年 NetCDF 的路径与年份列表
    返回：写出文件路径
    """
    if years is None:
        years = [2014, 2015, 2016, 2017]

    df = pd.read_parquet(station_parquet_in)
    if "lat" not in df.columns:
        raise ValueError("station parquet 缺少 'lat' 列")
    if "lon" in df.columns:
        lon_col = "lon"
    elif "lng" in df.columns:
        lon_col = "lng"
    else:
        raise ValueError("station parquet 需包含 'lon' 或 'lng' 列")

    st_lat = df["l at"].to_numpy() if False else df["lat"].to_numpy()  # placeholder for patch context
    # 修正：使用正确列
    st_lat = df["lat"].to_numpy()
    st_lon = df[lon_col].to_numpy()
    n_st = len(df)

    # 多年平均的分子/分母
    sum_vals = np.zeros(n_st, dtype=np.float64)
    cnt_vals = np.zeros(n_st, dtype=np.int32)

    for y in years:
        path = grid_pattern.format(year=y)
        if not os.path.exists(path):
            print(f"[add_hrlt_pcp_avg] 跳过不存在文件: {path}")
            continue
        try:
            with nc.Dataset(path, "r") as ds:
                if lat_name not in ds.variables or lon_name not in ds.variables or var_name not in ds.variables:
                    print(f"[add_hrlt_pcp_avg] 文件缺少变量 {lat_name}/{lon_name}/{var_name}: {path}")
                    continue
                glat = np.array(ds.variables[lat_name][:])
                glon = np.array(ds.variables[lon_name][:])
                var = ds.variables[var_name]  # shape (time, y, x)
                nt = var.shape[0]
                ny, nx = var.shape[1], var.shape[2]

                # 计算全年的逐日平均栅格（对内存友好：按时间分块累积）
                sum_grid = np.zeros((ny, nx), dtype=np.float64)
                cnt_grid = np.zeros((ny, nx), dtype=np.int32)
                chunk = 60
                fill = getattr(var, "_FillValue", None)
                scale = getattr(var, "scale_factor", 1.0)
                offset = getattr(var, "add_offset", 0.0)
                for s in range(0, nt, chunk):
                    e = min(nt, s + chunk)
                    arr = np.array(var[s:e, :, :], dtype=np.float32)  # (k, ny, nx)
                    if fill is not None:
                        mask = (arr == fill)
                        arr = np.where(mask, np.nan, arr)
                    if scale != 1.0 or offset != 0.0:
                        arr = arr * float(scale) + float(offset)
                    # 去除 inf/-inf，避免后续平均出现 inf
                    arr[~np.isfinite(arr)] = np.nan
                    valid = ~np.isnan(arr)
                    sum_batch = np.nansum(arr, axis=0)  # (ny, nx)
                    cnt_batch = np.sum(valid, axis=0).astype(np.int32)
                    sum_grid += sum_batch
                    cnt_grid += cnt_batch
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan).astype(np.float32)

                # 为该年的平均场为每个站点取最近邻
                vals = np.empty(n_st, dtype=np.float32)
                for i in range(n_st):
                    iy = _nearest_index(glat, st_lat[i])
                    ix = _nearest_index(glon, st_lon[i])
                    v = mean_grid[iy, ix]
                    if not np.isfinite(v):
                        v = np.nan
                    vals[i] = v
                valid_mask = ~np.isnan(vals)
                sum_vals[valid_mask] += vals[valid_mask]
                cnt_vals[valid_mask] += 1
        except Exception as e:
            print(f"[add_hrlt_pcp_avg] 处理 {path} 出错: {e}")
            continue

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(cnt_vals > 0, sum_vals / cnt_vals, np.nan).astype(np.float32)
    avg[~np.isfinite(avg)] = np.nan
    df[col_name] = avg

    out_path = station_parquet_out or station_parquet_in
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, engine="fastparquet")
    return out_path

def print_station_meta_stats(parquet_path: str) -> None:
    """
    读取站点属性文件（parquet），逐列打印属性的最小值与最大值（数值列）。
    对非数值列，打印数据类型与去重后的唯一值个数。
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"未找到文件：{parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"File: {parquet_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    if 'is_train' in df.columns:
        print(f"- is_train: dtype={df['is_train'].dtype}, unique={df['is_train'].nunique(dropna=True)}, num_train={df['is_train'].sum()}, num_test={len(df)-df['is_train'].sum()}")
    print(df.columns)
    exit()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            vmin = pd.to_numeric(s, errors="coerce").min()
            vmax = pd.to_numeric(s, errors="coerce").max()
            print(f"- {col}: min={vmin}, max={vmax}")
        else:
            nunique = s.nunique(dropna=True)
            print(f"- {col}: dtype={s.dtype}, unique={nunique}")

def split_train_test_by_dem_lc(
    parquet_in: str,
    parquet_out: str = None,
    test_frac: float = 0.2,
    dem_bins: int | list | None = 10,
    random_state: int = 42,
) -> str:
    """
    在 station_meta.parquet 上增加 is_train 列，依据 lc_type 与 dem 分层抽样，保证 test 覆盖地形与地表类型。
    - dem 分层：默认按分位数分成 10 桶（或传入明确的分箱边界列表）
    - 每个 (lc_type, dem_bin) 组按 test_frac 抽样为 test，至少 1 个（当组内样本数>0）
    - 结果写回 parquet_out（默认覆盖输入文件）
    """
    if not os.path.exists(parquet_in):
        raise FileNotFoundError(parquet_in)
    df = pd.read_parquet(parquet_in)
    if "lc_type" not in df.columns or "dem" not in df.columns:
        raise ValueError("需要列：lc_type 与 dem 才能进行分层抽样")

    # dem 分箱
    dem_series = pd.to_numeric(df["dem"], errors="coerce")
    if isinstance(dem_bins, list):
        dem_cat = pd.cut(dem_series, bins=dem_bins, include_lowest=True)
    else:
        # 使用 qcut 做近似均衡的分位数分箱；当重复值较多时，qcut 可能失败，退化为 cut
        try:
            dem_cat = pd.qcut(dem_series, q=int(dem_bins), duplicates="drop")
        except Exception:
            dem_cat = pd.cut(dem_series, bins=int(dem_bins), include_lowest=True)
    df["_dem_bin"] = dem_cat.astype(str)

    rng = np.random.default_rng(random_state)
    df["is_train"] = True
    # 分层抽样
    for (lc, db), grp in df.groupby(["lc_type", "_dem_bin"], dropna=False):
        idx = grp.index
        n = len(idx)
        if n == 0:
            continue
        n_test = max(1, int(round(n * test_frac)))
        select = rng.choice(idx, size=min(n_test, n), replace=False)
        df.loc[select, "is_train"] = False

    df = df.drop(columns=["_dem_bin"])
    out_path = parquet_out or parquet_in
    df.to_parquet(out_path, index=False, engine="fastparquet")
    return out_path

def _nc_time_to_datetime(time_var) -> np.ndarray:
    """将 nc 中的 time 变量转换为 UTC 时间戳（numpy datetime64[ns]）。"""
    from netCDF4 import num2date
    dt = num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    # 转为 numpy datetime64[ns]
    return np.array(pd.to_datetime(dt))

def add_station_grid_indices(
    station_parquet_in: str,
    static_nc: str = "/home/zsai/python_projects/prect500-data/static/landcover_on_dem300m.nc",
    pcp_nc: str = "/nas/zsai/prect500/cmorph_pcp/merged-30min_8km_daily/CMORPH_V1.0_ADJ_8km-daily_ZJ_merged_2020.nc",
    out_parquet: str = None,
    lat_col: str = "lat",
    lon_prefer: str = "lon",
) -> str:
    """
    为站点静态表增加两个网格数据源的最近邻索引：
      - 地表数据（DEM/Landcover 共网格）索引：demlc_row, demlc_col
      - 降水数据（CMORPH 8km daily）索引：pcp_row, pcp_col
    若 out_parquet 为空，则覆盖写回原文件。
    """
    if not os.path.exists(station_parquet_in):
        raise FileNotFoundError(station_parquet_in)
    df = pd.read_parquet(station_parquet_in)
    # 若之前已经生成过索引列，先删除旧列，避免沿用错误索引
    cols_to_drop = ["demlc_row", "demlc_col", "pcp_row", "pcp_col"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    if lat_col not in df.columns:
        raise ValueError(f"缺少纬度列: {lat_col}")
    # 经度列自动适配 lon/lng
    if lon_prefer in df.columns:
        lon_col = lon_prefer
    elif "lng" in df.columns:
        lon_col = "lng"
    elif "lon" in df.columns:
        lon_col = "lon"
    else:
        raise ValueError("缺少经度列：期望 'lon' 或 'lng'")
    st_lat = df[lat_col].to_numpy(dtype=float)
    st_lon = df[lon_col].to_numpy(dtype=float)
    # 读取静态网格 lat/lon（1D）
    with nc.Dataset(static_nc, "r") as sds:
        s_lat_name, s_lon_name = _coord_names(sds)
        if s_lat_name is None or s_lon_name is None:
            raise KeyError("静态网格缺少 lat/lon 坐标")
        s_lat = np.array(sds.variables[s_lat_name][:])
        s_lon = np.array(sds.variables[s_lon_name][:])
        if s_lat.ndim != 1 or s_lon.ndim != 1:
            raise ValueError("静态网格需为 1D lat/lon")
    # 读取降水网格 lat/lon（1D）
    with nc.Dataset(pcp_nc, "r") as pds:
        p_lat_name, p_lon_name = _coord_names(pds)
        if p_lat_name is None or p_lon_name is None:
            raise KeyError("降水网格缺少 lat/lon 坐标")
        p_lat = np.array(pds.variables[p_lat_name][:])
        p_lon = np.array(pds.variables[p_lon_name][:])
        if p_lat.ndim != 1 or p_lon.ndim != 1:
            raise ValueError("降水网格需为 1D lat/lon")
    # 最近邻索引（逐站）
    demlc_iy = np.array([_nearest_index(s_lat, v) for v in st_lat], dtype=np.int32)
    demlc_ix = np.array([_nearest_index(s_lon, v) for v in st_lon], dtype=np.int32)
    pcp_iy = np.array([_nearest_index(p_lat, v) for v in st_lat], dtype=np.int32)
    pcp_ix = np.array([_nearest_index(p_lon, v) for v in st_lon], dtype=np.int32)
    # 写入列
    df["demlc_row"] = demlc_iy
    df["demlc_col"] = demlc_ix
    df["pcp_row"] = pcp_iy
    df["pcp_col"] = pcp_ix
    # 保存
    out_path = out_parquet or station_parquet_in
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 使用 fastparquet 写入，避免 pyarrow 的本地文件系统注册冲突
    df.to_parquet(out_path, index=False, engine="fastparquet")
    print(df.head())
    print(f"[add_station_grid_indices] 已写出: {out_path}")
    return out_path

def _stations_nearest_indices(lat_grid: np.ndarray, lon_grid: np.ndarray, lat_s: np.ndarray, lon_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """为每个站点找到最近格点索引（返回 iy, ix，均为 shape=(N,) 的 int 数组）。"""
    iy = np.array([_nearest_index(lat_grid, v) for v in lat_s], dtype=int)
    ix = np.array([_nearest_index(lon_grid, v) for v in lon_s], dtype=int)
    return iy, ix

def _time_harmonics(dates: pd.Series) -> pd.DataFrame:
    """按日构建 day-of-year 时间谐波特征（sin, cos），返回与 dates 对齐的 DataFrame。"""
    doy = dates.dt.dayofyear.to_numpy()
    sin_doy = np.sin(2 * np.pi * (doy / 365.25))
    cos_doy = np.cos(2 * np.pi * (doy / 365.25))
    return pd.DataFrame({"sin_doy": sin_doy.astype(np.float32), "cos_doy": cos_doy.astype(np.float32)})

def build_daily_dynamic_features(
    year: int,
    stations_parquet: str,
    precip_nc_pattern: str = "/nas/zsai/prect_1km/ZJ_1km_prep_{year}.nc",
    wind_nc_pattern: str = "/home/zsai/python_projects/prect500-data/era5_100m_uv_0p1/{year}_uv100_hourly_117.5-123.5_26.5-31.5.nc",
    out_parquet: str = "/home/zsai/python_projects/prect500-data/daily_features/train_daily_features_{year}.parquet",
    lat_col: str = "lat",
    lon_col: str = "lng",
    ) -> str:
    """
    生成“站点×日期”的动态特征（从每年 1 月 3 日开始）：
      1) 基础静态：经纬度、dem、slope、aspect、lc_type（来源 stations_parquet）
      2) 时间谐波：sin_doy, cos_doy
      3) 日降水（最近格点）：hrlt_t, hrlt_t_1, hrlt_t_2（来自 ZJ_1km_prep_{year}.nc）
      4) 背景风（最近格点）：u100_mean, v100_mean（ERA5 100m，前一日12:00~当日11:00 平均，来自 {year}_uv100_hourly_*.nc）
    返回写入的 parquet 路径。
    """
    os.makedirs(os.path.dirname(out_parquet.format(year=year)), exist_ok=True)
    # 读取训练站点表
    st = pd.read_parquet(stations_parquet)
    #if "is_train" in st.columns:
        #st = st[st["is_train"] == True].copy()  # noqa
    st = st.reset_index(drop=True)
    for c in [lat_col, lon_col, "dem", "lc_type"]:
        if c not in st.columns:
            raise ValueError(f"缺少列: {c}")
    # 准备日期范围（从 1 月 3 日开始）#5月份到9月份
    d0 = pd.Timestamp(f"{year}-05-01")
    d1 = pd.Timestamp(f"{year}-09-30")
    days = pd.date_range(d0, d1, freq="D")
    num_days = len(days)
    # ---------- 降水：读取当年子区域 ----------
    p_nc = precip_nc_pattern.format(year=year)
    with nc.Dataset(p_nc, "r") as ds:
        plat = np.array(ds.variables["lat"][:])
        plon = np.array(ds.variables["lon"][:])
        ptime = _nc_time_to_datetime(ds.variables["time"])
        # 对齐到当年日期（确保有 t, t-1, t-2）
        p_df = pd.DataFrame({"time": pd.to_datetime(ptime)})
        p_df["date"] = p_df["time"].dt.normalize()
        # 构建日期 -> 索引映射
        date2idx = {d: i for i, d in enumerate(p_df["date"])}
        # 预计算最近格点索引
        iy_p, ix_p = _stations_nearest_indices(plat, plon, st[lat_col].to_numpy(), st[lon_col].to_numpy())
        prep_var = ds.variables["prep"]  # (T, Y, X)
        # 提取当年所有日期的 t/t-1/t-2
        hrlt_t = np.full((num_days, len(st)), np.nan, dtype=np.float32)
        hrlt_t_1 = np.full_like(hrlt_t, np.nan)
        hrlt_t_2 = np.full_like(hrlt_t, np.nan)
        for di, d in enumerate(days):
            di_t = date2idx.get(d.normalize(), None)
            di_t1 = date2idx.get((d - pd.Timedelta(days=1)).normalize(), None)
            di_t2 = date2idx.get((d - pd.Timedelta(days=2)).normalize(), None)
            if di_t is None or di_t1 is None or di_t2 is None:
                continue
            # 最近邻采样：先取二维场，再用成对索引选取每个站点的位置，避免 (N,N) 广播
            p_t  = prep_var[di_t,  :, :][iy_p, ix_p].astype(np.float32)
            p_t1 = prep_var[di_t1, :, :][iy_p, ix_p].astype(np.float32)
            p_t2 = prep_var[di_t2, :, :][iy_p, ix_p].astype(np.float32)
            hrlt_t[di, :]   = p_t
            hrlt_t_1[di, :] = p_t1
            hrlt_t_2[di, :] = p_t2
    # ---------- 风：ERA5 100m（前一日12:00 ~ 当日11:00 的逐小时 -> 当日平均） ----------
    w_nc = wind_nc_pattern.format(year=year)
    with nc.Dataset(w_nc, "r") as ds:
        wlat = np.array(ds.variables["latitude" if "latitude" in ds.variables else "lat"][:])
        wlon = np.array(ds.variables["longitude" if "longitude" in ds.variables else "lon"][:])
        wtime = _nc_time_to_datetime(ds.variables["valid_time"])
        # 变量名兼容（u100, v100 或 u, v）
        u_name = "u100" if "u100" in ds.variables else ("u" if "u" in ds.variables else None)
        v_name = "v100" if "v100" in ds.variables else ("v" if "v" in ds.variables else None)
        if u_name is None or v_name is None:
            raise KeyError("风文件缺少 u/v 变量（期望 u100/v100 或 u/v）")
        u = ds.variables[u_name][:]  # (T, Y, X)
        v = ds.variables[v_name][:]  # (T, Y, X)
        # 最近邻格点索引（风场网格）
        iy_w, ix_w = _stations_nearest_indices(
            wlat if wlat.ndim == 1 else wlat[:, 0],
            wlon if wlon.ndim == 1 else wlon[0, :],
            st[lat_col].to_numpy(),
            st[lon_col].to_numpy(),
        )
    # 将逐小时风场聚合为“前一日12:00~当日11:00”的日平均 u、v
    wtime_ts = pd.to_datetime(wtime)
    n_st = len(st)
    wind_u_mean = np.full((num_days, n_st), np.nan, dtype=np.float32)
    wind_v_mean = np.full((num_days, n_st), np.nan, dtype=np.float32)
    for di, d in enumerate(days):
        # 窗口：前一日 12:00 (含) -> 当日 12:00 (不含)，对应前一日12:00~当日11:00 共24小时
        start = (d - pd.Timedelta(days=1)) + pd.Timedelta(hours=12)
        end = d + pd.Timedelta(hours=12)
        mask = (wtime_ts >= start) & (wtime_ts < end)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        sum_u = np.zeros(n_st, dtype=np.float64)
        sum_v = np.zeros(n_st, dtype=np.float64)
        cnt = np.zeros(n_st, dtype=np.int32)
        for ti in idxs:
            # 先取该小时的二维场，再做最近邻采样，避免 (N,N) 广播
            u2 = u[ti, :, :][iy_w, ix_w].astype(np.float32)
            v2 = v[ti, :, :][iy_w, ix_w].astype(np.float32)
            valid = np.isfinite(u2) & np.isfinite(v2)
            if not np.any(valid):
                continue
            sum_u[valid] += u2[valid]
            sum_v[valid] += v2[valid]
            cnt[valid] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_u = np.where(cnt > 0, sum_u / cnt, np.nan).astype(np.float32)
            mean_v = np.where(cnt > 0, sum_v / cnt, np.nan).astype(np.float32)
        wind_u_mean[di, :] = mean_u
        wind_v_mean[di, :] = mean_v
    # ---------- 组装输出表 ----------
    # 静态基础列
    base = st[[lat_col, lon_col, "dem"]].copy()
    if "slope" in st.columns:
        base["slope"] = st["slope"]
    if "aspect" in st.columns:
        base["aspect"] = st["aspect"]
    base["lc_type"] = st["lc_type"]
    # 重复到“日期×站点”
    base_rep = pd.concat([base] * num_days, ignore_index=True)
    # 日期列与时间谐波
    date_col_series = pd.Series(np.repeat(days.values, len(st)), name="date")
    th = _time_harmonics(pd.to_datetime(date_col_series))
    # 动态降水/风（按相同顺序展开）
    def _flatten(mat: np.ndarray) -> np.ndarray:
        # mat shape: (D, N) -> (D*N,)
        return mat.reshape(-1, order="C")
    df_out = pd.DataFrame({
        "date": date_col_series.values,
        "lat": base_rep[lat_col].to_numpy(),
        "lng": base_rep[lon_col].to_numpy(),
        "dem": base_rep["dem"].to_numpy(),
        "lc_type": base_rep["lc_type"].to_numpy(),
    })
    if "slope" in base_rep.columns:
        df_out["slope"] = base_rep["slope"].to_numpy()
    if "aspect" in base_rep.columns:
        df_out["aspect"] = base_rep["aspect"].to_numpy()
    df_out["sin_doy"] = th["sin_doy"].to_numpy()
    df_out["cos_doy"] = th["cos_doy"].to_numpy()
    df_out["hrlt_t"] = _flatten(hrlt_t)
    df_out["hrlt_t_1"] = _flatten(hrlt_t_1)
    df_out["hrlt_t_2"] = _flatten(hrlt_t_2)
    # ERA5 100m 日平均风速分量（前一日12:00~当日11:00）
    df_out["u100_mean"] = _flatten(wind_u_mean)
    df_out["v100_mean"] = _flatten(wind_v_mean)
    # 追加站点 sid 以便合并
    df_out["sid"] = np.tile(st["sid"].to_numpy(), num_days)
    out_path = out_parquet.format(year=year)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_parquet(out_path, index=False, engine="fastparquet")
    return out_path


def merge_station_observations(
    obs_folder: str,
    output_csv: str,
    years: list = [2014, 2015, 2016, 2017, 2018, 2019]
):
    """
    合并所有站点观测数据到一个CSV文件。
    
    参数:
        obs_folder: 存放站点观测CSV文件的文件夹路径
        output_csv: 输出合并后的CSV文件路径
        years: 要处理的年份列表，默认2014-2019
    
    处理步骤:
        1. 读取指定年份范围内的所有CSV文件
        2. 合并所有数据
        3. 根据stationId和datetime去重
        4. 删除r2020为空的行
        5. 保存到输出文件
    """
    # 查找所有匹配的CSV文件
    csv_files = []
    for year in years:
        # 匹配文件名中包含年份的文件，如 "浙江自动站日数据2014*.csv"
        pattern = os.path.join(obs_folder, f"浙江自动站日数据{year}*.csv")
        csv_files.extend(glob(pattern))
    
    if not csv_files:
        raise ValueError(f"在 {obs_folder} 中未找到匹配的CSV文件")
    
    print(f"找到 {len(csv_files)} 个CSV文件，开始读取...")
    
    # 读取所有CSV文件并合并
    dfs = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            print(f"  读取: {os.path.basename(csv_file)}, 行数: {len(df)}")
            dfs.append(df)
        except Exception as e:
            print(f"  警告: 读取 {csv_file} 失败: {e}")
            continue
    
    if not dfs:
        raise ValueError("没有成功读取任何CSV文件")
    
    # 合并所有数据
    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(df_merged)}")
    
    # 确保datetime列是datetime类型
    if 'datetime' in df_merged.columns:
        df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])
    
    # 根据stationId和datetime去重（保留第一条）
    print("根据stationId和datetime去重...")
    before_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['stationId', 'datetime'], keep='first')
    after_dedup = len(df_merged)
    print(f"去重前: {before_dedup} 行, 去重后: {after_dedup} 行, 删除: {before_dedup - after_dedup} 行")
    
    # 删除r2020为空的行（包括NaN、None、空字符串）
    print("删除r2020为空的行...")
    before_dropna = len(df_merged)
    # 删除r2020为NaN、None的行
    df_merged = df_merged.dropna(subset=['r2020'])
    # 如果r2020是字符串类型，也删除空字符串
    if df_merged['r2020'].dtype == 'object':
        df_merged = df_merged[df_merged['r2020'].astype(str).str.strip() != '']
        # 也删除只包含空白字符的行
        df_merged = df_merged[df_merged['r2020'].astype(str).str.strip() != 'nan']
    after_dropna = len(df_merged)
    print(f"删除空值前: {before_dropna} 行, 删除后: {after_dropna} 行, 删除: {before_dropna - after_dropna} 行")
    
    # 保存到输出文件
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"合并后的数据已保存到: {output_csv}")
    print(f"最终数据行数: {len(df_merged)}")
    print(f"唯一站点数: {df_merged['stationId'].nunique()}")
    print(f"日期范围: {df_merged['datetime'].min()} 到 {df_merged['datetime'].max()}")
    
    return df_merged


def merge_dynamic_and_observations(
    dynamic_feature_paths: list[str],
    obs_csv: str,
    station_meta_parquet: str,
    out_parquet: str,
) -> str:
    """
    将动态特征与站点观测日降水进行合并，并生成观测/填充值。

    功能：
      1) 将多个动态特征 Parquet 文件（date×station 数据）合并。
      2) 通过 station_meta（含 sid 与 stationId 对应关系）补充 stationId。
      3) 与观测 CSV（包含 stationId, datetime, r2020）按 stationId+datetime 内连接，得到 P_obs_t。
      4) 构造前一天/前两天观测：P_filled_t_1, P_filled_t_2。如果缺观测则用 hrlt_t_1/hrlt_t_2 代理，并产出缺测指示列。

    参数：
      dynamic_feature_paths: 例如 [station_daily_features_2014.parquet, ...]。
      obs_csv: merge_station_observations 输出的 CSV。
      station_meta_parquet: 含 sid、stationId 的静态表。
      out_parquet: 输出文件路径。
    """
    if not dynamic_feature_paths:
        raise ValueError("dynamic_feature_paths 不能为空")
    dfs = []
    for path in dynamic_feature_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到动态特征文件: {path}")
        df_part = pd.read_parquet(path)
        if "date" in df_part.columns and "datetime" not in df_part.columns:
            df_part["datetime"] = pd.to_datetime(df_part["date"])
        elif "datetime" in df_part.columns:
            df_part["datetime"] = pd.to_datetime(df_part["datetime"])
        else:
            raise ValueError(f"{path} 缺少 date/datetime 列")
        dfs.append(df_part)
    dyn_df = pd.concat(dfs, ignore_index=True)
    # 确保 sid 存在或已有 stationId
    if "stationId" not in dyn_df.columns:
        if "sid" not in dyn_df.columns:
            raise ValueError("动态特征缺少 stationId / sid 列，无法对齐观测")
        meta = pd.read_parquet(station_meta_parquet)[["sid", "stationId"]]
        dyn_df = dyn_df.merge(meta, on="sid", how="left")
    dyn_df["stationId"] = dyn_df["stationId"].astype(str)
    dyn_df["datetime"] = pd.to_datetime(dyn_df["datetime"])

    # 读取观测
    obs_df = pd.read_csv(obs_csv)
    required_cols = {"stationId", "datetime", "r2020"}
    if not required_cols.issubset(obs_df.columns):
        missing = required_cols - set(obs_df.columns)
        raise ValueError(f"观测文件缺少列: {missing}")
    obs_df["stationId"] = obs_df["stationId"].astype(str)
    obs_df["datetime"] = pd.to_datetime(obs_df["datetime"])
    obs_df = obs_df.rename(columns={"r2020": "P_obs_t"})

    # 合并
    merged = dyn_df.merge(obs_df[["stationId", "datetime", "P_obs_t"]], on=["stationId", "datetime"], how="inner")
    merged = merged.sort_values(["stationId", "datetime"]).reset_index(drop=True)

    def _lag_obs(series: pd.Series, dates: pd.Series, lag: int) -> pd.Series:
        prev_vals = series.groupby(merged["stationId"]).shift(lag)
        prev_dates = dates.groupby(merged["stationId"]).shift(lag)
        valid = prev_dates.notna() & ((dates - prev_dates).dt.days == lag)
        return prev_vals.where(valid)

    merged["P_obs_prev_1"] = _lag_obs(merged["P_obs_t"], merged["datetime"], 1)
    merged["P_obs_prev_2"] = _lag_obs(merged["P_obs_t"], merged["datetime"], 2)

    merged["P_missing_t_1"] = merged["P_obs_prev_1"].isna().astype(np.int8)
    merged["P_missing_t_2"] = merged["P_obs_prev_2"].isna().astype(np.int8)

    if "hrlt_t_1" not in merged.columns or "hrlt_t_2" not in merged.columns:
        raise ValueError("动态特征缺少 hrlt_t_1 / hrlt_t_2 列，无法进行代理填充")

    merged["P_filled_t_1"] = merged["P_obs_prev_1"].fillna(merged["hrlt_t_1"])
    merged["P_filled_t_2"] = merged["P_obs_prev_2"].fillna(merged["hrlt_t_2"])

    # 清理辅助列
    merged.drop(columns=["P_obs_prev_1", "P_obs_prev_2"], inplace=True)

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    merged.to_parquet(out_parquet, index=False, engine="fastparquet")
    print(merged.head(), merged.tail())
    for col in merged.columns:
        print(col, merged[col].dtype)
    return out_parquet



def merge_monthly_station_obs_csvs(
    input_dir: str,
    years: list = None,
    out_csv: str = None,
    month_glob_fmt: str = "{year}-*.csv",
    drop_empty_r2020: bool = True,
):
    """
    合并按月切分的站点观测 CSV（如 2020-01.csv ... 2024-12.csv）为一个 2014-2024 的总表。
    
    参数：
      - input_dir: 月度 CSV 所在目录
      - years: 需要合并的年份列表，默认 [2014..2024]
      - out_csv: 输出路径，默认写入 input_dir/station_obs_merged_2014-2024.csv
      - month_glob_fmt: 月度文件匹配模板（含 {year}），默认 "{year}-*.csv"
      - drop_empty_r2020: 是否删除 r2020 为空的行，默认 True
    """
    if years is None:
        years = list(range(2014, 2024 + 1))
    paths = []
    for y in years:
        g = os.path.join(input_dir, month_glob_fmt.format(year=y))
        found = sorted(glob(g))
        if not found:
            print(f"[merge_monthly_station_obs_csvs] 年 {y} 未找到文件: {g}")
        paths.extend(found)
    paths.append("/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2019.csv")
    
    if not paths:
        raise FileNotFoundError("未找到任何月度 CSV 文件")
    parts = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # 统一类型
            if "stationId" in df.columns:
                df["stationId"] = df["stationId"].astype(str)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            # 去除 r2020 为空的记录（可配置）
            if drop_empty_r2020 and "r2020" in df.columns:
                df = df.dropna(subset=["r2020"])
                if df["r2020"].dtype == "object":
                    df = df[df["r2020"].astype(str).str.strip() != ""]
                    df = df[df["r2020"].astype(str).str.strip().str.lower() != "nan"]
            parts.append(df)
            print(f"[merge_monthly_station_obs_csvs] 读取: {os.path.basename(p)}, shape={df.shape}")
        except Exception as e:
            print(f"[merge_monthly_station_obs_csvs] 警告：读取失败 {p}: {e}")
    if not parts:
        raise ValueError("没有可用的数据帧可合并")
    merged = pd.concat(parts, ignore_index=True, sort=False)
    # 去重与排序
    subset = [c for c in ["stationId", "datetime"] if c in merged.columns]
    if subset:
        before = len(merged)
        merged = merged.drop_duplicates(subset=subset, keep="first")
        after = len(merged)
        print(f"[merge_monthly_station_obs_csvs] 去重: {before} -> {after}")
        merged = merged.sort_values(by=subset).reset_index(drop=True)
    # 输出
    if out_csv is None:
        out_csv = os.path.join(input_dir, "station_obs_merged_2014-2024.csv")
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[merge_monthly_station_obs_csvs] 已写出: {out_csv}, shape={merged.shape}")
    return out_csv


def analyze_precipitation_distribution(
    parquet_path: str,
    hrlt_col: str = "hrlt_t",
    pobs_col: str = "P_obs_t",
    thresholds: list = [0, 0.1, 1, 2, 5, 10, 25, 50, 75, 100],
    out_csv: str = None,
    scatter_hrlt_threshold: float = None,
    scatter_hrlt_gt: bool = True,  # True: > threshold, False: < threshold
    scatter_n_samples: int = 100000,
    scatter_out_png: str = None,
    random_seed: int = 42,
) -> str:
    """
    分析降水分布：统计 hrlt_t 和 P_obs_t 在不同阈值区间的数量及占比，并绘制散点图。
    
    参数：
      - parquet_path: 输入 parquet 文件路径
      - hrlt_col: hrlt_t 列名，默认 "hrlt_t"
      - pobs_col: P_obs_t 列名，默认 "P_obs_t"
      - thresholds: 降水阈值列表（mm），默认 [0, 0.1, 1, 2, 5, 10, 25, 50, 75, 100]
      - out_csv: 统计结果 CSV 输出路径；None 则自动生成（同目录下）
      - scatter_hrlt_threshold: 散点图筛选阈值（mm）；None 则不筛选
      - scatter_hrlt_gt: True 表示筛选 hrlt_t > threshold，False 表示 < threshold
      - scatter_n_samples: 散点图随机采样数量，默认 100000
      - scatter_out_png: 散点图输出路径；None 则自动生成
      - random_seed: 随机种子，默认 42
    
    返回：
      - out_csv 路径字符串
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"未找到文件: {parquet_path}")
    
    print(f"[analyze_precipitation_distribution] 读取: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if hrlt_col not in df.columns:
        raise ValueError(f"缺少列: {hrlt_col}")
    if pobs_col not in df.columns:
        raise ValueError(f"缺少列: {pobs_col}")
    
    # 提取有效数据（去除 NaN）
    hrlt = pd.to_numeric(df[hrlt_col], errors="coerce")
    pobs = pd.to_numeric(df[pobs_col], errors="coerce")
    valid_hrlt = hrlt.notna()
    valid_pobs = pobs.notna()
    
    print(f"[analyze_precipitation_distribution] 总行数: {len(df)}")
    print(f"[analyze_precipitation_distribution] hrlt_t 有效: {valid_hrlt.sum()}, P_obs_t 有效: {valid_pobs.sum()}")
    diff_hrlt_pobs =  pobs-hrlt
    print(f"[analyze_precipitation_distribution] hrlt_t - P_obs_t 差异: {diff_hrlt_pobs.sum()}")
    print(np.nanpercentile(diff_hrlt_pobs.to_numpy(dtype=np.float64), [0, 10, 25, 50, 75, 90, 100]))
    print(diff_hrlt_pobs.describe())
    abs_diff_hrlt_pobs = np.abs(diff_hrlt_pobs)
    print(np.nanpercentile(abs_diff_hrlt_pobs.to_numpy(dtype=np.float64), [0, 10, 25, 50, 75, 90, 100]))
    print(abs_diff_hrlt_pobs.describe())
    import matplotlib.pyplot as plt
    plt.hist(diff_hrlt_pobs.dropna().to_numpy(dtype=np.float64), bins=100)
    plt.savefig("diff_hrlt_pobs.png")
    plt.close()

    # 注意：math.log1p 只支持标量，pobs/hrlt 是 Series；这里必须用 np.log1p 做向量化。
    # 同时防御性地 clip 到 >=0，避免极少量负值导致 log1p 产生 NaN。
    pobs_c = pobs.clip(lower=0.0)
    hrlt_c = hrlt.clip(lower=0.0)
    q99=75
    diff = np.log1p(pobs_c)/np.log1p(q99) - np.log1p(hrlt_c)/np.log1p(q99)
    print(np.nanpercentile(diff.to_numpy(dtype=np.float64), [0, 10, 25, 50, 75, 90, 100]))
    print(diff.describe())
    plt.hist(diff.dropna().to_numpy(dtype=np.float64), bins=100)
    plt.savefig("diff.png")
    plt.close()
   
    exit()
    # 排序阈值
    thresholds = sorted([float(t) for t in thresholds])
    
    def _count_by_thresholds(values: pd.Series, valid_mask: pd.Series, name: str) -> pd.DataFrame:
        """按阈值区间统计数量和占比"""
        vals = values[valid_mask].to_numpy()
        total = len(vals)
        if total == 0:
            return pd.DataFrame()
        
        results = []
        for i in range(len(thresholds)):
            if i == 0:
                # 第一个区间: < thresholds[0]
                mask = vals < thresholds[0]
                label = f"<{thresholds[0]}"
            else:
                # 中间区间: [thresholds[i-1], thresholds[i])
                mask = (vals >= thresholds[i-1]) & (vals < thresholds[i])
                label = f"[{thresholds[i-1]}, {thresholds[i]})"
            
            count = int(mask.sum())
            pct = 100.0 * float(count) / float(total) if total > 0 else 0.0
            results.append({
                "variable": name,
                "threshold_range": label,
                "count": count,
                "percentage": pct,
            })
        
        # 最后一个区间: >= thresholds[-1]
        mask = vals >= thresholds[-1]
        count = int(mask.sum())
        pct = 100.0 * float(count) / float(total) if total > 0 else 0.0
        results.append({
            "variable": name,
            "threshold_range": f">={thresholds[-1]}",
            "count": count,
            "percentage": pct,
        })
        
        return pd.DataFrame(results)
    
    # 统计 hrlt_t
    stats_hrlt = _count_by_thresholds(hrlt, valid_hrlt, "hrlt_t")
    # 统计 P_obs_t
    stats_pobs = _count_by_thresholds(pobs, valid_pobs, "P_obs_t")
    
    # 合并结果
    stats_all = pd.concat([stats_hrlt, stats_pobs], ignore_index=True)
    
    # 输出 CSV
    if out_csv is None:
        base = os.path.splitext(parquet_path)[0]
        out_csv = f"{base}_precipitation_distribution_stats.csv"
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    stats_all.to_csv(out_csv, index=False)
    print(f"[analyze_precipitation_distribution] 统计结果已保存: {out_csv}")
    print("\n统计摘要:")
    print(stats_all.to_string(index=False))
    
    # 绘制散点图（如果指定了阈值）
    if scatter_hrlt_threshold is not None:
        rng = np.random.default_rng(random_seed)
        
        # 同时筛选 >threshold 和 <threshold 的数据
        mask_gt = (hrlt > float(scatter_hrlt_threshold)) & valid_hrlt & valid_pobs
        mask_lt = (hrlt < float(scatter_hrlt_threshold)) & valid_hrlt & valid_pobs
        
        df_gt = df[mask_gt].copy()
        df_lt = df[mask_lt].copy()
        n_gt = len(df_gt)
        n_lt = len(df_lt)
        
        print(f"\n[analyze_precipitation_distribution] 筛选条件: hrlt_t > {scatter_hrlt_threshold} 和 hrlt_t < {scatter_hrlt_threshold}")
        print(f"[analyze_precipitation_distribution] >{scatter_hrlt_threshold} 样本数: {n_gt}, <{scatter_hrlt_threshold} 样本数: {n_lt}")
        
        if n_gt == 0 and n_lt == 0:
            print("[analyze_precipitation_distribution] 警告: 筛选后无样本，跳过散点图绘制")
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 10), dpi=150)
            
            all_x = []
            all_y = []
            all_labels = []
            
            # 处理 >threshold 的数据
            if n_gt > 0:
                n_sample_gt = min(int(scatter_n_samples), n_gt)
                if n_sample_gt < n_gt:
                    sampled_idx_gt = rng.choice(n_gt, size=n_sample_gt, replace=False)
                    sampled_df_gt = df_gt.iloc[sampled_idx_gt].copy()
                else:
                    sampled_df_gt = df_gt.copy()
                
                hrlt_gt = pd.to_numeric(sampled_df_gt[hrlt_col], errors="coerce").to_numpy()
                pobs_gt = pd.to_numeric(sampled_df_gt[pobs_col], errors="coerce").to_numpy()
                valid_gt = np.isfinite(hrlt_gt) & np.isfinite(pobs_gt)
                hrlt_gt = hrlt_gt[valid_gt]
                pobs_gt = pobs_gt[valid_gt]
                
                if len(hrlt_gt) > 0:
                    all_x.append(pobs_gt)  # P_obs_t 作为横轴
                    all_y.append(hrlt_gt)  # hrlt_t 作为纵轴
                    all_labels.append(f"hrlt_t > {scatter_hrlt_threshold} (n={len(hrlt_gt)})")
            
            # 处理 <threshold 的数据
            if n_lt > 0:
                n_sample_lt = min(int(scatter_n_samples), n_lt)
                if n_sample_lt < n_lt:
                    sampled_idx_lt = rng.choice(n_lt, size=n_sample_lt, replace=False)
                    sampled_df_lt = df_lt.iloc[sampled_idx_lt].copy()
                else:
                    sampled_df_lt = df_lt.copy()
                
                hrlt_lt = pd.to_numeric(sampled_df_lt[hrlt_col], errors="coerce").to_numpy()
                pobs_lt = pd.to_numeric(sampled_df_lt[pobs_col], errors="coerce").to_numpy()
                valid_lt = np.isfinite(hrlt_lt) & np.isfinite(pobs_lt)
                hrlt_lt = hrlt_lt[valid_lt]
                pobs_lt = pobs_lt[valid_lt]
                
                if len(hrlt_lt) > 0:
                    all_x.append(pobs_lt)  # P_obs_t 作为横轴
                    all_y.append(hrlt_lt)  # hrlt_t 作为纵轴
                    all_labels.append(f"hrlt_t < {scatter_hrlt_threshold} (n={len(hrlt_lt)})")
            
            if len(all_x) == 0:
                print("[analyze_precipitation_distribution] 警告: 采样后无有效数据点，跳过散点图绘制")
            else:
                # 绘制散点图（三个子图：>10, <10(Zoomed), <10(Full X)）
                fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)
                
                # 计算全局范围
                all_x_combined = np.concatenate(all_x) if len(all_x) > 0 else np.array([])
                all_y_combined = np.concatenate(all_y) if len(all_y) > 0 else np.array([])
                if len(all_x_combined) > 0:
                    x_max = np.nanmax(all_x_combined)
                    x_min = np.nanmin(all_x_combined)
                    y_max = np.nanmax(all_y_combined)
                    y_min = np.nanmin(all_y_combined)
                else:
                    x_max, x_min, y_max, y_min = 100, 0, 100, 0
                
                # 1. 左图：hrlt_t > threshold (全范围)
                # 检查 all_labels[0] 是否包含 ">"
                has_gt = (len(all_x) > 0 and ">" in all_labels[0])
                
                if has_gt:
                    ax = axes[0]
                    x_data = all_x[0]
                    y_data = all_y[0]
                    label = all_labels[0]
                    
                    ax.scatter(x_data, y_data, alpha=0.3, s=1, c="red", edgecolors='none')
                    
                    # 1:1 线
                    plot_max = max(x_max, y_max)
                    plot_min = min(x_min, y_min)
                    ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", linewidth=2, label="1:1", alpha=0.7)
                    
                    ax.set_xlabel(f"{pobs_col} (mm)")
                    ax.set_ylabel(f"{hrlt_col} (mm)")
                    ax.set_title(f"{label}\n(Full Range)", fontsize=12, fontweight='bold')
                    ax.legend(loc='upper left')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([x_min, x_max])
                    ax.set_ylim([y_min, y_max])
                else:
                    axes[0].text(0.5, 0.5, "No Data (> threshold)", ha='center')
                    axes[0].axis('off')
                
                # 2. 中图：hrlt_t < threshold (Zoomed: X/Y < threshold*1.2)
                # 3. 右图：hrlt_t < threshold (Full X: Y < threshold*1.2, X Auto)
                # 寻找 < threshold 的数据
                idx_lt = -1
                for i, lab in enumerate(all_labels):
                    if "<" in lab:
                        idx_lt = i
                        break
                
                if idx_lt != -1:
                    x_lt = all_x[idx_lt]
                    y_lt = all_y[idx_lt]
                    label_lt = all_labels[idx_lt]
                    
                    limit_val = float(scatter_hrlt_threshold) * 1.2
                    
                    # --- 中图 (Zoomed) ---
                    ax = axes[1]
                    ax.scatter(x_lt, y_lt, alpha=0.3, s=1, c="blue", edgecolors='none')
                    ax.plot([0, limit_val], [0, limit_val], "k--", linewidth=2, label="1:1", alpha=0.7)
                    
                    ax.set_xlabel(f"{pobs_col} (mm)")
                    ax.set_ylabel(f"{hrlt_col} (mm)")
                    ax.set_title(f"{label_lt}\n(Zoomed to {limit_val:.1f}mm)", fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([0, limit_val])
                    ax.set_ylim([0, limit_val])
                    
                    # --- 右图 (Full X) ---
                    ax = axes[2]
                    ax.scatter(x_lt, y_lt, alpha=0.3, s=1, c="green", edgecolors='none')
                    # 1:1 线 (延伸到全图)
                    ax.plot([0, x_max], [0, x_max], "k--", linewidth=1, alpha=0.5, label="1:1")
                    
                    ax.set_xlabel(f"{pobs_col} (mm)")
                    ax.set_ylabel(f"{hrlt_col} (mm)")
                    ax.set_title(f"{label_lt}\n(Full X Range, Y < {limit_val:.1f}mm)", fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([x_min, x_max])  # 横轴全范围
                    ax.set_ylim([0, limit_val])  # 纵轴仍限制
                else:
                    axes[1].text(0.5, 0.5, "No Data (< threshold)", ha='center')
                    axes[1].axis('off')
                    axes[2].text(0.5, 0.5, "No Data (< threshold)", ha='center')
                    axes[2].axis('off')
                
                plt.suptitle(f"Scatter: {pobs_col} vs {hrlt_col} (threshold={scatter_hrlt_threshold}mm)", 
                           fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                
                if scatter_out_png is None:
                    base = os.path.splitext(parquet_path)[0]
                    scatter_out_png = f"{base}_scatter_{hrlt_col}_vs_{pobs_col}_subplots.png"
                out_dir_png = os.path.dirname(scatter_out_png)
                if out_dir_png:
                    os.makedirs(out_dir_png, exist_ok=True)
                plt.savefig(scatter_out_png, bbox_inches='tight')
                plt.close()
                print(f"[analyze_precipitation_distribution] 散点图已保存: {scatter_out_png}")
    
    return out_csv

    
if __name__ == '__main__':
    bounds = (117.5, 26.5, 123.5, 31.5)  # 浙江大致范围
    
        

    #3. 读取landcover地表覆盖数据  27.00138, 31.1986, 118.001389, 123.19861
    lc_file='/home/zsai/python_projects/prect500-data/static/C3S-LC-L4-LCCS-Map-300m-P1Y-2022-v2.1.1.area-subset.70.50.0.160.nc'
    lc_zj_file='/home/zsai/python_projects/prect500-data/static_new/landcover_zj300m.nc'
    #3.1 裁剪landcover地表覆盖数据到浙江范围
    shape, extent = save_landcover_zj_nc(lc_file, lc_zj_file)
    print('Landcover mosaic written:', lc_zj_file, 'shape:', shape, 'extent:', extent)
    exit()
    #3.0 读取landcover地表覆盖数据
    lats,lons,lc = read_landcover_zj(lc_file)
    print(lats.shape,lons.shape,lc.shape)
    print(lats[1]-lats[0],lats[1500]-lats[1499],   lons[1]-lons[0],lons[1500]-lons[1499])
    print(lats[0],lats[-1],lons[0],lons[-1]) 
    exit()
    #3.4landcover作图
    # 绘制原始 landcover
    plot_landcover_nc(
        "/home/zsai/python_projects/prect500-data/static/landcover_zj300m.nc",
        out_png="/home/zsai/python_projects/prect500/figs/landcover_zj300m.png",
        var_name="landcover",  )

    # 绘制重采样后的 landcover（与 DEM 共网格）
    plot_landcover_nc(
        "/home/zsai/python_projects/prect500-data/static/landcover_on_dem300m.nc",
    out_png="/home/zsai/python_projects/prect500/figs/landcover_on_dem.png",)
    exit()
    #3.2检查dem和landcover分辨率
    check_dem_and_landcover_resolution(
    "/home/zsai/python_projects/prect500-data/static/dem_zhejiang_10s.nc",
    "/home/zsai/python_projects/prect500-data/static/landcover_zj300m.nc",)
    #3.3将landcover重采样到dem网格；dem和landcover保持一样的网格
    resample_landcover_to_dem_grid(
    "/home/zsai/python_projects/prect500-data/static/landcover_zj300m.nc",
    "/home/zsai/python_projects/prect500-data/static/dem_zhejiang_10s.nc",
    "/home/zsai/python_projects/prect500-data/static/landcover_on_dem300m.nc",)
    exit()
    #3.5 生成landcover类别映射,用于类别编码
    generate_landcover_index_map(
    "/home/zsai/python_projects/prect500-data/static/landcover_on_dem300m.nc",
    "/home/zsai/python_projects/prect500-data/static/landcover_index_map.pkl",)
    exit()



    
    #2.4 比较 SRTM 和合并后的 DEM 在指定点上的差异
    lats = np.array([30.25, 29.87, 28.0,28.5,29.5])   # 杭州、宁波、温州
    lons = np.array([120.15, 121.55, 120.7,119.5,119.5])
    df = compare_dem_at_points('/home/zsai/python_projects/prect500-data/static_new/dem_zhejiang_10s_merged.nc',
    '/home/zsai/python_projects/prect500-data/static_new/dem_zhejiang_10s.nc',
    lats, lons,   label1='merged', label2='srtm')
    print(df)
    exit()
    #2.3 合并 SRTM 和 GEBCO DEM 到 10s 网格
    shape, extent = merge_dem_srtm_gebco(
    '/home/zsai/python_projects/prect500-data/static_new/dem_zhejiang_10s.nc',
    '/home/zsai/python_projects/prect500-data/static_new/dem_zhejiang_10s_gebco.nc',
    '/home/zsai/python_projects/prect500-data/static_new/dem_zhejiang_10s_merged.nc')

    #2.  DEM 镶嵌输出 NetCDF  27.001974 118.005295 31.193642 123.19696#2.  DEM 镶嵌输出 NetCDF  27.001974 118.005295 31.193642 123.19696
    #2.2 重采样 GEBCO 到 10s 网格
    shape, extent = resample_gebco_nc_to_grid(
        gebco_nc='/home/zsai/python_projects/prect500-data/static/GEBCO_2025_sub_ice_cropped.nc',
        out_nc='/home/zsai/python_projects/prect500-data/static_new/dem_zhejiang_10s_gebco.nc',
        target_bounds=(117.5, 26.5, 123.5, 31.5),
        target_res=10.0/3600.0,
        elev_var='elevation',
    )
    dem_folder = '/home/zsai/python_projects/prect500-data/static/srtm-zj'
    out_dem_nc = '/home/zsai/python_projects/prect500-data/static/dem_zhejiang.nc'
    #2.1 绘制 DEM 分布图
    plot_dem_nc(out_dem_nc,'figs/dem_zhejiang.png')
    exit()
    #2.0 镶嵌 DEM 瓦片
    bounds = (118.005295, 27.001974, 123.19696, 31.193642)  # 浙江大致范围
    res = 10.0/3600.0  # 300m（角度）
    shape, extent = mosaic_dem_tiles_to_nc(dem_folder, out_dem_nc, bounds, res)
    print('DEM mosaic written:', out_dem_nc, 'shape:', shape, 'extent:', extent)
    exit()
    
    
    
    
    
    
    
    
    #9 分析降水分布：统计不同阈值区间的数量及占比，绘制散点图
    analyze_precipitation_distribution(
        parquet_path="/home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024-train-and-test.parquet",
        hrlt_col="hrlt_t",
        pobs_col="P_obs_t",
        thresholds=[0, 0.1, 1, 2, 5, 10, 25, 50, 75, 100],
        out_csv="/home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024-train-and-test_precipitation_distribution_stats.csv",
        scatter_hrlt_threshold=10.0,  # 筛选 hrlt_t > 10mm 的样本
        scatter_hrlt_gt=True,  # True: > threshold, False: < threshold
        scatter_n_samples=100000,
        scatter_out_png="/home/zsai/python_projects/prect500-data/cmorph-stations/scatter_hrlt_vs_pobs_hrlt_gt_10mm.png",
        random_seed=42,
    )
    exit()
   
    
    #8 估计站点观测日降水Tweedie分布幂指数
    res = estimate_tweedie_power_from_station_obs(
    station_obs_parquet="/home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024.parquet",
    value_col="P_obs_t",
    date_col="date",     # 如果文件里只有 datetime，就会自动切到 "datetime"
    sid_col="sid",
    block_length_days=30,
    min_mean=0.1,        # 过滤几乎全 0 的块，你可以根据数据调
    min_var=0.01,
    drop_zero_only_blocks=True,
    out_json="/nas/zsai/prect500/samples/tweedie_power_p.json",
            )
    print(res)
    exit()
    
    df=pd.read_parquet("/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train_with_cmorph-train-and-test.parquet")
    print(df.shape)
    for col in df.columns:
        print(col, df[col].dtype)
    print(df[df['is_train']==True][['sid','dem','lc_type']].head(10))
    print(df[df['is_train']==False][['sid','dem','lc_type']].head(10))
    exit()
    #6.2 生成基础静态、时间谐波、格点日降水、背景风特征
    for year in [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]:
        build_daily_dynamic_features(
            year=year,
            stations_parquet="/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train_with_cmorph-train-and-test.parquet",
            precip_nc_pattern=f"/nas/zsai/prect500/cmorph_pcp/merged-30min_8km_daily/CMORPH_V1.0_ADJ_8km-daily_ZJ_merged_{year}.nc",
            wind_nc_pattern=f"/home/zsai/python_projects/prect500-data/era5_100m_uv_0p1/{year}_uv100_hourly_117.5-123.5_26.5-31.5.nc",
            out_parquet=f"/home/zsai/python_projects/prect500-data/cmorph-stations/part_daily_features-train-and-test/station_daily_features_{year}.parquet"
        )
    #6.3 生成（增加）站点观测日降水特征
    merge_dynamic_and_observations(
        dynamic_feature_paths=[f"/home/zsai/python_projects/prect500-data/cmorph-stations/part_daily_features-train-and-test/station_daily_features_{year}.parquet" for 
                        year in [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]],
        obs_csv="/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2024.csv",
        station_meta_parquet="/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train_with_cmorph-train-and-test.parquet",
        out_parquet=f"/home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024-train-and-test.parquet"
    )
    exit()
    #6.4 绘制缺失hrlt/cmorph降水的站点（位于海上）
    meta = pd.read_parquet("/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train_with_cmorph.parquet")
    nan_count = meta["cmorph_pcp_avg"].isna().sum()
    print(f"cmorph_pcp_avg 为 NaN 的站点数: {nan_count}")
    # 绘制包含缺失站点的降水分布图
    if nan_count >= 0:
        plot_prect_first_frame("/nas/zsai/prect500/cmorph_pcp/merged-30min_8km_daily/CMORPH_V1.0_ADJ_8km-daily_ZJ_merged_2014.nc","figs/cmorph_frame_15_with_nan_stations.png",
            frame_index=114+15,stations_df=meta,station_mask_col="cmorph_pcp_avg",station_mask_value=True,  # 筛选 NaN 值
            scatter_color="red",scatter_marker="x", scatter_size=80,
        )
    exit()
    
    #7、训练集的缩放系数计算 
    #   目标系数："pcp_q99_grid"  "pcp_q99_station"  "u_q99_abs" "v_q99_abs"  "dem_max"
    years_train = [2014, 2015, 2016, 2017, 2018, 2020, 2021]
    compute_scaling_factors_for_training(
        years=years_train,
        cmorph_pattern="/nas/zsai/prect500/cmorph_pcp/merged-30min_8km_daily/CMORPH_V1.0_ADJ_8km-daily_ZJ_merged_{year}.nc",
        dem_nc="/home/zsai/python_projects/prect500-data/static/dem_zhejiang_10s.nc",
        wind_pattern="/home/zsai/python_projects/prect500-data/era5_100m_uv_0p1/{year}_uv100_hourly_117.5-123.5_26.5-31.5.nc",
        station_obs_parquet="/home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024.parquet",
        out_json="/nas/zsai/prect500/samples/scaling-calculated-.json",
    )
    exit()
     
    
    df=pd.read_parquet("/home/zsai/python_projects/prect500-data/cmorph-stations/station_obs_merged_2014-2024.parquet")
    print(df.shape)
    for col in df.columns:
        print(col, df[col].dtype)
    exit()
    

    #6. 站点动态特征数据生成
    
    
    
    #6.1 站点观测日降水数据合并
    merge_station_observations(
        obs_folder="/home/zsai/python_projects/prect500-data/nb_stations_obs",
        output_csv="/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2019.csv",
        years=[2014, 2015, 2016, 2017, 2018, 2019]
    )
    exit()
    merge_monthly_station_obs_csvs(
    input_dir="/home/zsai/python_projects/prect500-data/nb_stations_obs",
    years=[ 2020, 2021,2022,2023,2024],
    out_csv="/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2024.csv",
    )
    exit()

    
    
    
    
    
    
    


    #5. 构造图 prepare_data_gen_graph.py


    #4. 站点静态特征数据生成
    # 打印站点静态特征数据统计信息
    print_station_meta_stats("/home/zsai/python_projects/prect500-data/stations/station_meta_train.parquet")
    exit()
    # 4.1 按dem和lc类型划分训练和测试集
    split_train_test_by_dem_lc("/home/zsai/python_projects/prect500-data/stations/station_meta.parquet",
                               "/home/zsai/python_projects/prect500-data/stations/station_meta_train.parquet",
                               test_frac=0.12,
                               dem_bins=10,
                               random_state=42)
    exit()
    
    #4.1 生成站点静态特征数据
    #4.1.2 站点静态表，补充静态地表数据和降水格点数据的行列索引
    add_station_grid_indices(
    "/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train_with_cmorph.parquet",)
    exit()
    #4.1.1 生成站点静态表
    build_station_attributes(
        csv_glob="../prect500-data/nb_stations_obs/浙江自动站日数据201*.csv",
        id_col="stationId", lat_col="lat", lon_col="lng",
        date_col="datetime", precip_col="r2020",
        dem_nc="/home/zsai/python_projects/prect500-data/static/dem_zhejiang_10s.nc",
        lc_nc="/home/zsai/python_projects/prect500-data/static/landcover_zj300m.nc",
        out_parquet="/home/zsai/python_projects/prect500-data/stations/station_meta.parquet"
        )
    exit()
    #4.1.1增加一列格点降水场的均值，用于构建特征图
    add_hrlt_pcp_avg(
    station_parquet_in="/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train.parquet",
    station_parquet_out="/home/zsai/python_projects/prect500-data/cmorph-stations/station_meta_train_with_cmorph.parquet",
    grid_pattern="/nas/zsai/prect500/cmorph_pcp/merged-30min_8km_daily/CMORPH_V1.0_ADJ_8km-daily_ZJ_merged_{year}.nc",
    years=[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    col_name="cmorph_pcp_avg",
    )#, 2022, 2023, 2024
    exit()
    
    

    


    
    

    #1. 读取浙江格点降水 27.001974 118.005295 31.193642 123.19696
    #1.2 绘制格点降水分布图
    plot_prect_first_frame("/nas/zsai/prect_1km/ZJ_1km_prep_2014.nc", 
    "figs/cnzj-prect20140702.png", target_time="2014-07-02", plot_total=False,
    out_png_total="figs/cnzj-prect_frame_0.png",
    frame_index=0)
    exit()

    #1.1 批量裁剪浙江格点降水到浙江范围
    nc_path = '/nas/zsai/prect_1km/China_1km_prep_2010.nc'
    batch_crop_prect_zj('/nas/zsai/prect_1km')
    exit()
    lats, lons, prep, time, ij = read_prect_1km(nc_path)
    print('子区域索引 i0:i1, j0:j1 =', ij)
    print('lats.shape:', np.shape(lats), 'lons.shape:', np.shape(lons))
    # 仅打印基本信息避免卡顿
    print('prep.shape:', np.shape(prep), 'time.shape:', np.shape(time))
    print(lats[1]-lats[0],lats[500]-lats[499],   lons[1]-lons[0],lons[600]-lons[599])
    print(lats[0],lats[-1],lons[0],lons[-1]) 
    exit()


    
    
    
    
    



    
    