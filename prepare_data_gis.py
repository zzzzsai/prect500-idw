import numpy as np
from prepare_data import compare_dem_at_points,merge_dem_srtm_gebco,resample_gebco_nc_to_grid,plot_dem_nc,mosaic_dem_tiles_to_nc
DEFAULT_PROVINCE_SHP_PATH = "/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp"

if __name__ == '__main__':
    #2.  DEM 镶嵌输出 NetCDF; 以srtm为主，缺的用gebco填补 
    
    dem_folder = '/home/zsai/python_projects/prect500-data/static/srtm-zj'
    #out_dem_nc = 'static/dem_zhejiang_srtm500m.nc'
    bounds = (118.0, 27.05, 123.19, 31.25)  # 浙江大致范围
    res = 0.005  # 500m
    #2.1 绘制 DEM 分布图
    plot_dem_nc(
        'static/dem_zhejiang_500m_merged.nc',
        'figs/dem_zhejiang_srtm500m-merged.png',
        province_shp_path=DEFAULT_PROVINCE_SHP_PATH,
    )
    exit()
    #2.4 比较 SRTM 和合并后的 DEM 在指定点上的差异
    lats = np.array([30.25, 29.87, 28.0,28.5,29.5])   # 杭州、宁波、温州
    lons = np.array([120.15, 121.55, 120.7,119.5,119.5])
    df = compare_dem_at_points('static/dem_zhejiang_500m_merged.nc',
    'static/dem_zhejiang_srtm500m.nc',
    lats, lons,   label1='merged', label2='srtm')
    print(df)
    exit()
    #2.3 合并 SRTM 和 GEBCO DEM 到 10s 网格
    shape, extent = merge_dem_srtm_gebco(
    'static/dem_zhejiang_srtm500m.nc',
    'static/dem_zhejiang_500m_gebco.nc',
    'static/dem_zhejiang_500m_merged.nc')
    exit()


    #2.2 重采样 GEBCO 到 10s 网格
    shape, extent = resample_gebco_nc_to_grid(
        gebco_nc='/home/zsai/python_projects/prect500-data/static/GEBCO_2025_sub_ice_cropped.nc',
        out_nc='static/dem_zhejiang_500m_gebco.nc',
        target_bounds=bounds,
        target_res=res,
        elev_var='elevation',
    )
    exit()

    #2.0 镶嵌 DEM 瓦片
    
    shape, extent = mosaic_dem_tiles_to_nc(dem_folder, out_dem_nc, bounds, res)
    print('DEM mosaic written:', out_dem_nc, 'shape:', shape, 'extent:', extent)
    
    exit()
    


    
    
    
    