# prect500-idw

浙江省 500 m 格点降水插值、极端降水指标与 USRI 暴雨风险区划的本地科研脚本仓库。无服务、无远端、无 CI。

## 怎么跑

在仓库根目录、已有 `netCDF4` / `numpy` / `pandas` / `rasterio` / `scipy` / `matplotlib` 的 Python 环境中执行。USRI 现役命令：

```bash
python usri_mapping.py --mode hazard        # 或 exposure / usri / attribution / dem-stations
python -c "from risk_assessment_mapping import plot_risk_assessment_maps; plot_risk_assessment_maps()"
```

上游插值入口是 `interpolation.py` 的 `__main__`（按注释块切换步骤，默认会 `exit()` 提前结束）。详细路径与权重以 `USRI暴雨风险区划分析方案.md` 第 8–10 节为准。

## 技术栈

Python 脚本 + NetCDF/GeoTIFF + Matplotlib 出图。网格参考 `static/dem_zhejiang_500m_merged.nc`。省界 shp 在兄弟仓库 `prect500-v2/china_shp/`。

## 目录与约定

- 现役产出：`newout/`（逐日降水、指标、`newout/usri/`）。
- 早期 2016–2024 / 汛期试验：`out/`、`newout-wet-seasons/`，不要当现役输入。
- `*.nc` / `*.csv` / `*.png` 已被 gitignore；不要把大体积数据重新纳入版本库。
- 用户已确认**不做**分区统计与典型区域诊断，不要加回方案或代码。
- 自然断点图备份 `newout/usri/figs/usri_risk_zoning_2016_2025_500m-自然断点.png` 不要覆盖。

## 当前状态与下一步

截至 2026-11-13：Hazard → Exposure → USRI → 分级 → 归因及图件已在 `newout/usri/`。未做方案第 6 步结果检查。本机缺少站点 CSV `prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv` 和 GIS 目录 `/nas/zsai/python_projects/prect500-idw-gis-data`，重算 Exposure / DEM-站点图前先确认这两处。
