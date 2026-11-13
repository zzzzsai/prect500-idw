# Python 文件功能说明

按数据流说明当前仓库里的脚本。现役网格降水与 USRI 产出在 `newout/`；`out/` 是早期 2016–2024 试验目录。

## 1. `prepare_data.py` / `prepare_data_gis.py`

底层地理数据处理库，以及 DEM 镶嵌 / GEBCO 重采样 / SRTM+GEBCO 合并的手工入口。

- 裁剪浙江范围、处理 SRTM 瓦片与 GEBCO，写出 `static/dem_zhejiang_*.nc`。
- `prepare_data_gis.py` 的 `__main__` 按注释块切换步骤，默认只画 `static/dem_zhejiang_500m_merged.nc`。

## 2. `data_eda.py`

站点观测质控与覆盖率统计，为插值准备有效日期和站点集合。

- 默认读取 `/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv`。
- 可按覆盖率阈值筛站、统计逐日有效站点数。
- 现役逐日站点数表：`newout/valid_station_daily_counts_2016_2025.csv`。

## 3. `interpolation.py`

把站点降水插值到浙江 500 m DEM 网格，并合并为逐年日 nc。

- 实现是 Scipy 薄板样条（`(lon, lat, elev)`），不是 ANUSPLIN 原软件。
- 函数默认 `out_dir` 仍写 `out/daily_precip_tps_500m`；**现役 `__main__` 使用 `newout/daily_precip_tps_500m`**，合并文件为 `prep_daily_2016_2025.nc`。
- `__main__` 用 `exit()` 切换步骤，改之前先看当前激活的那一段。

## 4. `plot_daily_precip.py`

从合并后的逐日 nc 抽一天，画浙江省降水空间分布图。默认读 `newout/daily_precip_tps_500m/prep_daily_2016_2025.nc`。

## 5. `typhoon_days.py`

用 IBTrACS 最佳路径判断影响浙江的台风个例和台风日。

- 浙江矩形框默认 `(118.0, 123.19, 27.05, 31.25)`，可在进入/离开前后加时间窗口。
- 现役轨迹图在 `newout/ibtracs.WP.list.v04r01-2016-2025_zhejiang_tracks_*.png`。

## 6. `hazard_mapping.py`

基于逐日降水 nc 计算极端降水指标（Rx1day、SDII、R10/20/50mm、R95p/R99p、台风贡献等），并出图。按年、月、台风日/非台风日筛选。USRI 的 R99p 输入来自这里的产物：`newout/r95p_typhoon_contribution/r99p_typhoon_contribution_2016_2025_1to12.nc`。

## 7. `fetch_zhejiang_tif.py`

一次性工具：用 HTTP Range 从 Zenodo 的 CLCD zip 里只抽浙江 tif，写出仓库根目录 `CLCD_v01_2025_albert_zhejiang.tif`。不是日常流水线。

## 8. `usri_mapping.py`

USRI 计算主脚本。CLI：`--mode hazard | exposure | usri | attribution | dem-stations`。默认路径在文件顶部常量。

- Hazard：等权合成 `Norm(R99p)`、`Norm(Prob)`、`Norm(SDII50)`，P0–P99.5 截断归一化。
- Exposure：人口 0.4 / GDP 0.3 / 耕地 0.15 / 建成区 0.15；人口和 GDP 先 `log1p`。
- 产出：`newout/usri/*.nc`。

## 9. `risk_assessment_mapping.py`

USRI 出图。`plot_risk_assessment_maps()` 一次生成全部图；也可单调 Hazard / Exposure / 风险分区 / 归因。自研 Jenks 自然断点（环境无 `jenkspy`）。图件在 `newout/usri/figs/`。

## 整体关系

1. DEM：`prepare_data.py` / `prepare_data_gis.py` → `static/`
2. 站点与日期：`data_eda.py`
3. 插值：`interpolation.py` → `newout/daily_precip_tps_500m/`
4. 台风日：`typhoon_days.py`
5. 极端指标：`hazard_mapping.py`
6. USRI：`usri_mapping.py` + `risk_assessment_mapping.py`
