# 浙江省城市暴雨气候风险综合评估指数（USRI）分析方案

## 1. 研究目标

本项目面向浙江省 500 m 空间分辨率网格，构建城市暴雨气候风险综合评估指数（Urban Storm Rainfall Risk Index, USRI），用于识别暴雨气候风险的空间分布格局，并进一步划分低风险、中风险、中高风险和高风险区域。

综合风险指数采用如下结构：

```text
USRI = Hazard × Exposure
```

其中：

```text
Hazard = Σ αi × Norm(Xi)
Exposure = Σ βj × Norm(Yj)
```

`Hazard` 表示暴雨致灾因子危险性，`Exposure` 表示承灾体暴露度。所有子指标在参与合成前统一到浙江省 500 m 网格，并通过最大-最小值归一化缩放到 `[0, 1]`。

## 2. 现有数据基础

当前已有数据包括：

- 逐日降水合并文件：`newout/daily_precip_tps_500m/prep_daily_2016_2025.nc`
- 逐日降水空间分辨率：浙江省 500 m DEM 网格
- 已计算的极端降水百分位阈值：R95p、R99p
- 人口数据：LandScan 中国区域人口数据
- GDP 数据：`数据GDP2020年.rar`
- 30 m 地表覆盖数据：`N50_25_2025LC030.rar`、`N50_30_2025LC030.rar`、`N51_25_2025LC030.rar`、`N51_30_2025LC030.rar`
- 浙江省边界 shp：`/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp`
- 目标网格参考文件：`static/dem_zhejiang_500m_merged.nc`

## 3. Hazard 危险性指标体系

危险性部分基于 2016-2025 年逐日降水数据构建。初步建议采用 3 个核心指标：

### 3.1 R99p 极端降水强度

R99p 表示每个 500 m 格点在研究期内湿日降水的第 99 百分位阈值，用于刻画极端降水强度。

当前已在 `hazard_mapping.py` 中计算过 R95p/R99p 阈值，后续可直接读取已有 R99p 阈值 nc 文件作为危险性强度指标。

R95p 可作为敏感性分析指标，也可在后续模型比较中用于替代 R99p。

### 3.2 Prob 暴雨发生概率

Prob 用于刻画暴雨事件发生频率。按中国法定暴雨标准，日降水量大于等于 50 mm 记为暴雨日。

计算公式：

```text
Prob = N(RR >= 50 mm) / N(valid days)
```

其中 `N(RR >= 50 mm)` 为 2016-2025 年研究期内每个格点暴雨日数，`N(valid days)` 为该格点有效降水日数。

### 3.3 SDII50 暴雨日平均降水强度

SDII50 用于刻画暴雨日的平均降水强度。

计算公式：

```text
SDII50 = Σ RR(RR >= 50 mm) / N(RR >= 50 mm)
```

若某格点在研究期内没有暴雨日，则该格点 SDII50 记为 `NaN`。

### 3.4 Hazard 合成

初版危险性指数建议为：

```text
Hazard = α1 × Norm(R99p) + α2 × Norm(Prob) + α3 × Norm(SDII50)
```

初版曾建议 R99p 0.4 / Prob 0.3 / SDII50 0.3。**该建议已作废。** 当前代码与第 9 节实际权重为等权 1/3（`usri_mapping.py` 默认 `weights=(1/3, 1/3, 1/3)`）。若做敏感性分析，再改权重。

## 4. Exposure 暴露度指标体系

暴露度部分用于描述人口、经济资产和土地利用承灾体在空间上的暴露情况。建议采用 4 个指标：

### 4.1 人口密度

基于 LandScan 人口数据，统一处理到浙江省 500 m 网格。

若原始人口数据为栅格人口总量，应优先采用面积加权聚合或重采样方式得到 500 m 格点人口密度或人口数量。

### 4.2 GDP 空间密度

基于 GDP 数据构建空间化 GDP 密度指标，并统一到 500 m 网格。

若 GDP 数据为行政区统计数据，需要结合土地利用、夜光或人口等辅助变量进行空间化；若已是栅格数据，则重采样到目标网格。

### 4.3 耕地面积占比

基于 30 m 地表覆盖数据提取耕地类别，在每个 500 m 目标格点范围内统计耕地面积占比。

计算公式：

```text
CroplandRatio = cultivated land area / grid cell area
```

该指标用于反映农业承灾体暴露度。

### 4.4 城市建设区面积占比

基于 30 m 地表覆盖数据提取城市建设用地类别，在每个 500 m 目标格点范围内统计建设用地面积占比。

计算公式：

```text
BuiltupRatio = built-up area / grid cell area
```

该指标用于反映城市建成区承灾体暴露度。

### 4.5 Exposure 合成

初版暴露度指数建议为：

```text
Exposure = β1 × Norm(PopDensity)
         + β2 × Norm(GDPDensity)
         + β3 × Norm(CroplandRatio)
         + β4 × Norm(BuiltupRatio)
```

初始权重建议：

```text
PopDensity    : 0.35
GDPDensity    : 0.35
CroplandRatio : 0.15
BuiltupRatio  : 0.15
```

如果研究更强调城市内涝和城市资产暴露，可提高 `BuiltupRatio` 权重；如果更强调农业损失，可提高 `CroplandRatio` 权重。

## 5. 空间统一与预处理

所有指标必须统一到同一个目标网格：

```text
static/dem_zhejiang_500m_merged.nc
```

该网格同时也是逐日降水文件使用的 500 m 网格。

预处理原则：

- 降水指标已在目标网格上，可直接使用。
- 人口和 GDP 数据需要重采样或面积加权聚合到 500 m 网格。
- 30 m 地表覆盖数据需要按目标网格统计类别面积占比。
- 浙江省外区域统一设为 `NaN`。
- 所有指标使用一致的浙江省边界掩膜。
- 输出的中间指标均保留 `lat`、`lon` 坐标，便于后续叠加和绘图。

## 6. 指标归一化方法

所有正向指标采用最大-最小值归一化：

```text
Norm(X) = (X - Xmin) / (Xmax - Xmin)
```

处理要求：

- 只在浙江省有效区域内计算 `Xmin` 和 `Xmax`。
- `NaN` 区域不参与最小值和最大值计算。
- 归一化后仍将省外区域保持为 `NaN`。
- 若某指标存在极端异常值，可考虑使用 1% 和 99% 分位数截尾后再归一化。

## 7. USRI 合成与风险分级

在获得 `Hazard` 和 `Exposure` 后，计算：

```text
USRI = Hazard × Exposure
```

该乘积形式可以避免以下两类误判：

- 危险性高但人口和资产暴露低的区域被过度高估。
- 人口和资产暴露高但暴雨危险性较低的区域被过度高估。

USRI 输出后，建议同时生成连续型指数图和风险等级图。

风险等级划分建议采用自然断点法（Jenks natural breaks），划分为 4 级：

```text
低风险
中风险
中高风险
高风险
```

若 Jenks 分类结果受局部极端值影响较大，可与分位数法结果进行对比。

## 8. 产出文件汇总

USRI 相关结果统一输出到：

```text
newout/usri/
newout/usri/figs/
```

主要脚本：

```text
usri_mapping.py                 # Hazard / Exposure / USRI / 归因计算；DEM 与站点分布图
risk_assessment_mapping.py      # Hazard / Exposure / USRI 风险分区 / 归因绘图
```

### 8.1 数据产品（NetCDF）

| 步骤 | 产出内容 | 文件路径 |
|------|----------|----------|
| 第一步 Hazard | 原始与归一化危险性因子、Hazard 综合指数 | `newout/usri/hazard_r99p_prob_sdii50_2016_2025.nc` |
| 第二步 Exposure | 原始与归一化暴露度因子、Exposure 综合指数 | `newout/usri/exposure_population_gdp_landcover_500m.nc` |
| 第三步 USRI | Hazard、Exposure、USRI | `newout/usri/usri_2016_2025_500m.nc` |
| 第五步 归因 | 主导驱动类型、Hazard/Exposure 因子贡献率 | `newout/usri/usri_driver_type_500m.nc` |

### 8.2 图件产品（PNG）

| 步骤 | 产出内容 | 文件路径 |
|------|----------|----------|
| 基础底图 | DEM 与降水站点分布 | `newout/usri/figs/dem_station_distribution_500m.png` |
| 第一步 Hazard | 危险性综合空间分布 | `newout/usri/figs/hazard_spatial_distribution_2016_2025.png` |
| 第一步 Hazard | R99p / SDII50 / Prob 因子拼图 | `newout/usri/figs/hazard_factors_r99p_sdii50_prob_2016_2025.png` |
| 第一步 Hazard | 归一化危险性因子拼图 | `newout/usri/figs/hazard_normalized_factors_r99p_sdii50_prob_2016_2025.png` |
| 第二步 Exposure | 暴露度综合空间分布 | `newout/usri/figs/exposure_spatial_distribution_500m.png` |
| 第二步 Exposure | 人口 / GDP / 耕地 / 建成区因子拼图 | `newout/usri/figs/exposure_factors_population_gdp_landcover_500m.png` |
| 第二步 Exposure | 归一化暴露度因子拼图 | `newout/usri/figs/exposure_normalized_factors_population_gdp_landcover_500m.png` |
| 第四步 风险分级 | USRI 四等级风险分区图 | `newout/usri/figs/usri_risk_zoning_2016_2025_500m.png` |
| 第五步 归因 | 主导驱动类型空间分布 | `newout/usri/figs/usri_driver_type_map_500m.png` |
| 第五步 归因 | Hazard 内部因子贡献率图 | `newout/usri/figs/usri_hazard_factor_contribution_500m.png` |
| 第五步 归因 | Exposure 内部因子贡献率图 | `newout/usri/figs/usri_exposure_factor_contribution_500m.png` |

### 8.3 关键输入数据路径

| 用途 | 文件路径 |
|------|----------|
| 逐日降水合并文件 | `newout/daily_precip_tps_500m/prep_daily_2016_2025.nc` |
| R99p 阈值文件 | `newout/r95p_typhoon_contribution/r99p_typhoon_contribution_2016_2025_1to12.nc` |
| 500 m DEM 网格 | `static/dem_zhejiang_500m_merged.nc` |
| 站点观测 CSV | `/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv` |
| 人口 / GDP / GlobeLand30 数据目录 | `/nas/zsai/python_projects/prect500-idw-gis-data` |
| 浙江省边界 shp | `/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp` |

## 9. 实施步骤与对应产出

### 第零步：DEM 与站点分布底图

读取 500 m DEM 与站点经纬度，叠加浙江省边界，输出 DEM 与降水站点分布图。

**产出：**

- 图件：`newout/usri/figs/dem_station_distribution_500m.png`

**运行方式：**

```bash
python usri_mapping.py --mode dem-stations
```

### 第一步：计算 Hazard 指标

从 `prep_daily_2016_2025.nc` 读取逐日降水，计算：

- `Prob`
- `SDII50`

读取已有 R99p 阈值文件，构建：

- `Norm(R99p)`
- `Norm(Prob)`
- `Norm(SDII50)`
- `Hazard`

当前实际权重为等权重：

```text
Hazard = 1/3 × Norm(R99p) + 1/3 × Norm(Prob) + 1/3 × Norm(SDII50)
```

归一化采用 P0–P99.5 百分位截断后的 robust min-max。

**产出：**

- 数据：`newout/usri/hazard_r99p_prob_sdii50_2016_2025.nc`
  - 变量：`r99p`、`prob`、`sdii50`、`norm_r99p`、`norm_prob`、`norm_sdii50`、`hazard`
- 图件：
  - `newout/usri/figs/hazard_spatial_distribution_2016_2025.png`
  - `newout/usri/figs/hazard_factors_r99p_sdii50_prob_2016_2025.png`
  - `newout/usri/figs/hazard_normalized_factors_r99p_sdii50_prob_2016_2025.png`

**运行方式：**

```bash
python usri_mapping.py --mode hazard
python -c "from risk_assessment_mapping import plot_hazard_maps; plot_hazard_maps()"
```

### 第二步：处理 Exposure 指标

将人口、GDP 和 30 m GlobeLand30 地表覆盖数据统一到 500 m 网格，计算：

- `PopDensity`
- `GDPDensity`
- `CroplandRatio`
- `BuiltupRatio`
- `Exposure`

当前实际权重：

```text
Exposure = 0.4 × Norm(PopDensity)
         + 0.3 × Norm(GDPDensity)
         + 0.15 × Norm(CroplandRatio)
         + 0.15 × Norm(BuiltupRatio)
```

人口和 GDP 先做 `log1p` 变换，再做 P0–P99.5 百分位截断归一化；耕地和建成区直接做百分位截断归一化。耕地代码为 `10`，建成区代码为 `80`。

**产出：**

- 数据：`newout/usri/exposure_population_gdp_landcover_500m.nc`
  - 变量：`pop_density`、`gdp_density`、`cropland_ratio`、`builtup_ratio`、对应 `norm_*` 与 `exposure`
- 图件：
  - `newout/usri/figs/exposure_spatial_distribution_500m.png`
  - `newout/usri/figs/exposure_factors_population_gdp_landcover_500m.png`
  - `newout/usri/figs/exposure_normalized_factors_population_gdp_landcover_500m.png`

**运行方式：**

```bash
python usri_mapping.py --mode exposure
python -c "from risk_assessment_mapping import plot_exposure_maps; plot_exposure_maps()"
```

### 第三步：合成 USRI

读取 `Hazard` 和 `Exposure`，计算：

```text
USRI = Hazard × Exposure
```

**产出：**

- 数据：`newout/usri/usri_2016_2025_500m.nc`
  - 变量：`hazard`、`exposure`、`usri`

**运行方式：**

```bash
python usri_mapping.py --mode usri
```

### 第四步：风险分级

对浙江省有效区域内的 USRI 值进行自然断点分类，划分为低、中、中高、高 4 级，输出风险分区图。

本次实际自然断点为：

```text
低风险：   0.000000 – 0.077042
中风险：   0.077042 – 0.160950
中高风险： 0.160950 – 0.273844
高风险：   0.273844 – 0.781104
```

**产出：**

- 图件：`newout/usri/figs/usri_risk_zoning_2016_2025_500m.png`

**运行方式：**

```bash
python -c "from risk_assessment_mapping import plot_usri_risk_zoning_map; plot_usri_risk_zoning_map()"
```

### 第五步：危险性、暴露度归因分析

风险分级完成后，进一步诊断不同高风险区的形成原因，回答“风险高主要由危险性驱动，还是由暴露度驱动，或二者共同驱动”的问题。

#### 5.1 Hazard-Exposure 二维归因矩阵

读取每个 500 m 格点的 `Hazard`、`Exposure`、`USRI`，将 `Hazard` 与 `Exposure` 分别按 P75 划分为高、低两类，构建二维归因矩阵：

```text
Hazard 高 + Exposure 高     -> 双高驱动型
Hazard 高 + Exposure 中低   -> 危险性主导型
Hazard 中低 + Exposure 高   -> 暴露度主导型
Hazard 中低 + Exposure 中低 -> 一般风险区
```

本次实际阈值：

```text
Hazard P75   = 0.534872
Exposure P75 = 0.374229
```

#### 5.2 主导类型空间分布图

根据二维归因矩阵，为每个格点赋予主导类型：

- `0`：一般风险区
- `1`：危险性主导型
- `2`：暴露度主导型
- `3`：双高驱动型

#### 5.3 Hazard 内部因子贡献率

由于当前危险性指数采用等权重合成：

```text
Hazard = 1/3 × Norm(R99p) + 1/3 × Norm(Prob) + 1/3 × Norm(SDII50)
```

可计算每个格点的危险性因子贡献率：

```text
R99p_contrib   = (1/3 × Norm(R99p)) / Hazard
Prob_contrib   = (1/3 × Norm(Prob)) / Hazard
SDII50_contrib = (1/3 × Norm(SDII50)) / Hazard
```

#### 5.4 Exposure 内部因子贡献率

当前暴露度指数权重为：

```text
Exposure = 0.4 × Norm(PopDensity)
         + 0.3 × Norm(GDPDensity)
         + 0.15 × Norm(CroplandRatio)
         + 0.15 × Norm(BuiltupRatio)
```

可计算每个格点的暴露度因子贡献率：

```text
Pop_contrib      = (0.4 × Norm(PopDensity)) / Exposure
GDP_contrib      = (0.3 × Norm(GDPDensity)) / Exposure
Cropland_contrib = (0.15 × Norm(CroplandRatio)) / Exposure
Builtup_contrib  = (0.15 × Norm(BuiltupRatio)) / Exposure
```

#### 5.5 产出文件

- 数据：`newout/usri/usri_driver_type_500m.nc`
  - 变量：`driver_type`、`r99p_contrib`、`prob_contrib`、`sdii50_contrib`、`pop_contrib`、`gdp_contrib`、`cropland_contrib`、`builtup_contrib`，以及 `hazard`、`exposure`、`usri`
- 图件：
  - `newout/usri/figs/usri_driver_type_map_500m.png`
  - `newout/usri/figs/usri_hazard_factor_contribution_500m.png`
  - `newout/usri/figs/usri_exposure_factor_contribution_500m.png`

**运行方式：**

```bash
python usri_mapping.py --mode attribution
python -c "from risk_assessment_mapping import plot_attribution_maps; plot_attribution_maps()"
```

### 第六步：结果检查

建议逐项检查：

- 每个子指标是否和浙江省边界一致。
- 是否存在异常的空值区域。
- 归一化后是否在 `[0, 1]`。
- Hazard、Exposure 和 USRI 的高值区是否符合地理和气候认知。
- 分级结果是否稳定，可与分位数法做对比。

## 10. 脚本与开发状态

已实现脚本：

```text
usri_mapping.py
risk_assessment_mapping.py
```

已完成功能：

1. 通用 nc 读写与 robust min-max 归一化。
2. `compute_hazard_index()` 与 Hazard 绘图。
3. 人口 / GDP / 土地覆盖重采样与 `compute_exposure_index()`。
4. `compute_usri_index()`。
5. Jenks 风险分级与风险分区绘图。
6. `compute_usri_attribution()` 与归因绘图。
7. DEM 与站点分布绘图。

