# prect500-idw

用浙江省地面站降水做 500 m 网格插值，计算极端降水指标，并合成城市暴雨气候风险指数（USRI = Hazard × Exposure）。这是单机科研脚本仓库，没有 Web 服务、没有 git remote。

## 现役结论（2026-11-13）

USRI 全流程（Hazard → Exposure → USRI → Jenks 四级分区 → P75 归因）已跑通，数据与图在 `newout/usri/`。方案文档第 6 步「结果检查」尚未做。

权威机制与命令见 [`USRI暴雨风险区划分析方案.md`](USRI暴雨风险区划分析方案.md) 第 8–10 节。Agent 约束见 [`AGENTS.md`](AGENTS.md)。脚本职责见 [`Python文件功能说明.md`](Python文件功能说明.md)。

## 怎么跑

```bash
python usri_mapping.py --mode hazard
python usri_mapping.py --mode exposure
python usri_mapping.py --mode usri
python usri_mapping.py --mode attribution
python usri_mapping.py --mode dem-stations
python -c "from risk_assessment_mapping import plot_risk_assessment_maps; plot_risk_assessment_maps()"
```

当前实现：Hazard 三因子等权 1/3；Exposure 人口 0.4 / GDP 0.3 / 耕地 0.15 / 建成区 0.15；归一化用 P0–P99.5 截断。不要使用方案第 3.4 节里已标明作废的 0.4/0.3/0.3 建议权重。

上游 TPS 插值与逐日 nc 合并在 `interpolation.py`。台风日划分在 `typhoon_days.py`，指标在 `hazard_mapping.py`。这些脚本的 `__main__` 是手工切换的实验入口，不是稳定 CLI。

## 目录

| 路径 | 角色 |
|------|------|
| `usri_mapping.py` / `risk_assessment_mapping.py` | USRI 计算与出图 |
| `newout/` | 现役插值、指标、USRI 产出 |
| `out/`、`newout-wet-seasons/` | 早期 2016–2024 / 汛期试验，非现役输入 |
| `static/` | 500 m DEM |
| `实施方案.md` | 更早的汛期 SCI 研究计划，不是 USRI 运行说明 |

## 外部依赖（本机核查）

| 路径 | 2026-11-13 状态 |
|------|-----------------|
| `newout/daily_precip_tps_500m/prep_daily_2016_2025.nc` | 存在 |
| `/home/zsai/python_projects/prect500-v2/china_shp/省界_Project.shp` | 存在 |
| `/home/zsai/python_projects/prect500-data/nb_stations_obs/station_obs_merged_2014-2025.csv` | **缺失**（重跑 dem-stations / 插值需要） |
| `/nas/zsai/python_projects/prect500-idw-gis-data` | **缺失**（重跑 Exposure 需要） |

## 下一步

1. 按方案第 6 步做结果检查（边界、空值、归一化范围、高值区、与分位数分级对比）。
2. 可选：Hazard 权重与归因 `--high-percentile` 敏感性分析。
3. 用户已排除分区统计和典型区域诊断。
