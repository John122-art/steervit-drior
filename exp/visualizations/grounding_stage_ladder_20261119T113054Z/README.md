# Grounding stage ladder — 2026-11-19

比较 `official`、`pure`、`init`、`stage1`、`stage2` 五个阶段在同图不同颜色/方位提示下的定位头与 CLS 注意力输出，包含无文本条件。原始输出目录完整保留。

| 场景 | 定位头图 | CLS 注意力图 | 指标与说明 | 运行记录 |
|---|---|---|---|---|
| 白车 / 黑车 `166b654c0db95953` | [PNG](166b654c0db95953_cam_f0/seg_grid.png) | [PNG](166b654c0db95953_cam_f0/cls_grid.png) | [summary.md](166b654c0db95953_cam_f0/summary.md) | [run.json](166b654c0db95953_cam_f0/run.json) |
| 白车 / 红车 `4109b79d84ea5053` | [PNG](4109b79d84ea5053_cam_f0/seg_grid.png) | [PNG](4109b79d84ea5053_cam_f0/cls_grid.png) | [summary.md](4109b79d84ea5053_cam_f0/summary.md) | [run.json](4109b79d84ea5053_cam_f0/run.json) |

每组包含两张图、`metrics.json`、`run.json`、`maps.npz`、`summary.md` 和生成脚本 `compare_grounding_stages.py`。NPZ 使用 Git LFS，克隆后运行 `git lfs pull` 获取。

指标是手工框内的空间概率质量/注意力占比及分布总变差，不是定位准确率或规划评分。不同阶段的分辨率与结构设置以各组 `run.json` 为准。运行记录保留原始数据和模型路径；复现需要对应数据及 checkpoint，并调整路径。
