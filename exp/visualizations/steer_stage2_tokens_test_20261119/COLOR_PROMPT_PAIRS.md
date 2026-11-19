# 同图颜色词对照：最新两组四相机可视化

两组使用同一个 Stage-2 checkpoint（epoch 10，step 8866），完整严格加载 509 项，无缺失或多余项。四相机图片同时输入，仅修改前相机文字，其他相机保留原始文字。共享色标，无 PCA。

| 对照 | 场景 | 综合图 | 7 页 PDF | 完整运行记录 |
|---|---|---|---|---|
| 白车 / 红车 | `4109b79d84ea5053` | [PNG](4109b79d84ea5053/cam_f0_white_red_car_4cam/comparison.png) | [PDF](4109b79d84ea5053/cam_f0_white_red_car_4cam/prompt_comparison.pdf) | [JSON](4109b79d84ea5053/cam_f0_white_red_car_4cam/run.json) |
| 白车 / 黑车 | `166b654c0db95953` | [PNG](166b654c0db95953/cam_f0_white_black_car_4cam/comparison.png) | [PDF](166b654c0db95953/cam_f0_white_black_car_4cam/prompt_comparison.pdf) | [JSON](166b654c0db95953/cam_f0_white_black_car_4cam/run.json) |

## 结果

- 白车 / 红车：两种提示的主要 patch 热点均集中于白车，未观察到明确的白车至红车切换。patch 分布总变差距离 0.041093，平均 register attention 分布总变差距离 0.060781。
- 白车 / 黑车：两种提示均响应黑车与画面左边缘，未观察到明确的颜色目标切换。patch 分布总变差距离 0.080187，平均 register attention 分布总变差距离 0.052121。

总变差距离为 `0.5 * sum(abs(p1-p2))`，范围 0–1，描述分布变化量，不是定位准确率。两组没有颜色专属人工目标标注；这些结果不构成正确颜色定位的正面证据，也不能概括全部语言理解能力。本次只执行视觉主干，没有执行下游规划和轨迹评分。

## 文件与复现

每组目录包含原图、7 张 PNG、7 页 PDF、HTML、前相机和四相机 NPZ 特征、运行记录、推理日志及当时使用的可视化脚本。四相机输入顺序为 `cam_f0, cam_b0, cam_l0, cam_r0`。目录中的 `index.html` 可下载后本地打开。

- [白车 / 红车网页](4109b79d84ea5053/cam_f0_white_red_car_4cam/index.html)
- [白车 / 黑车网页](166b654c0db95953/cam_f0_white_black_car_4cam/index.html)
- [白车 / 黑车详细说明和复现命令](166b654c0db95953/cam_f0_white_black_car_4cam/README.md)
- [汇总及源文件 SHA-256](color_prompt_pairs_summary.json)
- [可视化脚本](../../../scripts/viz/visualize_steer_prompt_pair.py)

PDF 和 NPZ 沿用仓库的 Git LFS 配置，克隆后运行 `git lfs pull` 获取。运行记录保留了原始本地路径，复现时需要调整数据和 checkpoint 路径；此更新未包含模型 checkpoint 或完整数据集。
