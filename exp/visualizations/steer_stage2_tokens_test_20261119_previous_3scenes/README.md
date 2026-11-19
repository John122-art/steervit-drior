# 测试结果与阅读说明

使用 `best-epoch=10-step=8866.ckpt`，严格加载全部 509 个状态项，缺失/多余项均为 0。选择 3 个不同日志的 test 场景，共 12 张图片；选择在模型推理之前完成，没有按结果筛选。

优先打开 `index.html` 浏览完整图集，或打开 `token_comparison.pdf`（24 页：每张图片先放综合对照，再放原始语言下的 16 个场景 token）。空语言和错配语言下的逐 token 图也保存在各图片子目录。

综合对照图中：第一行是输入/参考框、自动生成参考区域、原始语言相对空语言造成的 patch 特征变化；第二行是输出 patch 经训练后的 grounding head 得到的定位图；第三行是 16 个场景 token 的平均自注意力；第四行是编码器最终 patch 特征的 PCA。第二至四行从左到右均为原始语言、空语言、错配语言。

| 条件 | 参考区域内的平均 patch 定位概率 | 热图最大值落入参考区域 |
|---|---:|---:|
| 原始语言 | 84.22% | 10/11 |
| 空语言 | 27.47% | 6/11 |
| 错配语言 | 78.49% | 10/11 |

这里的百分比是空间 softmax 概率落在参考 mask 内的质量，不是准确率或 IoU。12 张图中 11 张有有效参考 mask，另 1 张仍展示但不计入定位统计。参考 mask 是数据流水线自动生成的 audit 标签，可能有遗漏或偏差。

观察：11/11 张有效图片的原始语言定位概率质量高于空语言，9/11 高于错配语言。patch grounding 能明显集中于车辆、骑行者等目标，16 个场景 token 的注意力则还覆盖道路、交通灯、边缘等信息。两种图含义不同，不能把分数当作同一指标比较。

关键限制：错配语言的平均结果仍达 78.49%，有 2 张图超过原始语言；例如骑行者样本的前视图，原始/空/错配语言分别为 92.92%/10.60%/84.43%。这些结果支持“语言分支影响了输出”，但不足以证明它能严格按照每条文字区分目标；通用目标显著性也可能贡献很大。PCA 色彩也不代表目标类别。

完整原始特征保存在每张图目录的 `features.npz`：三个条件的 patch 特征 [3,3936,768]、场景特征 [3,16,768]、neck 后场景特征 [3,16,256]、注意力和 grounding 概率。特征保存为 FP16，概率/注意力为 FP32。`metrics.csv` 保存逐图统计，`run.json` 保存加载记录，`validation.json` 保存产物检查结果。

这次运行的是原生逐相机图文编码器路径，使用该相机对应的 scene embeddings；没有运行轨迹规划评分，也没有训练或改动权重。

---

# Stage-2 token visualization

Checkpoint: `/mnt/workspace/project/LQH/Drivor-orial/DrivoR-main/exp/ke/steer_stage2_capatch_vitb_8gpu_b16_12ep/11.13_05.07/lightning_logs/version_0/checkpoints/best-epoch=10-step=8866.ckpt`

Strict load: 509 tensors; epoch 10, step 8866.

Open index.html for all views. Each view contains comparison.png, three scene_tokens_*.png grids, image.jpg, metadata.json and features.npz.

Scene tokens: last self-attention block, average of all 12 heads, query rows 0..15, patch columns 17..3952. Patch-conditional normalization is used for display and spatial mass metrics; raw full-softmax attention is saved. Hooks only read QKV.

Patch tokens: final encoder norm, 48 x 82 x 768; grounding uses the checkpoint linear head and spatial softmax. PCA uses a joint basis and color scale across the three text conditions within each view. After-neck scene tokens (16 x 256) are also saved.

Correct and empty language share the same image. Shuffled language is the same camera caption from the next selected scene (cyclic). Empty language disables CA through the native row_active mask. No GridMask or text dropout in eval mode. Camera rows are independent in this encoder; their own learned scene embeddings are selected.

Targets are the dataset automatic audit-only projection masks, not independently hand-labeled evidence. Invalid views are displayed but omitted from localization metrics. A mismatched caption can still describe some visible objects; these examples cannot establish dataset-wide accuracy or causal object-level correctness.

Heatmaps clip at the 99.5th percentile only for display. Each comparison row shares a scale across language conditions; each numbered scene token shares a scale across its three plots. Grounding and attention are distinct quantities, so colors across different rows are not directly comparable. All metrics use unclipped probabilities.

Summary (valid views only):

```json
{
  "correct": {
    "valid_views": 11,
    "mean_target_mass": 0.8421769358895042,
    "peak_hits": 10,
    "mean_register_target_mass": 0.16753869605335323
  },
  "empty": {
    "valid_views": 11,
    "mean_target_mass": 0.27468185567043046,
    "peak_hits": 6,
    "mean_register_target_mass": 0.14093819015066733
  },
  "shuffled": {
    "valid_views": 11,
    "mean_target_mass": 0.7849251113154672,
    "peak_hits": 10,
    "mean_register_target_mass": 0.1625350568104874
  }
}
```
