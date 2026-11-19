# 30 场景扩展可视化（无 PCA）

当前结果：30 个不同日志的 test 场景、120 张图片、360 个语言条件推理。
checkpoint：`/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main/exp/ke/steer_stage2_capatch_vitb_8gpu_b16_12ep/11.13_05.07/lightning_logs/version_0/checkpoints/best-epoch=10-step=8866.ckpt`，epoch 10、step 8866。
完整模型严格加载 509 个状态项，缺失和多余项均为 0。

## 更新后的统计

| 语言条件 | 有效图片 | 目标区域内平均 patch 概率 | 最高点命中 |
|---|---:|---:|---:|
| 原始语言 | 96 | 76.36% | 91/96 |
| 空语言 | 96 | 22.92% | 47/96 |
| 离谱无关语言 | 96 | 52.56% | 66/96 |

原始语言相对空语言，目标区域内概率平均变化 +53.43 个百分点，在 95/96 张有效图片上更高；相对离谱无关语言，平均变化 +23.79 个百分点，在 85/96 张上更高。

这些统计描述的是定位分布随语言改变的情况，不能单独证明逐词语义理解或规划收益。尤其无关文字并不描述图中真实目标；其“目标区域概率”仍使用原始文字对应的参考 mask，衡量的是保留了多少原目标响应。空间 softmax 必须把概率分配到某些 patch，当前头没有“没有匹配对象”输出，因此错误语言下仍出现热区不等于模型认为文字正确。离谱文字是分布外测试，不能代替同场景、不同真实对象之间的精细语义对照。

以上百分比是空间概率质量，不是准确率或 IoU。参考区域来自自动生成的 audit 标签。
全部 120 张图片均展示；统计只纳入参考 mask 有效且原始文字非空的图片。
统计表按有效图片等权平均，run.json 另提供按场景等权的结果及按场景重采样的配对差值区间；这些区间仅描述本次选取样本的波动。

## 与上一版的关系

上一版只有 3 个场景，原始/空/其他驾驶场景文字的概率分别为 84.22%/27.47%/78.49%。
本版保留原来 3 个场景，再用 seed=2 从其余日志随机选择 27 个不同日志、每个日志随机选一个场景；选择发生在推理之前，不按效果筛选。
本版错误输入改为固定的奇幻/太空/海底等无关描述，不再使用其他驾驶场景的文字。
样本和错误输入均改变，因此不能把本版与旧版百分比的差异解释为模型能力发生变化。以上扩展统计替代旧版的小样本结论。

## 怎样读图

打开 index.html；总览 PDF 是 token_comparison.pdf，每个场景还有 8 页 scene_comparison.pdf。
综合对照的第一行：输入及参考框、自动参考 mask、原始语言相对空语言引起的 patch 特征相对 L2 变化。
第二行：最终 patch 特征经过已训练 grounding head 和空间 softmax 后的定位图。
第三行：最后一层 12 个头平均后的 16 个场景 token 自注意力均值。
第二、三行从左到右为原始、空、离谱无关语言。图顶及每个子目录 metadata.json 显示实际文字。
scene_tokens_correct/empty/unrelated.png 分别显示 16 个 token 的逐个注意力。

场景 token 图是最后一层的注意力读出，patch 图是有定位监督的预测头输出，两者含义和监督不同。
不能根据热图更集中就断言 patch 表征更好、16 token 丢失了全部定位信息，或直接使用 patch 一定改善规划。
本次没有规划评分，只有原生逐相机图文编码器推理。

## 复现与数据

图像预处理与工程一致：1148×672，ImageNet 标准化；每个相机使用自己的 scene embeddings。
使用 eval、FP16 autocast，RoBERTa 和可视化注意力重建使用 FP32。无 GridMask/text dropout。
三个条件作为独立相机行组成 batch=3，模型没有行间交互；训练配置及 checkpoint 均不修改。
场景 token 的 query 是 0..15；patch 从 17 起，共 48×82=3936 个。hook 只读 QKV，不改变推理。
注意力先对所有 key 做 softmax，再取 patch 列；显示时按每个场景 token 的 patch 总质量归一化。
热图仅显示时截断到 99.5 分位；同一行的三个条件共用色阶，每个编号 token 也共用三条件色阶。统计使用未截断的概率。

每个图片目录包含原图、4 张可视化、metadata.json、features.npz。
features.npz 保存三个条件的 patch [3,3936,768]、scene [3,16,768]、neck 输出 [3,16,256]（FP16），以及概率、logits、原始注意力等 FP32 数组；没有 PCA 数组。
selected_samples.json、unrelated_prompts.json 和 selection.json 在推理之前固定样本与错误输入。
metrics.csv/metrics.json 保存逐图统计；run.json 保存加载、精度、汇总和自检信息；validation.json 保存产物检查。

## 原来 3 个场景的同样本复核

保留原来 12 张图片，使用相同 checkpoint；12 张参考 mask 逐项一致。以下统计使用其中 11 张有效图片：

| 条件 | 上一版 | 扩展版中的原 3 场景 |
|---|---:|---:|
| 原始语言 | 84.22% | 84.22% |
| 空语言 | 27.47% | 27.47% |
| 错误语言（类型不同） | 78.49% | 59.47% |

原始/空语言结果基本保持一致；错误文字从其他驾驶场景描述换成离谱无关描述后，原目标响应降低。这比混合样本直接比较两个版本更能说明错误输入类型的影响，但仍不等于逐对象语义匹配能力的证明。

上一版 3 场景完整产物已保存在 [previous_3scenes](../steer_stage2_tokens_test_20261119_previous_3scenes/index.html)，其中旧 PCA 图只在归档中保留。当前图集没有 PCA。

复现可运行工程脚本（生成新的输出目录）：

```bash
bash /mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main/scripts/viz/run_steer_tokens_test.sh --scene-count 30 --seed 2
```

## 最新补充：同图颜色词对照

[白车/红车与白车/黑车四相机对照](COLOR_PROMPT_PAIRS.md)：包含 PNG、7 页 PDF、原始特征、运行记录及复现脚本。两例均未显示明确的按颜色切换目标，本次未执行下游规划评估。
