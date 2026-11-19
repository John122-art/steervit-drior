# 四相机输入：白车 / 黑车语言对照

场景 `166b654c0db95953`，同时输入 cam_f0、cam_b0、cam_l0、cam_r0，输入形状 `[1,4,3,672,1148]`。仅将 cam_f0 文本分别替换成 `the white car` 和 `the black car`；其他三个相机始终使用各自原始文本。模型处于 eval 模式，随机种子为 2。

使用原 Stage-2 `best-epoch=10-step=8866.ckpt`，509 个 state_dict 项严格加载，无缺失或多余项。本次运行的是训练所用的四相机 `image_backbone.forward` 入口，输出四相机共 64 个 register 特征；可视化提取其中 cam_f0 对应的 16 个。没有执行下游规划或评估轨迹。

## 查看结果

- [综合对比图](comparison.png)：左列白车，右列黑车；上排原图，中排 patch grounding，下排 16 个 register 的平均 attention。
- [白车语言：16 个 register](register_tokens_prompt_1.png)
- [黑车语言：16 个 register](register_tokens_prompt_2.png)
- [7 页 PDF](prompt_comparison.pdf)：含综合图、两组 16-token 图，以及全部同编号 register 的逐项并排对比。
- [网页](index.html)、[完整运行记录与检查](run.json)、[推理日志](inference.log)。

沿用先前方法：patch 热图是最终归一化 patch 特征经过 grounding head 后的空间 softmax；register 热图是最后一个 block 中 12 个 head 平均的 register-to-patch attention，在 patch 范围内重新归一化。两种语言共享色标（99.5 分位截断仅用于显示），单个 register 在两种语言之间也共享色标。没有 PCA。

## 观察与限制

本场景未出现清晰的白车/黑车目标切换：两种语言的 patch 热图都强烈响应中间黑车与画面左边缘，白车语言没有把主要响应转移到右侧白色 SUV。16 个 register 的平均 attention 也相近。

两种语言的 patch 分布总变差距离为 **0.080187**，平均 register attention 分布总变差距离为 **0.052121**。定义为 `0.5 * sum(abs(p_white - p_black))`，范围 0–1，仅表示分布变化量，不是定位准确率。两种语言的 patch 最大值位置均在网格 `(row=27, col=0)`（从 0 开始）。

这些结果表明颜色词影响了输出，但本例没有提供模型能正确按白/黑颜色定位的正面证据。图中不止一辆浅色车辆，而且右侧明显的白色车辆是 SUV，因此单场景、两条短语不足以概括全部语言理解能力。本次没有白车/黑车专属人工目标标注，未复用原始长句的目标区域来计算“颜色定位准确率”。

保存的 `four_camera_features.npz` 包含两种语言条件下全部四相机的 patch、register、neck 输出和 attention；`features.npz` 是 cam_f0 的同一份结果。检查确认四相机特征形状正确、前相机提取一致、数值有限、softmax 与 logits 一致、概率和为 1；其他三个相机在两次运行中的已保存特征与热图完全一致。

## 复现

在工程根目录运行，`--output` 指向一个尚不存在的新目录：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
source /mnt/zhaozc_workspace/envs/drivorpython/bin/activate
OMP_NUM_THREADS=4 python scripts/viz/visualize_steer_prompt_pair.py \
  --source exp/visualizations/steer_stage2_tokens_test_20261119/166b654c0db95953/cam_f0 \
  --checkpoint exp/ke/steer_stage2_capatch_vitb_8gpu_b16_12ep/11.13_05.07/lightning_logs/version_0/checkpoints/best-epoch=10-step=8866.ckpt \
  --output /tmp/steer_white_black_4cam_reproduce \
  --prompts 'the white car' 'the black car'
```
