# 颜色/方位提示：沿权重链对比

- 图片：`/mnt/data/navsim/test_sensor_blobs/test/2021.08.30.14.54.34_veh-40_00439_00835/CAM_F0/7af4a2f29e3c5aeb.jpg`（cam_f0）
- 提示：'the white car', 'the red car', 'the car on the left side', 'the white car in front'；另加一列无文本
- 框（手工标注，归一化坐标）：white SUV=(0.241, 0.506, 0.39, 0.695); red car=(0.081, 0.491, 0.198, 0.591)

每一格是该文本下，seg（定位头概率）/ cls（CLS 注意力占比）落在各个框里的比例。

| 行 | 文本 | seg:white SUV | seg:red car | cls:white SUV | cls:red car |
|---|---|---|---|---|---|
| official | (no text) | 0.022 | 0.005 | 0.047 | 0.009 |
| official | the white car | 0.619 | 0.000 | 0.585 | 0.007 |
| official | the red car | 0.001 | 0.399 | 0.038 | 0.583 |
| official | the car on the left side | 0.281 | 0.225 | 0.384 | 0.159 |
| official | the white car in front | 0.627 | 0.000 | 0.601 | 0.006 |
| pure | (no text) | 0.031 | 0.007 | 0.033 | 0.006 |
| pure | the white car | 0.715 | 0.000 | 0.517 | 0.010 |
| pure | the red car | 0.013 | 0.535 | 0.088 | 0.481 |
| pure | the car on the left side | 0.482 | 0.187 | 0.356 | 0.144 |
| pure | the white car in front | 0.725 | 0.000 | 0.557 | 0.008 |
| init | (no text) | 0.072 | 0.017 | 0.024 | 0.003 |
| init | the white car | 0.766 | 0.002 | 0.400 | 0.006 |
| init | the red car | 0.019 | 0.535 | 0.070 | 0.422 |
| init | the car on the left side | 0.213 | 0.381 | 0.303 | 0.096 |
| init | the white car in front | 0.763 | 0.004 | 0.461 | 0.006 |
| stage1 | (no text) | 0.084 | 0.020 | 0.024 | 0.003 |
| stage1 | the white car | 0.599 | 0.017 | 0.588 | 0.037 |
| stage1 | the red car | 0.551 | 0.060 | 0.555 | 0.053 |
| stage1 | the car on the left side | 0.517 | 0.073 | 0.564 | 0.059 |
| stage1 | the white car in front | 0.575 | 0.028 | 0.603 | 0.034 |
| stage2 | (no text) | 0.115 | 0.056 | 0.013 | 0.001 |
| stage2 | the white car | 0.613 | 0.010 | 0.308 | 0.016 |
| stage2 | the red car | 0.588 | 0.019 | 0.289 | 0.016 |
| stage2 | the car on the left side | 0.558 | 0.032 | 0.358 | 0.019 |
| stage2 | the white car in front | 0.575 | 0.019 | 0.335 | 0.012 |

## 换提示后热图变了多少（总变差 TV，0 = 完全相同，1 = 完全不重叠）

| 行 | 文本 A | 文本 B | seg TV | cls TV |
|---|---|---|---|---|
| official | (no text) | the white car | 0.885 | 0.670 |
| official | (no text) | the red car | 0.957 | 0.705 |
| official | (no text) | the car on the left side | 0.875 | 0.646 |
| official | (no text) | the white car in front | 0.898 | 0.692 |
| official | the white car | the red car | 0.984 | 0.735 |
| official | the white car | the car on the left side | 0.527 | 0.325 |
| official | the white car | the white car in front | 0.025 | 0.084 |
| official | the red car | the car on the left side | 0.464 | 0.514 |
| official | the red car | the white car in front | 0.984 | 0.747 |
| official | the car on the left side | the white car in front | 0.530 | 0.315 |
| pure | (no text) | the white car | 0.881 | 0.643 |
| pure | (no text) | the red car | 0.947 | 0.659 |
| pure | (no text) | the car on the left side | 0.861 | 0.657 |
| pure | (no text) | the white car in front | 0.884 | 0.691 |
| pure | the white car | the red car | 0.982 | 0.590 |
| pure | the white car | the car on the left side | 0.336 | 0.315 |
| pure | the white car | the white car in front | 0.025 | 0.098 |
| pure | the red car | the car on the left side | 0.654 | 0.420 |
| pure | the red car | the white car in front | 0.982 | 0.622 |
| pure | the car on the left side | the white car in front | 0.339 | 0.307 |
| init | (no text) | the white car | 0.839 | 0.567 |
| init | (no text) | the red car | 0.911 | 0.620 |
| init | (no text) | the car on the left side | 0.809 | 0.601 |
| init | (no text) | the white car in front | 0.841 | 0.598 |
| init | the white car | the red car | 0.961 | 0.549 |
| init | the white car | the car on the left side | 0.689 | 0.317 |
| init | the white car | the white car in front | 0.030 | 0.110 |
| init | the red car | the car on the left side | 0.302 | 0.422 |
| init | the red car | the white car in front | 0.957 | 0.549 |
| init | the car on the left side | the white car in front | 0.690 | 0.290 |
| stage1 | (no text) | the white car | 0.830 | 0.751 |
| stage1 | (no text) | the red car | 0.813 | 0.741 |
| stage1 | (no text) | the car on the left side | 0.808 | 0.766 |
| stage1 | (no text) | the white car in front | 0.818 | 0.773 |
| stage1 | the white car | the red car | 0.086 | 0.068 |
| stage1 | the white car | the car on the left side | 0.137 | 0.075 |
| stage1 | the white car | the white car in front | 0.038 | 0.047 |
| stage1 | the red car | the car on the left side | 0.058 | 0.061 |
| stage1 | the red car | the white car in front | 0.064 | 0.090 |
| stage1 | the car on the left side | the white car in front | 0.107 | 0.082 |
| stage2 | (no text) | the white car | 0.754 | 0.493 |
| stage2 | (no text) | the red car | 0.735 | 0.491 |
| stage2 | (no text) | the car on the left side | 0.711 | 0.532 |
| stage2 | (no text) | the white car in front | 0.731 | 0.508 |
| stage2 | the white car | the red car | 0.041 | 0.062 |
| stage2 | the white car | the car on the left side | 0.081 | 0.108 |
| stage2 | the white car | the white car in front | 0.046 | 0.068 |
| stage2 | the red car | the car on the left side | 0.054 | 0.110 |
| stage2 | the red car | the white car in front | 0.038 | 0.088 |
| stage2 | the car on the left side | the white car in front | 0.046 | 0.087 |

图：`seg_grid.png`（定位头）、`cls_grid.png`（CLS 注意力，即官方 demo 的热图）。
