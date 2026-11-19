# 颜色/方位提示：沿权重链对比

- 图片：`/mnt/data/navsim/test_sensor_blobs/test/2021.09.09.18.29.25_veh-39_00427_00556/CAM_F0/262870c171365b8e.jpg`（cam_f0）
- 提示：'the white car', 'the black car', 'the white SUV on the right', 'the black car in the center'；另加一列无文本
- 框（手工标注，归一化坐标）：black car=(0.49, 0.491, 0.582, 0.65); white SUV=(0.604, 0.473, 0.712, 0.615); dark car left=(0.0, 0.499, 0.048, 0.591)

每一格是该文本下，seg（定位头概率）/ cls（CLS 注意力占比）落在各个框里的比例。

| 行 | 文本 | seg:black car | seg:white SUV | seg:dark car left | cls:black car | cls:white SUV | cls:dark car left |
|---|---|---|---|---|---|---|---|
| official | (no text) | 0.029 | 0.077 | 0.009 | 0.046 | 0.039 | 0.004 |
| official | the white car | 0.044 | 0.747 | 0.003 | 0.049 | 0.493 | 0.008 |
| official | the black car | 0.562 | 0.006 | 0.000 | 0.184 | 0.095 | 0.008 |
| official | the white SUV on the right | 0.024 | 0.783 | 0.001 | 0.024 | 0.528 | 0.031 |
| official | the black car in the center | 0.549 | 0.010 | 0.000 | 0.425 | 0.040 | 0.003 |
| pure | (no text) | 0.026 | 0.026 | 0.005 | 0.029 | 0.024 | 0.002 |
| pure | the white car | 0.076 | 0.559 | 0.001 | 0.060 | 0.370 | 0.007 |
| pure | the black car | 0.606 | 0.023 | 0.039 | 0.217 | 0.094 | 0.046 |
| pure | the white SUV on the right | 0.062 | 0.596 | 0.001 | 0.032 | 0.398 | 0.013 |
| pure | the black car in the center | 0.672 | 0.018 | 0.001 | 0.292 | 0.079 | 0.016 |
| init | (no text) | 0.096 | 0.021 | 0.007 | 0.103 | 0.017 | 0.001 |
| init | the white car | 0.060 | 0.276 | 0.002 | 0.070 | 0.133 | 0.002 |
| init | the black car | 0.622 | 0.003 | 0.009 | 0.383 | 0.055 | 0.012 |
| init | the white SUV on the right | 0.061 | 0.563 | 0.000 | 0.032 | 0.342 | 0.003 |
| init | the black car in the center | 0.671 | 0.002 | 0.000 | 0.399 | 0.030 | 0.004 |
| stage1 | (no text) | 0.090 | 0.023 | 0.010 | 0.103 | 0.017 | 0.001 |
| stage1 | the white car | 0.494 | 0.145 | 0.055 | 0.353 | 0.189 | 0.023 |
| stage1 | the black car | 0.481 | 0.101 | 0.084 | 0.359 | 0.176 | 0.027 |
| stage1 | the white SUV on the right | 0.401 | 0.232 | 0.060 | 0.310 | 0.225 | 0.032 |
| stage1 | the black car in the center | 0.506 | 0.153 | 0.043 | 0.381 | 0.185 | 0.029 |
| stage2 | (no text) | 0.118 | 0.035 | 0.027 | 0.078 | 0.011 | 0.001 |
| stage2 | the white car | 0.403 | 0.083 | 0.119 | 0.237 | 0.079 | 0.010 |
| stage2 | the black car | 0.451 | 0.046 | 0.112 | 0.219 | 0.078 | 0.011 |
| stage2 | the white SUV on the right | 0.428 | 0.151 | 0.071 | 0.258 | 0.097 | 0.008 |
| stage2 | the black car in the center | 0.524 | 0.072 | 0.061 | 0.272 | 0.085 | 0.010 |

## 换提示后热图变了多少（总变差 TV，0 = 完全相同，1 = 完全不重叠）

| 行 | 文本 A | 文本 B | seg TV | cls TV |
|---|---|---|---|---|
| official | (no text) | the white car | 0.823 | 0.580 |
| official | (no text) | the black car | 0.851 | 0.534 |
| official | (no text) | the white SUV on the right | 0.867 | 0.625 |
| official | (no text) | the black car in the center | 0.842 | 0.665 |
| official | the white car | the black car | 0.930 | 0.511 |
| official | the white car | the white SUV on the right | 0.073 | 0.160 |
| official | the white car | the black car in the center | 0.927 | 0.648 |
| official | the black car | the white SUV on the right | 0.959 | 0.537 |
| official | the black car | the black car in the center | 0.044 | 0.300 |
| official | the white SUV on the right | the black car in the center | 0.956 | 0.693 |
| pure | (no text) | the white car | 0.877 | 0.558 |
| pure | (no text) | the black car | 0.850 | 0.540 |
| pure | (no text) | the white SUV on the right | 0.913 | 0.593 |
| pure | (no text) | the black car in the center | 0.869 | 0.630 |
| pure | the white car | the black car | 0.825 | 0.407 |
| pure | the white car | the white SUV on the right | 0.065 | 0.157 |
| pure | the white car | the black car in the center | 0.843 | 0.493 |
| pure | the black car | the white SUV on the right | 0.869 | 0.444 |
| pure | the black car | the black car in the center | 0.147 | 0.205 |
| pure | the white SUV on the right | the black car in the center | 0.873 | 0.510 |
| init | (no text) | the white car | 0.781 | 0.354 |
| init | (no text) | the black car | 0.709 | 0.509 |
| init | (no text) | the white SUV on the right | 0.909 | 0.583 |
| init | (no text) | the black car in the center | 0.725 | 0.526 |
| init | the white car | the black car | 0.864 | 0.433 |
| init | the white car | the white SUV on the right | 0.454 | 0.337 |
| init | the white car | the black car in the center | 0.891 | 0.482 |
| init | the black car | the white SUV on the right | 0.931 | 0.487 |
| init | the black car | the black car in the center | 0.168 | 0.170 |
| init | the white SUV on the right | the black car in the center | 0.933 | 0.523 |
| stage1 | (no text) | the white car | 0.760 | 0.596 |
| stage1 | (no text) | the black car | 0.760 | 0.591 |
| stage1 | (no text) | the white SUV on the right | 0.783 | 0.582 |
| stage1 | (no text) | the black car in the center | 0.762 | 0.617 |
| stage1 | the white car | the black car | 0.102 | 0.061 |
| stage1 | the white car | the white SUV on the right | 0.142 | 0.100 |
| stage1 | the white car | the black car in the center | 0.055 | 0.092 |
| stage1 | the black car | the white SUV on the right | 0.204 | 0.104 |
| stage1 | the black car | the black car in the center | 0.117 | 0.072 |
| stage1 | the white SUV on the right | the black car in the center | 0.160 | 0.106 |
| stage2 | (no text) | the white car | 0.681 | 0.357 |
| stage2 | (no text) | the black car | 0.688 | 0.349 |
| stage2 | (no text) | the white SUV on the right | 0.677 | 0.388 |
| stage2 | (no text) | the black car in the center | 0.663 | 0.399 |
| stage2 | the white car | the black car | 0.080 | 0.061 |
| stage2 | the white car | the white SUV on the right | 0.160 | 0.086 |
| stage2 | the white car | the black car in the center | 0.198 | 0.093 |
| stage2 | the black car | the white SUV on the right | 0.176 | 0.093 |
| stage2 | the black car | the black car in the center | 0.166 | 0.087 |
| stage2 | the white SUV on the right | the black car in the center | 0.142 | 0.072 |

图：`seg_grid.png`（定位头）、`cls_grid.png`（CLS 注意力，即官方 demo 的热图）。
