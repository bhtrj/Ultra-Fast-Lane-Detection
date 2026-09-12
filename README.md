# 车道线检测实验（MindSpore 2.7.2 / Ascend）

华为产学合作课程《第三章 语义分割》的两个云端实验，**开箱即跑**：仓库自带真实 TuSimple sample 数据，
clone 下来按下面 5 步即可完成「训练 → 推理 → 可视化」全流程。

| 实验 | 算法 | 思路 | sample 数据 | 参考结果 |
|---|---|---|---|---|
| 实验一 | **Ultra-Fast-Lane-Detection**（ParsingNet + ResNet18） | 把车道线检测变成**分类**：4 条车道 × 56 行锚点 × 101 个横向格子 | 60 训练 / 20 测试 | 1 轮 acc 0.50，8 轮 acc 0.59 |
| 实验二 | **DeepLabV3**（ResNet50 + ASPP） | **语义分割**：逐像素判断是不是车道线 | 80 训练 / 20 验证（含官方掩码） | 1 轮 mIoU 0.48 |

> 数据规模说明：真实 TuSimple 官方下载渠道已全部失效（S3 404、OpenDataLab 需登录 AK/SK）。
> 本仓库 sample 取自 HuggingFace 镜像 `dhbloo/TuSimple` 的**真实**图像与官方标注（非程序生成），
> 属于官方全量 3626 张的一小部分，用于教学演示与链路验证，**不足以复现论文精度**。

---

## 一、环境要求

华为云 ModelArts Notebook：

- 镜像：`mindspore_2.7.2-cann_8.5.2-py_3.11-euler_2.10.11-aarch64-snt9b`
- 规格：Ascend 910B（snt9b1 即可）
- 数据盘：≥ 20 GB（仓库 31 MB，权重约 230 MB/份）

## 二、5 步跑通

```bash
# 0) 进入 Notebook 终端，每个新终端都要先加载 CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 取代码
git clone <本仓库地址> ch3_lane && cd ch3_lane

# 2) 环境自检（看到 Ascend 张量运算: [[2.0, 2.0], [2.0, 2.0]] 即通过）
python 0_env_check.py

# 3) 实验一（约 3 分钟）
python demo/exp1_ultrafast/step1_data_real.py        # 校验 sample + 生成 train_gt.txt
python demo/exp1_ultrafast/step2_train.py --epochs 8 --batch_size 2
python demo/exp1_ultrafast/step3_infer.py            # 结果图 -> out/exp1/demo_grid.jpg

# 4) 实验二（20 轮约 18 分钟；只想验证链路可先 --epochs 1）
python demo/exp2_deeplabv3/step1_data_real.py        # 校验 sample（仓库自带）
python demo/exp2_deeplabv3/step2_train.py --epochs 20 --batch_size 2
python demo/exp2_deeplabv3/step3_infer.py            # 结果图 -> out/exp2/demo_grid.jpg

# 5) 在 Notebook 单元格里看结果图
```

```python
import cv2, matplotlib.pyplot as plt
img = cv2.imread('out/exp1/demo_grid.jpg')[:, :, ::-1]
plt.figure(figsize=(13, 7)); plt.imshow(img); plt.axis('off')
plt.title('Ultra-Fast   Green = GT   Red = Prediction', fontsize=14); plt.show()
```

也可以一步全跑：`bash run_all.sh`（默认实验一 8 轮 + 实验二 20 轮）。

## 三、目录结构

```
.
├── 0_env_check.py              环境自检
├── run_all.sh                  一键跑两个实验
├── prepare_real_data.py        可选：从 HF 镜像再拉全量真实数据（644 张 / 140MB）
├── code_uf/                    实验一算法源码（已适配 2.7.2）
│   ├── src/（7 个模块，train.py 全部用到）  train.py  convert_tusimple.py  config/
├── code_deeplab/src/           实验二算法源码（已适配 2.7.2）
├── demo/
│   ├── exp1_ultrafast/         step1_data_real.py / step2_train.py / step3_infer.py
│   └── exp2_deeplabv3/         step1_data_real.py / step2_train.py / step3_infer.py
├── data/
│   ├── tusimple_sample/        实验一：train_set(60) + test_set(20) + 官方标注
│   └── laneseg_sample/         实验二：images(100) + masks(100) + train.txt/val.txt
└── out/                        训练权重与结果图（运行时生成）
```

## 四、代码来源

| 内容 | 来源 |
|---|---|
| 实验一算法代码 | `https://github.com/zhouyifeng888/Ultra-Fast-Lane-Detection` |
| 实验二算法代码 | `https://github.com/mindspore-courses/computer_vision`（子目录 `deeplabv3`） |
| 真实数据集 | HuggingFace 镜像 `dhbloo/TuSimple`（原官方 TuSimple benchmark） |
| 训练/推理脚本、数据抽样脚本 | 本仓库新增（原仓库无可直接运行的训练与可视化脚本） |

## 五、MindSpore 2.7.2 适配点（已改好，供教学讲解）

**实验一（4 处）**

| 位置 | 原写法（1.x） | 2.7.2 写法 |
|---|---|---|
| `src/network.py` 第 78、80 行 | `P.ResizeBilinear((h, w))(x)` | `P.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)` |
| `train.py` 第 23 行 | `from mindspore.train.serialization import ...` | `from mindspore import load_checkpoint, load_param_into_net, save_checkpoint` |
| `src/dataset.py` 第 227 行 | `dataset.map(..., column_order=...)` | 去掉该参数，改用 `dataset.project([...])` |
| `src/resnet.py` | `nn.Pad` + `MaxPool2d(pad_mode='valid')` | `nn.MaxPool2d(..., pad_mode='same')`（Ascend FP32 下原写法无对应算子） |

**实验二（2 处）**

| 位置 | 原写法 | 2.7.2 写法 |
|---|---|---|
| `src/deeplab_v3.py` 第 182 行 | `nn.AvgPool2d(size[2])(x)` | `nn.AdaptiveAvgPool2d((1, 1))(x)` |
| `src/deeplab_v3.py` 第 224 行 | `ops.ResizeBilinear(...)` | `ops.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)` |

## 六、实测踩坑

1. **每个终端都要 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`**，否则 `Unsupported device target Ascend`。
2. **实验二不能用 Adam**：从头训练 + batch=2 + BatchNorm 时，Adam（1e-3 / 1e-4）会在第 3 步 loss 变 `nan`。
   实测 **Momentum(lr=1e-3, momentum=0.9)** 稳定收敛，脚本默认即为 Momentum（`--opt adam` 可切换对比）。
3. **实验二真实数据需要较多轮次**：车道线仅占约 2% 像素，1 轮时模型会全预测背景（mIoU ≈ 0.48，图表看不出车道线），
   20 轮左右才能看出形状。演示建议 20 轮，边跑边讲 ASPP 原理。
4. **训练必须用 PyNative 模式**（实验一的 `Total_loss.construct` 内有 `print`，Graph 模式无法编译）。
5. **SSH 跑长任务会断**：ModelArts 上 `nohup`/`setsid` 后台进程会随 SSH 断开被杀，
   长训练请直接在 **Jupyter Terminal / Notebook** 里执行。

## 七、想用更多真实数据

```bash
python prepare_real_data.py        # 从 HF 镜像拉 644 张真实图（140MB，约 3 分钟）到 data/tusimple_full
python demo/exp1_ultrafast/step1_data_real.py --n_train 200 --n_test 50
```
