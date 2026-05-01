# PVRP + SNAA + Lite 论文工程说明

## 1. 项目概述

- **论文方向**: 基于深度学习的无人机航拍车辆识别方法设计
- **基础模型**: YOLO11s
- **主数据集**: UAVDT (3类: car, truck, bus)
- **辅助数据集**: VisDrone-DET 车辆子集

## 2. 创新点对应代码

| 论文创新点 | 代码模块 | 文件位置 |
|-----------|---------|---------|
| P2 代理细节支路 | `P2Proxy` | `ultralytics/nn/modules/block.py` |
| 抗混叠语义回灌融合 | `ProxyFuse` | `ultralytics/nn/modules/block.py` |
| 近邻车辆解耦头适配器 | `NDA` | `ultralytics/nn/modules/block.py` |
| 高层路径选择性瘦身 | `C3k2Lite` / `DSBottleneck` | `ultralytics/nn/modules/block.py` |
| 尺度-近邻感知吸引损失 | `SNAALoss` | `ultralytics/utils/loss.py` |

## 3. 修改文件清单

| 文件 | 改动类型 | 改动内容 |
|------|----------|----------|
| `ultralytics/nn/modules/block.py` | 新增 | DSBottleneck, C3k2Lite, P2Proxy, ProxyFuse, NDA |
| `ultralytics/nn/modules/__init__.py` | 修改 | 导出新模块 |
| `ultralytics/nn/tasks.py` | 修改 | 注册新模块到 parse_model, 修复 _predict_once |
| `ultralytics/utils/loss.py` | 新增+修改 | SNAALoss 类, v8DetectionLoss 集成 SNAA |
| `ultralytics/cfg/default.yaml` | 修改 | 新增 7 个 SNAA 超参 |
| `ultralytics/models/yolo/detect/train.py` | 修改 | loss_names 动态添加 snaa_loss |
| `ultralytics/cfg/models/11/yolo11-pvrp.yaml` | 新增 | 完整 PVRP 架构 |
| `ultralytics/cfg/models/11/yolo11-pvrp-s1.yaml` | 新增 | P2Proxy 消融 |
| `ultralytics/cfg/models/11/yolo11-pvrp-s3.yaml` | 新增 | NDA 消融 |
| `ultralytics/cfg/models/11/yolo11-pvrp-s12.yaml` | 新增 | P2Proxy+ProxyFuse 消融 |
| `ultralytics/cfg/models/11/yolo11-pvrp-s13.yaml` | 新增 | P2Proxy+NDA 消融 |
| `ultralytics/cfg/models/11/yolo11-pvrp-lite.yaml` | 新增 | PVRP+Lite 完整版 |
| `ultralytics/cfg/models/11/yolo11-pvrp-lite-s4.yaml` | 新增 | 仅瘦身消融 |
| `auto_train_all.py` | 修改 | 完整实验矩阵 (16组) |
| `tools/profile_models.py` | 新增 | 参数/FLOPs/FPS 统计工具 |
| `tools/collect_results.py` | 新增 | 实验结果汇总工具 |

## 4. 配置文件与实验对应

| 配置文件 | 实验 | 说明 |
|---------|------|------|
| `yolo11.yaml` | Exp01 Baseline | 标准 YOLO11s |
| `yolo11-pvrp.yaml` | Exp02/03/12/13 | PVRP 完整 (可选 SNAA) |
| `yolo11-pvrp-s1.yaml` | Exp06 | 仅 P2Proxy |
| `yolo11-pvrp-s3.yaml` | Exp07 | 仅 NDA |
| `yolo11-pvrp-s12.yaml` | Exp08 | P2Proxy+ProxyFuse |
| `yolo11-pvrp-s13.yaml` | Exp09 | P2Proxy+NDA |
| `yolo11-pvrp-lite.yaml` | Exp04/05 | PVRP+Lite (可选 SNAA) |
| `yolo11-pvrp-lite-s4.yaml` | Exp10 | 仅瘦身 |

## 5. 参数量与性能汇总

| 模型 | Params | GFLOPs | FPS (GPU) |
|------|--------|--------|-----------|
| Baseline | 9.43M | 10.8 | ~78 |
| PVRP Full | 12.40M | 34.8 | ~53 |
| PVRP-Lite | 11.49M | 34.1 | ~41 |
| PVRP-S1 | 14.82M | 32.1 | ~62 |
| PVRP-S3 | 11.95M | 16.0 | ~51 |
| PVRP-S12 | 10.20M | 20.3 | ~49 |
| PVRP-S13 | 12.45M | 22.9 | ~45 |
| Lite-S4 | 11.77M | 19.2 | ~55 |

## 6. 运行命令

### Baseline
```bash
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml').load('yolo11s.pt')
model.train(data='UAVDT.yaml', imgsz=960, epochs=150, batch=16, optimizer='AdamW', lr0=0.001, cos_lr=True, patience=30, name='Exp01_Baseline')
"
```

### PVRP Full (无 SNAA)
```bash
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/11/yolo11-pvrp.yaml').load('yolo11s.pt')
model.train(data='UAVDT.yaml', imgsz=960, epochs=150, batch=16, optimizer='AdamW', lr0=0.001, cos_lr=True, patience=30, snaa=False, name='Exp02_PVRP_Main')
"
```

### PVRP Full + SNAA
```bash
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/11/yolo11-pvrp.yaml').load('yolo11s.pt')
model.train(data='UAVDT.yaml', imgsz=960, epochs=150, batch=16, optimizer='AdamW', lr0=0.001, cos_lr=True, patience=30, snaa=True, name='Exp03_PVRP_SNAA')
"
```

### PVRP-Lite + SNAA (完整模型)
```bash
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/11/yolo11-pvrp-lite.yaml').load('yolo11s.pt')
model.train(data='UAVDT.yaml', imgsz=960, epochs=150, batch=16, optimizer='AdamW', lr0=0.001, cos_lr=True, patience=30, snaa=True, name='Exp05_PVRP_Lite_SNAA_Full')
"
```

### 批量运行所有实验
```bash
conda activate yolo11
cd d:/1yolo/yolo11s2
python auto_train_all.py
```

## 7. 推荐实验顺序

### 第一轮 (快速判断方案有效性，优先跑)
1. **Exp01 Baseline** — 建立基线
2. **Exp02 PVRP Full** — 验证主创新点
3. **Exp03 PVRP+SNAA** — 验证损失函数

### 第二轮 (消融实验)
4. **Exp06 S1 (P2Proxy)** — 验证 P2 代理支路
5. **Exp07 S3 (NDA)** — 验证近邻解耦
6. **Exp08 S12 (ProxyFuse)** — 验证语义回灌

### 第三轮 (副创新与完整消融)
7. **Exp10 Lite-S4** — 验证瘦身效果
8. **Exp04 PVRP-Lite** — 验证瘦身+PVRP 联合
9. **Exp05 PVRP-Lite+SNAA** — 完整模型

### 第四轮 (SNAA 内部消融)
10. **Exp12 SNAA ScaleOnly** — 仅尺度项
11. **Exp13 SNAA NoRepulsion** — 无排斥项

## 8. 快速判断模块价值

- 跑完 Exp01 和 Exp02 后，如果 mAP50:95 提升 > 0.5，说明 PVRP 有效
- 跑完 Exp02 和 Exp03 后，如果 SNAA 再提升 > 0.3，说明损失函数有效
- 如果某个消融实验 mAP 下降 > 1.0，说明该子模块可能负优化，优先删减

## 9. 论文结果表建议

### 主表 (Table 1): 核心对比
Exp01 vs Exp02 vs Exp03 vs Exp04 vs Exp05

### 消融表 (Table 2): PVRP 子模块
Exp01 vs Exp06 vs Exp07 vs Exp08 vs Exp09 vs Exp02

### 消融表 (Table 3): Lite 副创新
Exp01 vs Exp10 vs Exp04

### 消融表 (Table 4): SNAA 内部项
Exp02 vs Exp12 vs Exp13 vs Exp03

### 应记录指标
mAP50, mAP50:95, Precision, Recall, APs (小目标), Params, GFLOPs, FPS, Latency
