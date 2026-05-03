# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research project for UAV aerial vehicle detection, modifying ultralytics v8.4.45. The paper proposes three innovations on top of YOLO11s:

1. **PVRP** (P2-Proxy Guided Fine-Grained Vehicle Recovery Pyramid) — P2-level features as a proxy detail branch fused into P3 via semantic gating, with NDA for neighbor decoupling
2. **SNAA** (Scale-Neighbor Aware Attraction Loss) — custom loss replacing IoU similarity with scale-normalized attraction + neighbor repulsion
3. **Lite** (Selective Slimming) — C3k2Lite using depthwise separable bottlenecks for P4/P5 branches

Dataset: UAVDT (car/truck/bus), 3 classes. Training runs on a Linux machine at `/home/ssssss/1yolo/`.

## Commands

### Run all ablation experiments
```bash
python auto_train_all.py
```

### Run a single experiment
```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml").load("yolo11s.pt")
model.train(data="UAVDT.yaml", imgsz=640, epochs=150, batch=16, snaa=True, name="my_exp")
```

### Profile all model variants (Params, GFLOPs, FPS)
```bash
python tools/profile_models.py
```

### Collect and summarize experiment results (outputs LaTeX table)
```bash
python tools/collect_results.py /home/ssssss/1yolo/Ablation_Results
```

### Run tests (stock ultralytics suite, no custom tests for PVRP/SNAA)
```bash
pytest tests/
```

## Architecture

### Modified ultralytics files (6 files)

| File | Change |
|------|--------|
| `ultralytics/nn/modules/block.py` | 5 custom modules: DSBottleneck, C3k2Lite, P2Proxy, ProxyFuse, NDA |
| `ultralytics/nn/modules/__init__.py` | Exports the 5 new modules |
| `ultralytics/nn/tasks.py` | Registers modules in `parse_model()`; ProxyFuse has custom multi-input channel extraction |
| `ultralytics/utils/loss.py` | SNAALoss class + integration into v8DetectionLoss |
| `ultralytics/models/yolo/detect/train.py` | Conditionally adds snaa_loss to loss_names |
| `ultralytics/cfg/default.yaml` | 8 SNAA hyperparameters (snaa, snaa_weight, snaa_kappa, snaa_tau, snaa_beta, snaa_alpha_max, snaa_gamma, snaa_margin) |

### Custom modules (all in `ultralytics/nn/modules/block.py`)

- **P2Proxy** (~line 2080): CSP-style branch processing P2 features. Optional `downsample=True` for stride-2 to P3 resolution.
- **ProxyFuse** (~line 2119): Fuses P2 proxy into P3 with anti-aliasing downsampling + semantic gate (AdaptiveAvgPool → 1x1 → sigmoid).
- **NDA** (~line 2184): Local contrast enhancement `x - avg_pool(x)` + 1x1 conv.
- **C3k2Lite** (~line 2056): Lightweight C3k2 using DSBottleneck instead of standard Bottleneck.
- **DSBottleneck** (~line 2020): Depthwise separable bottleneck (depthwise + pointwise conv).

### Custom loss (in `ultralytics/utils/loss.py`)

- **SNAALoss** (~line 109): Scale-normalized center deviation + neighbor repulsion term. Activated by `snaa=True`. Weight controlled by `snaa_weight` (default 0.2), independent from box/cls/dfl weights.

### Model YAML configs (`ultralytics/cfg/models/11/`)

8 configs for ablation: `yolo11s.yaml` (baseline), `yolo11s-pvrp.yaml` (full PVRP), `yolo11s-pvrp-lite.yaml` (PVRP+Lite), plus 5 ablation variants (`-s1`, `-s3`, `-s12`, `-s13`, `-lite-s4`).

### Experiment matrix (`auto_train_all.py`)

16 experiments across 5 paper tables. Exp01-05 are active; Exp06-15 are commented out. Each experiment specifies: model YAML, SNAA toggle, batch size. SNAA hyperparameters can be overridden per-experiment.

### Tools

- `tools/profile_models.py` — Params/GFLOPs/FPS measurement via thop
- `tools/collect_results.py` — Parses results.csv, prints summary + LaTeX table

## Key Conventions

- Pretrained weights (`yolo11s.pt`) are loaded via `.load("yolo11s.pt")` after constructing from YAML
- The `UAVDT.yaml` dataset path points to a Linux machine; update for local development
- `auto_train_all.py` has `epochs=1` as a placeholder; real training uses 150 epochs
- All model configs use `nc: 80` (COCO default) which gets overridden to 3 at training time
