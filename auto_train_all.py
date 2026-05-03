import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deterministic.*")
import torch  # type: ignore
from ultralytics import YOLO


def main():
    # =========================================================
    # PVRP + SNAA + Lite 完整消融实验任务列表
    # ---------------------------------------------------------
    # Table 1: 核心对比 (Exp01-05)
    # Table 2: PVRP 子模块消融 (Exp06-09)
    # Table 3: Lite 副创新消融 (Exp10-11)
    # Table 4: SNAA 内部项消融 (Exp12-13)
    # Table 5: 强基线对比 (Exp14-16)
    # =========================================================
    experiments = [
        # =====================================================
        # Table 1: 核心对比实验
        # =====================================================
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s.yaml",
            "name": "Exp01_Baseline",
            "snaa": False,
            "batch": 32,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
            "name": "Exp02_PVRP_Main",
            "snaa": False,
            "batch": 32,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp.yaml",
            "name": "Exp03_PVRP_SNAA",
            "snaa": True,
            "batch": 32,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml",
            "name": "Exp04_PVRP_Lite",
            "snaa": False,
            "batch": 32,
        },
        {
            "yaml": "ultralytics/cfg/models/11/yolo11s-pvrp-lite.yaml",
            "name": "Exp05_PVRP_Lite_SNAA_Full",
            "snaa": True,
            "batch": 32,
        },

        # =====================================================
        # Table 2: PVRP 主创新子模块消融
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp-s1.yaml",
        #     "name": "Exp06_PVRP_S1_P2Proxy",
        #     "snaa": False,
        #     "batch": 16,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp-s3.yaml",
        #     "name": "Exp07_PVRP_S3_NDA",
        #     "snaa": False,
        #     "batch": 16,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp-s12.yaml",
        #     "name": "Exp08_PVRP_S12_ProxyFuse",
        #     "snaa": False,
        #     "batch": 16,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp-s13.yaml",
        #     "name": "Exp09_PVRP_S13_ProxyNDA",
        #     "snaa": False,
        #     "batch": 16,
        # },

        # # =====================================================
        # # Table 3: Lite 副创新子模块消融
        # # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp-lite-s4.yaml",
        #     "name": "Exp10_Lite_S4_SlimOnly",
        #     "snaa": False,
        #     "batch": 16,
        # },

        # # =====================================================
        # # Table 4: SNAA 内部项消融
        # # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp.yaml",
        #     "name": "Exp12_SNAA_ScaleOnly",
        #     "snaa": True,
        #     "snaa_beta": 0.0,
        #     "batch": 16,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/11/yolo11-pvrp.yaml",
        #     "name": "Exp13_SNAA_NoRepulsion",
        #     "snaa": True,
        #     "snaa_margin": 0.0,
        #     "batch": 16,
        # },

        # =====================================================
        # Table 5: 强基线对比 (YOLO26s / RT-DETR)
        # =====================================================
        # {
        #     "yaml": "ultralytics/cfg/models/26/yolo26s.yaml",
        #     "name": "Exp14_YOLO26s",
        #     "snaa": False,
        #     "batch": 12,
        # },
        # {
        #     "yaml": "ultralytics/cfg/models/rt-detr/rtdetr-l.yaml",
        #     "name": "Exp15_RTDETR_l",
        #     "snaa": False,
        #     "batch": 8,
        # },
    ]

    # =========================================================
    # 循环执行实验
    # =========================================================
    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"  实验 {i + 1}/{len(experiments)}: {exp['name']}")
        print(f"  配置文件: {exp['yaml']}")
        print(f"  SNAA: {exp['snaa']}  |  Batch: {exp['batch']}")
        print(f"{'=' * 60}\n")

        model = YOLO(exp["yaml"]).load("yolo11s.pt")

        common_kwargs = dict(
            # --- 数据集 ---
            data="UAVDT.yaml",
            imgsz=640,
            batch=exp["batch"],
            name=exp["name"],
            project="/home/ssssss/1yolo/Ablation_Results",
            device=0,
            workers=8,
            val=True,
            plots=True,
            save=True,
            amp=True,
            cache=False,

            # --- 优化器 (标准 AdamW) ---
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            cos_lr=True,
            warmup_epochs=3.0,

            # --- 损失权重 (标准值) ---
            box=7.5,
            cls=0.5,
            dfl=1.5,
            cls_pw=0.0,

            # --- SNAA ---
            snaa=exp["snaa"],

            # --- 训练策略 ---
            epochs=150,
            patience=30,

            # --- 数据增强 (航拍适配) ---
            mosaic=1.0,
            close_mosaic=15,
            mixup=0.0,
            copy_paste=0.0,
            degrees=25.0,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            erasing=0.1,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,

            # --- 正则化 ---
            dropout=0.0,
        )

        # 透传 SNAA 内部参数覆盖
        for k in ("snaa_kappa", "snaa_tau", "snaa_beta",
                   "snaa_alpha_max", "snaa_gamma", "snaa_margin"):
            if k in exp:
                common_kwargs[k] = exp[k]

        model.train(**common_kwargs)
        torch.cuda.empty_cache()

    print("\n  所有消融实验已全部执行完毕！")


if __name__ == "__main__":
    main()
