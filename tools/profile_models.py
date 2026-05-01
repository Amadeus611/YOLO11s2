"""Profile all PVRP model variants: Params, FLOPs, FPS, latency."""
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.nn.tasks import yaml_model_load, DetectionModel


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_fps(model, imgsz=640, warmup=5, runs=20, device="cpu"):
    """Measure inference FPS and latency."""
    model.eval()
    x = torch.randn(1, 3, imgsz, imgsz).to(device)
    model = model.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    # Measure
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps


def get_flops(model, imgsz=640, device="cpu"):
    """Calculate GFLOPs using thop or torch profiler."""
    try:
        from thop import profile
        x = torch.randn(1, 3, imgsz, imgsz).to(device)
        model_dev = model.to(device)
        flops, _ = profile(model_dev, inputs=(x,), verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        pass

    try:
        from ultralytics.utils.torch_utils import get_flops
        return get_flops(model, imgsz=imgsz)
    except Exception:
        pass

    return None


def main():
    configs = [
        ("Baseline",       "ultralytics/cfg/models/11/yolo11.yaml"),
        ("PVRP Full",      "ultralytics/cfg/models/11/yolo11-pvrp.yaml"),
        ("PVRP-S1",        "ultralytics/cfg/models/11/yolo11-pvrp-s1.yaml"),
        ("PVRP-S3",        "ultralytics/cfg/models/11/yolo11-pvrp-s3.yaml"),
        ("PVRP-S12",       "ultralytics/cfg/models/11/yolo11-pvrp-s12.yaml"),
        ("PVRP-S13",       "ultralytics/cfg/models/11/yolo11-pvrp-s13.yaml"),
        ("PVRP-Lite",      "ultralytics/cfg/models/11/yolo11-pvrp-lite.yaml"),
        ("Lite-S4",        "ultralytics/cfg/models/11/yolo11-pvrp-lite-s4.yaml"),
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"{'Model':20s} | {'Params':>10s} | {'GFLOPs':>8s} | {'Latency':>10s} | {'FPS':>8s}")
    print("-" * 75)

    for name, cfg in configs:
        try:
            d = yaml_model_load(cfg)
            d["scale"] = "s"
            d["nc"] = 3
            model = DetectionModel(d, ch=3).to(device)
            model.eval()

            params, _ = count_params(model)
            gflops = get_flops(model, imgsz=640, device=device)
            latency, fps = measure_fps(model, imgsz=640, device=device)

            gflops_str = f"{gflops:.1f}" if gflops else "N/A"
            print(f"{name:20s} | {params/1e6:8.2f}M | {gflops_str:>8s} | {latency:8.1f}ms | {fps:6.1f}")
        except Exception as e:
            print(f"{name:20s} | FAILED: {e}")

    # Print summary table for paper
    print(f"\n\n{'='*75}")
    print("Paper Table Format (copy-paste ready):")
    print(f"{'='*75}")
    print(f"{'Model':20s} & {'Params(M)':>10s} & {'GFLOPs':>8s} & {'FPS':>8s} \\\\")
    print("\\hline")
    for name, cfg in configs:
        try:
            d = yaml_model_load(cfg)
            d["scale"] = "s"
            d["nc"] = 3
            model = DetectionModel(d, ch=3).to(device)
            model.eval()
            params, _ = count_params(model)
            gflops = get_flops(model, imgsz=640, device=device)
            _, fps = measure_fps(model, imgsz=640, device=device)
            gflops_str = f"{gflops:.1f}" if gflops else "N/A"
            print(f"{name:20s} & {params/1e6:8.2f} & {gflops_str:>8s} & {fps:6.1f} \\\\")
        except Exception:
            pass


if __name__ == "__main__":
    main()
