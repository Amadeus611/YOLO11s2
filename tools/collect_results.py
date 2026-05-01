"""Collect training results from all experiments into summary tables."""
import os
import json
import csv
from pathlib import Path


def find_experiment_dirs(base_dir):
    """Find all experiment directories under the results base."""
    results = {}
    base = Path(base_dir)
    if not base.exists():
        return results
    for d in sorted(base.iterdir()):
        if d.is_dir():
            args_file = d / "args.yaml"
            results_file = d / "results.csv"
            if args_file.exists():
                results[d.name] = {
                    "dir": d,
                    "args_file": args_file,
                    "results_file": results_file,
                }
    return results


def parse_results_csv(csv_path):
    """Parse YOLO results.csv to get best epoch metrics."""
    if not csv_path.exists():
        return None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    # Find best mAP50-95
    best_map = 0
    best_row = None
    for row in rows:
        for key in row:
            if "map50-95" in key.lower() or "map50-95" in key:
                try:
                    val = float(row[key])
                    if val > best_map:
                        best_map = val
                        best_row = row
                except (ValueError, KeyError):
                    pass

    if best_row is None and rows:
        best_row = rows[-1]

    return best_row


def parse_args_yaml(yaml_path):
    """Parse experiment args."""
    try:
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def collect_all(results_base="/home/ssssss/1yolo/Ablation_Results"):
    """Collect and print summary of all experiments."""
    experiments = find_experiment_dirs(results_base)

    if not experiments:
        print(f"No experiments found at {results_base}")
        print("Update the path in this script to match your results directory.")
        return

    print(f"\n{'='*80}")
    print(f"  Experiment Results Summary")
    print(f"{'='*80}")
    print(f"{'Experiment':30s} | {'mAP50':>8s} | {'mAP50-95':>10s} | {'Precision':>10s} | {'Recall':>8s}")
    print("-" * 80)

    for name, info in experiments.items():
        results = parse_results_csv(info["results_file"])
        if results is None:
            print(f"{name:30s} | {'(no results)':>8s}")
            continue

        # Extract metrics (column names vary by ultralytics version)
        metrics = {}
        for key, val in results.items():
            key_lower = key.strip().lower()
            if "map50-95" in key_lower:
                metrics["mAP50-95"] = val
            elif "map50" in key_lower and "95" not in key_lower:
                metrics["mAP50"] = val
            elif "precision" in key_lower:
                metrics["precision"] = val
            elif "recall" in key_lower:
                metrics["recall"] = val

        mAP50 = metrics.get("mAP50", "N/A")
        mAP5095 = metrics.get("mAP50-95", "N/A")
        prec = metrics.get("precision", "N/A")
        rec = metrics.get("recall", "N/A")

        print(f"{name:30s} | {mAP50:>8s} | {mAP5095:>10s} | {prec:>10s} | {rec:>8s}")

    # Print paper-ready LaTeX table
    print(f"\n\n{'='*80}")
    print("LaTeX Table Format:")
    print(f"{'='*80}")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation Results on UAVDT}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Model & mAP@50 & mAP@50:95 & Precision & Recall \\\\")
    print("\\midrule")

    for name, info in experiments.items():
        results = parse_results_csv(info["results_file"])
        if results is None:
            continue
        metrics = {}
        for key, val in results.items():
            key_lower = key.strip().lower()
            if "map50-95" in key_lower:
                metrics["mAP50-95"] = val
            elif "map50" in key_lower and "95" not in key_lower:
                metrics["mAP50"] = val
            elif "precision" in key_lower:
                metrics["precision"] = val
            elif "recall" in key_lower:
                metrics["recall"] = val

        mAP50 = metrics.get("mAP50", "-")
        mAP5095 = metrics.get("mAP50-95", "-")
        prec = metrics.get("precision", "-")
        rec = metrics.get("recall", "-")
        print(f"{name} & {mAP50} & {mAP5095} & {prec} & {rec} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/ssssss/1yolo/Ablation_Results"
    collect_all(base)
