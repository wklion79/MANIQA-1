import argparse
import csv
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired comparison of base_random and global_fixed5 test predictions."
    )
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--global_csv", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--output", default="crop_comparison.json")
    return parser.parse_args()


def load_predictions(path):
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            rows[row["image_name"]] = {
                "target": float(row["target"]),
                "prediction": float(row["prediction"])
            }
    if not rows:
        raise ValueError("Prediction CSV is empty: {}".format(path))
    return rows


def metrics(target, prediction):
    return {
        "srcc": float(spearmanr(target, prediction).statistic),
        "plcc": float(pearsonr(target, prediction).statistic),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2)))
    }


def percentile_interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def main():
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be at least 1.")

    base = load_predictions(args.base_csv)
    global_fixed = load_predictions(args.global_csv)
    if set(base) != set(global_fixed):
        missing_from_global = sorted(set(base) - set(global_fixed))
        missing_from_base = sorted(set(global_fixed) - set(base))
        raise ValueError(
            "Prediction files use different image sets. "
            "Missing from global: {}; missing from base: {}".format(
                missing_from_global[:5], missing_from_base[:5]
            )
        )

    names = sorted(base)
    target = np.array([base[name]["target"] for name in names])
    target_global = np.array([global_fixed[name]["target"] for name in names])
    if not np.allclose(target, target_global, atol=1e-7):
        raise ValueError("Target scores differ between the two prediction files.")

    pred_base = np.array([base[name]["prediction"] for name in names])
    pred_global = np.array([global_fixed[name]["prediction"] for name in names])
    base_metrics = metrics(target, pred_base)
    global_metrics = metrics(target, pred_global)

    rng = np.random.RandomState(args.seed)
    deltas = {"srcc": [], "plcc": [], "mae": [], "rmse": []}
    valid_samples = 0
    for _ in range(args.bootstrap):
        indices = rng.randint(0, len(names), size=len(names))
        if np.unique(target[indices]).size < 2:
            continue
        sample_base = metrics(target[indices], pred_base[indices])
        sample_global = metrics(target[indices], pred_global[indices])
        if not all(np.isfinite(list(sample_base.values()) + list(sample_global.values()))):
            continue
        valid_samples += 1
        for metric_name in deltas:
            deltas[metric_name].append(sample_global[metric_name] - sample_base[metric_name])

    if valid_samples == 0:
        raise ValueError("No valid bootstrap samples could be calculated.")

    result = {
        "n_images": len(names),
        "bootstrap_samples": valid_samples,
        "base_random": base_metrics,
        "global_fixed5": global_metrics,
        "delta_global_minus_base": {
            name: float(global_metrics[name] - base_metrics[name]) for name in deltas
        },
        "delta_95_ci": {name: percentile_interval(values) for name, values in deltas.items()},
        "interpretation": "Positive SRCC/PLCC deltas favor global_fixed5; negative MAE/RMSE deltas favor global_fixed5."
    }

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
