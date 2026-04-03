from argparse import ArgumentParser
import json
from pathlib import Path


LOWER_IS_BETTER_METRICS = {
    "eer",
    "elapsed_seconds",
    "mean_seconds",
    "median_seconds",
    "min_seconds",
    "max_seconds",
    "score",
}


def load_json(path):
    path = Path(path).expanduser().resolve()
    return path, json.loads(path.read_text())


def pairwise_by_dataset(results):
    return {item["dataset_format"]: item for item in results.get("pairwise", [])}


def speed_by_dataset(results):
    lookup = {}
    for item in results.get("speed", []):
        operations = {op["name"]: op for op in item.get("operations", [])}
        lookup[item["dataset_format"]] = {
            "metadata": {key: value for key, value in item.items() if key != "operations"},
            "operations": operations,
        }
    return lookup


def relative_percent(old, new):
    if old == 0:
        return None
    return ((new - old) / old) * 100.0


def ratio(old, new):
    if old == 0:
        return None
    return new / old


def compare_metric(old, new, metric_name):
    result = {
        "old": old,
        "new": new,
        "delta": new - old,
        "relative_percent": relative_percent(old, new),
        "ratio_new_over_old": ratio(old, new),
    }
    if metric_name in LOWER_IS_BETTER_METRICS:
        result["direction"] = "better" if new < old else "worse" if new > old else "same"
    else:
        result["direction"] = "better" if new > old else "worse" if new < old else "same"
    return result


def compare_pairwise(old_results, new_results):
    old_lookup = pairwise_by_dataset(old_results)
    new_lookup = pairwise_by_dataset(new_results)
    comparison = {}
    for dataset in sorted(set(old_lookup) | set(new_lookup)):
        old_item = old_lookup.get(dataset)
        new_item = new_lookup.get(dataset)
        if old_item is None or new_item is None:
            comparison[dataset] = {"status": "missing_in_one_result"}
            continue
        comparison[dataset] = {
            "roc_auc": compare_metric(old_item["roc_auc"], new_item["roc_auc"], "roc_auc"),
            "eer": compare_metric(old_item["eer"], new_item["eer"], "eer"),
            "elapsed_seconds": compare_metric(old_item["elapsed_seconds"], new_item["elapsed_seconds"], "elapsed_seconds"),
        }
    return comparison


def compare_speed(old_results, new_results):
    old_lookup = speed_by_dataset(old_results)
    new_lookup = speed_by_dataset(new_results)
    comparison = {}
    for dataset in sorted(set(old_lookup) | set(new_lookup)):
        old_item = old_lookup.get(dataset)
        new_item = new_lookup.get(dataset)
        if old_item is None or new_item is None:
            comparison[dataset] = {"status": "missing_in_one_result"}
            continue

        op_names = sorted(set(old_item["operations"]) | set(new_item["operations"]))
        dataset_result = {}
        for op_name in op_names:
            old_op = old_item["operations"].get(op_name)
            new_op = new_item["operations"].get(op_name)
            if old_op is None or new_op is None:
                dataset_result[op_name] = {"status": "missing_in_one_result"}
                continue

            op_result = {
                "mean_seconds": compare_metric(old_op["mean_seconds"], new_op["mean_seconds"], "mean_seconds"),
                "median_seconds": compare_metric(old_op["median_seconds"], new_op["median_seconds"], "median_seconds"),
            }
            if op_name == "compare_identical_image":
                old_score = old_op.get("last_result", [None])[0]
                new_score = new_op.get("last_result", [None])[0]
                if old_score is not None and new_score is not None:
                    op_result["score"] = compare_metric(old_score, new_score, "score")
            dataset_result[op_name] = op_result
        old_functional = old_item["metadata"].get("functional_summary")
        new_functional = new_item["metadata"].get("functional_summary")
        if old_functional is not None or new_functional is not None:
            dataset_result["_functional_summary"] = {
                "old": None if old_functional is None else old_functional.get("is_functional_iris_recognizer"),
                "new": None if new_functional is None else new_functional.get("is_functional_iris_recognizer"),
            }
        comparison[dataset] = dataset_result
    return comparison


def print_pairwise(comparison):
    print("Pairwise summary")
    for dataset, metrics in comparison.items():
        print(f"- {dataset}")
        if "status" in metrics:
            print(f"  status: {metrics['status']}")
            continue
        for metric_name in ("roc_auc", "eer", "elapsed_seconds"):
            metric = metrics[metric_name]
            rel = metric["relative_percent"]
            ratio_value = metric["ratio_new_over_old"]
            rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
            ratio_text = "n/a" if ratio_value is None else f"{ratio_value:.3f}x"
            print(
                f"  {metric_name}: {metric['old']:.6f} -> {metric['new']:.6f} "
                f"({metric['direction']}, delta {metric['delta']:+.6f}, {rel_text}, ratio {ratio_text})"
            )


def print_speed(comparison):
    print("\nSpeed summary")
    for dataset, operations in comparison.items():
        print(f"- {dataset}")
        if "status" in operations:
            print(f"  status: {operations['status']}")
            continue
        for op_name, metrics in operations.items():
            if op_name == "_functional_summary":
                print(
                    f"  functional recognizer: {metrics['old']} -> {metrics['new']}"
                )
                continue
            if "status" in metrics:
                print(f"  {op_name}: {metrics['status']}")
                continue
            mean_metric = metrics["mean_seconds"]
            rel = mean_metric["relative_percent"]
            ratio_value = mean_metric["ratio_new_over_old"]
            rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
            ratio_text = "n/a" if ratio_value is None else f"{ratio_value:.3f}x"
            print(
                f"  {op_name}: {mean_metric['old']:.6f}s -> {mean_metric['new']:.6f}s "
                f"({mean_metric['direction']}, delta {mean_metric['delta']:+.6f}s, {rel_text}, ratio {ratio_text})"
            )
            if "score" in metrics:
                score_metric = metrics["score"]
                rel = score_metric["relative_percent"]
                ratio_value = score_metric["ratio_new_over_old"]
                rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
                ratio_text = "n/a" if ratio_value is None else f"{ratio_value:.3f}x"
                print(
                    f"    identical score: {score_metric['old']:.6f} -> {score_metric['new']:.6f} "
                    f"({score_metric['direction']}, delta {score_metric['delta']:+.6f}, {rel_text}, ratio {ratio_text})"
                )


def main():
    parser = ArgumentParser(description="Compare two pipeline baseline JSON result files.")
    parser.add_argument("old_results", help="Baseline JSON to compare from.")
    parser.add_argument("new_results", help="Baseline JSON to compare to.")
    parser.add_argument("--output", default=None, help="Optional JSON file for the structured comparison.")
    args = parser.parse_args()

    old_path, old_results = load_json(args.old_results)
    new_path, new_results = load_json(args.new_results)

    comparison = {
        "old_results": str(old_path),
        "new_results": str(new_path),
        "pairwise": compare_pairwise(old_results, new_results),
        "speed": compare_speed(old_results, new_results),
    }

    print_pairwise(comparison["pairwise"])
    print_speed(comparison["speed"])

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, indent=2))
        print(f"\nSaved comparison JSON to {output_path}")


if __name__ == "__main__":
    main()
