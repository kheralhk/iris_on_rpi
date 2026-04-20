from argparse import ArgumentParser
import json
from pathlib import Path


LOWER_IS_BETTER_METRICS = {
    "eer",
    "mean_seconds",
    "score",
}


def load_json(path):
    path = Path(path).expanduser().resolve()
    return path, json.loads(path.read_text())


def pairwise_by_dataset(results):
    return {item["dataset_format"]: item for item in results.get("pairwise", [])}


def speed_section(results):
    speed = results.get("speed")
    if speed is None:
        return None
    if isinstance(speed, list):
        if not speed:
            return None
        speed = speed[0]
    return {
        "metadata": {key: value for key, value in speed.items() if key != "operations"},
        "operations": {op["name"]: op for op in speed.get("operations", [])},
    }


def relative_percent(old, new):
    if old == 0:
        return None
    return ((new - old) / old) * 100.0


def compare_metric(old, new, metric_name):
    result = {
        "old": old,
        "new": new,
        "delta": new - old,
        "relative_percent": relative_percent(old, new),
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
        }
    return comparison


def compare_speed(old_results, new_results):
    old_speed = speed_section(old_results)
    new_speed = speed_section(new_results)
    if old_speed is None or new_speed is None:
        return {"status": "missing_in_one_result"}

    op_names = sorted(set(old_speed["operations"]) | set(new_speed["operations"]))
    comparison = {
        "dataset_format": {
            "old": old_speed["metadata"].get("dataset_format"),
            "new": new_speed["metadata"].get("dataset_format"),
        },
        "feature_extractor": {
            "old": (
                old_speed["metadata"].get("feature_extractor")
                or old_speed["metadata"].get("mode")
                or old_speed["metadata"].get("segmentation_backend")
                or old_speed["metadata"].get("segmentation")
            ),
            "new": (
                new_speed["metadata"].get("feature_extractor")
                or new_speed["metadata"].get("mode")
                or new_speed["metadata"].get("segmentation_backend")
                or new_speed["metadata"].get("segmentation")
            ),
        },
        "operations": {},
    }

    for op_name in op_names:
        old_op = old_speed["operations"].get(op_name)
        new_op = new_speed["operations"].get(op_name)
        if old_op is None or new_op is None:
            comparison["operations"][op_name] = {"status": "missing_in_one_result"}
            continue
        comparison["operations"][op_name] = {
            "mean_seconds": compare_metric(old_op["mean_seconds"], new_op["mean_seconds"], "mean_seconds"),
        }

    old_functional = old_speed["metadata"].get("functional_summary")
    new_functional = new_speed["metadata"].get("functional_summary")
    if old_functional is not None or new_functional is not None:
        comparison["functional_summary"] = {
            "old": None if old_functional is None else old_functional.get("is_functional_iris_recognizer"),
            "new": None if new_functional is None else new_functional.get("is_functional_iris_recognizer"),
        }
    return comparison


def print_pairwise(comparison):
    print("Pairwise summary")
    for dataset, metrics in comparison.items():
        print(f"- {dataset}")
        if "status" in metrics:
            print(f"  status: {metrics['status']}")
            continue
        for metric_name in ("roc_auc", "eer"):
            metric = metrics[metric_name]
            rel = metric["relative_percent"]
            rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
            print(
                f"  {metric_name}: {metric['old']:.6f} -> {metric['new']:.6f} "
                f"({metric['direction']}, delta {metric['delta']:+.6f}, {rel_text})"
            )


def print_speed(comparison):
    print("\nSpeed summary")
    if "status" in comparison:
        print(f"- status: {comparison['status']}")
        return

    print(
        f"- dataset/pipeline: {comparison['dataset_format']['old']} / {comparison['feature_extractor']['old']} "
        f"-> {comparison['dataset_format']['new']} / {comparison['feature_extractor']['new']}"
    )
    for op_name, metrics in comparison["operations"].items():
        if "status" in metrics:
            print(f"  {op_name}: {metrics['status']}")
            continue
        mean_metric = metrics["mean_seconds"]
        rel = mean_metric["relative_percent"]
        rel_text = "n/a" if rel is None else f"{rel:+.2f}%"
        print(
            f"  {op_name}: {mean_metric['old']:.6f}s -> {mean_metric['new']:.6f}s "
            f"({mean_metric['direction']}, delta {mean_metric['delta']:+.6f}s, {rel_text})"
        )
    functional = comparison.get("functional_summary")
    if functional is not None:
        print(f"  functional recognizer: {functional['old']} -> {functional['new']}")


def main():
    parser = ArgumentParser(description="Compare two pipeline benchmark JSON result files.")
    parser.add_argument("old_results", help="Pipeline JSON to compare from.")
    parser.add_argument("new_results", help="Pipeline JSON to compare to.")
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
