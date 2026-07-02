from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from benchmark_pipeline import (
    DEFAULT_OUTPUT_DIR as BENCHMARK_OUTPUT_DIR,
    build_myseg_band_getter,
    build_rotation_offsets,
    build_unet_band_getter,
    build_wahet_band_getter,
    compute_iriscode_bit_count,
    format_result,
    load_iris_classifier_class,
    run_pairwise_benchmark,
    segmenter_model_path,
)
from dataset_loaders import DATASET_CHOICES, resolve_dataset
from filter_loader import load_filter_bank


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "benchmark_repeated_subsets"


def mean_std(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "n": int(values.size),
    }


def summarize_dataset_runs(dataset_runs, target_fprs):
    summary = {
        "runs": int(len(dataset_runs)),
        "eer": mean_std([run["eer"] for run in dataset_runs]),
        "eer_hd_threshold": mean_std([run["eer_hd_threshold"] for run in dataset_runs]),
        "roc_auc": mean_std([run["roc_auc"] for run in dataset_runs]),
        "kept_sample_count": mean_std([run["kept_sample_count"] for run in dataset_runs]),
        "skipped_sample_count": mean_std([run["skipped_sample_count"] for run in dataset_runs]),
        "target_fpr": {},
    }

    for target_fpr in target_fprs:
        key = f"{target_fpr:g}"
        target_rows = []
        for run in dataset_runs:
            match = None
            for row in run.get("target_fpr", []):
                if np.isclose(float(row["target_fmr"]), float(target_fpr)):
                    match = row
                    break
            if match is not None:
                target_rows.append(match)

        summary["target_fpr"][key] = {
            "tpr": mean_std([row.get("tpr") for row in target_rows]),
            "fnmr": mean_std([row.get("fnmr") for row in target_rows]),
            "threshold": mean_std([row.get("threshold") for row in target_rows]),
            "actual_fmr": mean_std([row.get("actual_fmr") for row in target_rows]),
        }

    return summary


def summarize_part_runs(dataset_runs):
    part_results = [
        run["rotation_consistency_classifier"]
        for run in dataset_runs
        if run.get("rotation_consistency_classifier") is not None
    ]
    if not part_results:
        return None

    return {
        "runs": int(len(part_results)),
        "eer": mean_std([result["summary"]["eer"] for result in part_results]),
        "eer_hd_threshold": mean_std(
            [result["summary"]["eer_hd_threshold"] for result in part_results]
        ),
        "roc_auc": mean_std([result["summary"]["roc_auc"] for result in part_results]),
    }


def print_dataset_summary(dataset, summary, target_fprs):
    eer = summary["eer"]
    threshold = summary["eer_hd_threshold"]
    print(
        f"{dataset}: EER mean={eer['mean']:.6f} std={eer['std']:.6f} "
        f"(n={eer['n']}); EER HD threshold mean={threshold['mean']:.6f} std={threshold['std']:.6f}"
    )
    for target_fpr in target_fprs:
        key = f"{target_fpr:g}"
        target_summary = summary["target_fpr"][key]
        tpr = target_summary["tpr"]
        threshold = target_summary["threshold"]
        print(
            f"  TAR/TPR @ FAR/FPR={target_fpr * 100:.4f}%: "
            f"mean={tpr['mean']:.6f} std={tpr['std']:.6f}; "
            f"threshold mean={threshold['mean']:.6f} std={threshold['std']:.6f}"
        )


def print_part_summary(summary, parts, aggregation):
    eer = summary["eer"]
    threshold = summary["eer_hd_threshold"]
    roc_auc = summary["roc_auc"]
    print(
        f"  {parts}-part {aggregation} HD: "
        f"EER mean={eer['mean']:.6f} std={eer['std']:.6f}; "
        f"threshold mean={threshold['mean']:.6f} std={threshold['std']:.6f}; "
        f"ROC AUC mean={roc_auc['mean']:.6f} std={roc_auc['std']:.6f}"
    )


def main():
    parser = ArgumentParser(
        description=(
            "Run benchmark_pipeline repeatedly on random dataset subsets and report mean/std metrics."
        )
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        nargs="+",
        required=True,
        choices=[dataset for dataset in DATASET_CHOICES if dataset != "auto"],
        help="Dataset or datasets to evaluate.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of random subset runs.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed. Runs use seed-start, seed-start+1, ...",
    )
    parser.add_argument("--rotation", type=int, default=71)
    parser.add_argument("--rotation-step", type=int, default=1)
    parser.add_argument(
        "--rotation-method",
        choices=["roll", "recompute"],
        default="roll",
        help="Rotation evaluation method. Roll rotation is the default.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=None,
        help="Also evaluate part-split HD using this many iriscode parts.",
    )
    parser.add_argument(
        "--part-aggregation",
        choices=["mean", "median"],
        default="mean",
        help="Combine part HD scores using their mean (default) or median.",
    )
    parser.add_argument("--max-id", dest="max_identities", type=int, required=True)
    parser.add_argument(
        "--max-img-per-id",
        dest="max_images_per_identity",
        type=int,
        default=5,
        help="Randomly sample at most this many images per identity. Default: 5.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        nargs="*",
        default=[0.001, 0.0001],
        help="Target FAR/FPR values to summarize. Use decimal values.",
    )
    parser.add_argument("--threshold", type=float, default=0.335)
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=None,
        help=(
            "Skip comparisons with fewer than this many overlapping valid iriscode bits "
            "after applying the selected rotation."
        ),
    )
    parser.add_argument(
        "--radial-bands",
        type=int,
        default=None,
        help="Divide iriscode bits into this many radial bands for --exclude-radial-bands.",
    )
    parser.add_argument(
        "--exclude-radial-bands",
        type=int,
        nargs="+",
        default=None,
        help="One-based radial band numbers to exclude. Band 1 is closest to the pupil.",
    )
    parser.add_argument("--filters", default=None)
    parser.add_argument("--iris-engine", choices=["current", "legacy"], default="current")
    parser.add_argument(
        "--segmenter",
        choices=["unet", "unet-int8", "myseg", "wahet"],
        default="unet-int8",
    )
    parser.add_argument(
        "--segmenter-model",
        type=Path,
        default=None,
        help=(
            "Override the ONNX model for --segmenter unet or unet-int8. "
            "The unet-int8 default is models/upp_scse_mobilenetv2_int8.onnx."
        ),
    )
    parser.add_argument("--test", choices=["pairwise", "probe"], default="pairwise")
    parser.add_argument(
        "--output-name",
        default="repeated_subset_summary",
        help="Output run name. A .json suffix is added when omitted.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.max_identities < 1:
        raise ValueError("--max-id must be at least 1")
    if args.max_images_per_identity < 1:
        raise ValueError("--max-img-per-id must be at least 1")
    if args.parts is not None and args.parts < 1:
        raise ValueError("--parts must be at least 1")
    if args.test == "probe" and args.max_images_per_identity < 2:
        raise ValueError("--test probe needs --max-img-per-id of at least 2")
    if args.test == "probe" and args.parts is not None:
        raise ValueError("--parts is only supported with --test pairwise")
    if args.rotation_method == "roll" and args.iris_engine != "current":
        raise ValueError("--rotation-method roll is only available with --iris-engine current")
    if args.min_valid_pixels is not None and args.min_valid_pixels < 0:
        raise ValueError("--min-valid-pixels cannot be negative")
    if args.segmenter_model is not None and args.segmenter not in {"unet", "unet-int8"}:
        raise ValueError("--segmenter-model is only valid with --segmenter unet or unet-int8")
    if args.exclude_radial_bands is not None and args.radial_bands is None:
        raise ValueError("--exclude-radial-bands requires --radial-bands")
    if args.radial_bands is not None:
        if args.radial_bands < 2:
            raise ValueError("--radial-bands must be at least 2")
        excluded_bands = set(args.exclude_radial_bands or [])
        invalid_bands = sorted(
            value for value in excluded_bands if value < 1 or value > args.radial_bands
        )
        if invalid_bands:
            raise ValueError(
                f"Excluded radial bands must be between 1 and {args.radial_bands}: "
                f"{invalid_bands}"
            )
        if len(excluded_bands) >= args.radial_bands:
            raise ValueError("At least one radial band must remain included")
    for target_fpr in args.target_fpr:
        if target_fpr < 0.0 or target_fpr > 1.0:
            raise ValueError("--target-fpr values must be between 0 and 1")

    output_name = Path(args.output_name).expanduser()
    if output_name.suffix != ".json":
        output_name = output_name.with_suffix(".json")
    output_path = output_name if output_name.is_absolute() else DEFAULT_OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_model_path = segmenter_model_path(args.segmenter, args.segmenter_model)
    if selected_model_path is not None and not selected_model_path.exists():
        raise FileNotFoundError(f"Segmentation model not found: {selected_model_path}")

    selected_filters, filters_source = load_filter_bank(args.filters)
    iris_classifier_class = load_iris_classifier_class(args.iris_engine)
    iriscode_bits = compute_iriscode_bit_count(selected_filters, iris_classifier_class)
    rotation_offsets = build_rotation_offsets(args.rotation, args.rotation_step)
    band_getter = None
    if args.segmenter == "wahet":
        band_getter = build_wahet_band_getter()
    elif args.segmenter == "myseg":
        band_getter = build_myseg_band_getter()
    elif args.segmenter == "unet-int8" or args.segmenter_model is not None:
        band_getter = build_unet_band_getter(args.segmenter, args.segmenter_model)

    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")
    print(f"Iris engine: {args.iris_engine}")
    print(f"Test protocol: {args.test}")
    print(f"Rotation method: {args.rotation_method}")
    if args.parts is not None:
        print(f"Part-split HD: {args.parts} parts, aggregation={args.part_aggregation}")
    if args.radial_bands is not None:
        print(
            f"Radial bands: {args.radial_bands}, "
            f"excluded={sorted(set(args.exclude_radial_bands or []))}"
        )
    print(f"Total iriscode bits from filters: {iriscode_bits}")
    print(f"Seeds: {args.seed_start}..{args.seed_start + args.runs - 1}")

    results = {
        "datasets": args.datasets,
        "runs": int(args.runs),
        "seeds": [int(args.seed_start + index) for index in range(args.runs)],
        "filters_count": int(len(selected_filters)),
        "filters_source": filters_source,
        "iris_engine": args.iris_engine,
        "iriscode_bits": int(iriscode_bits),
        "segmentation_source": args.segmenter,
        "segmentation_model_path": (
            None if selected_model_path is None else str(selected_model_path)
        ),
        "test_protocol": args.test,
        "max_identities": int(args.max_identities),
        "max_images_per_identity": int(args.max_images_per_identity),
        "rotation": int(args.rotation),
        "rotation_step": int(args.rotation_step),
        "rotation_method": args.rotation_method,
        "rotation_offsets": [int(offset) for offset in rotation_offsets],
        "parts": args.parts,
        "part_aggregation": args.part_aggregation if args.parts is not None else None,
        "target_fpr": [float(value) for value in args.target_fpr],
        "min_valid_pixels": None if args.min_valid_pixels is None else int(args.min_valid_pixels),
        "radial_band_exclusion": {
            "band_count": None if args.radial_bands is None else int(args.radial_bands),
            "excluded_bands": sorted(set(args.exclude_radial_bands or [])),
        },
        "per_run": [],
        "summary": {},
    }

    for dataset in args.datasets:
        dataset_path, dataset_format = resolve_dataset(None, dataset)
        dataset_runs = []
        print(f"Evaluating dataset: {dataset_format}")
        for run_index in range(args.runs):
            seed = args.seed_start + run_index
            print(f"  run {run_index + 1}/{args.runs}, seed={seed}")
            run_result = run_pairwise_benchmark(
                dataset_path,
                dataset_format,
                args.rotation,
                rotation_offsets,
                BENCHMARK_OUTPUT_DIR,
                selected_filters,
                filters_source,
                include_rotation_consistency=args.parts is not None,
                rotation_consistency_parts=args.parts or 5,
                rotation_consistency_eliminate=0,
                rotation_consistency_score="hd",
                rotation_consistency_aggregation=args.part_aggregation,
                radial_band_count=args.radial_bands,
                excluded_radial_bands=args.exclude_radial_bands,
                fixed_threshold=args.threshold,
                target_fprs=args.target_fpr,
                iris_engine=args.iris_engine,
                test_protocol=args.test,
                rotation_method=args.rotation_method,
                min_valid_pixels=args.min_valid_pixels,
                max_identities=args.max_identities,
                max_images_per_identity=args.max_images_per_identity,
                seed=seed,
                band_getter=band_getter,
                segmentation_source=args.segmenter,
                segmentation_model_path=selected_model_path,
            )
            dataset_runs.append(run_result)
            part_result = run_result.get("rotation_consistency_classifier")
            results["per_run"].append(
                {
                    "dataset": dataset_format,
                    "seed": int(seed),
                    "eer": run_result["eer"],
                    "eer_hd_threshold": run_result["eer_hd_threshold"],
                    "roc_auc": run_result["roc_auc"],
                    "kept_sample_count": run_result["kept_sample_count"],
                    "skipped_sample_count": run_result["skipped_sample_count"],
                    "protocol_summary": run_result["protocol_summary"],
                    "target_fpr": run_result["target_fpr"],
                    "parts": part_result,
                }
            )

        dataset_summary = summarize_dataset_runs(dataset_runs, args.target_fpr)
        part_summary = summarize_part_runs(dataset_runs)
        if part_summary is not None:
            dataset_summary["parts"] = part_summary
        results["summary"][dataset_format] = dataset_summary
        print_dataset_summary(dataset_format, dataset_summary, args.target_fpr)
        if part_summary is not None:
            print_part_summary(part_summary, args.parts, args.part_aggregation)

    output_path.write_text(json.dumps(format_result(results), indent=2))
    print(f"Saved repeated subset summary to {output_path}")


if __name__ == "__main__":
    main()
