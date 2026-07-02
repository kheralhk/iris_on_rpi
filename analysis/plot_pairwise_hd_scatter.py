from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier
from hamming_distance_distribution import (
    compute_pairwise_scores_iriscode,
    evaluate_scores,
    precompute_codes,
    summarize_label_pairs,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pairwise_hd_scatter"


def order_pairwise_points(pairwise, labels, sort_by):
    scores = pairwise["scores"]
    same_class = pairwise["same_class"]

    if sort_by == "pair":
        return np.arange(scores.size)
    if sort_by == "score":
        return np.argsort(scores, kind="stable")
    if sort_by == "category":
        return np.lexsort((scores, ~same_class))
    if sort_by == "label":
        label1 = labels[pairwise["idx1"]].astype(str)
        label2 = labels[pairwise["idx2"]].astype(str)
        return np.lexsort((label2, label1, ~same_class))

    raise ValueError(f"Unsupported sort mode: {sort_by}")


def save_pairwise_csv(csv_path, pairwise, labels, image_names, order):
    output = Path(csv_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    idx1 = pairwise["idx1"]
    idx2 = pairwise["idx2"]
    scores = pairwise["scores"]
    same_class = pairwise["same_class"]
    best_offset = pairwise["best_offset"]
    direction = pairwise["direction"]

    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "comparison_index",
                "original_pair_index",
                "category",
                "hamming_distance",
                "idx1",
                "idx2",
                "label1",
                "label2",
                "image_name1",
                "image_name2",
                "best_offset",
                "direction",
            ]
        )
        for comparison_index, original_index in enumerate(order):
            left = idx1[original_index]
            right = idx2[original_index]
            writer.writerow(
                [
                    comparison_index,
                    int(original_index),
                    "mated" if same_class[original_index] else "non_mated",
                    float(scores[original_index]),
                    int(left),
                    int(right),
                    labels[left],
                    labels[right],
                    image_names[left],
                    image_names[right],
                    int(best_offset[original_index]),
                    int(direction[original_index]),
                ]
            )

    return output


def save_metadata_json(json_path, metadata):
    output = Path(json_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    return output


def plot_hd_scatter(pairwise, evaluation, order, figure_path, title):
    scores = pairwise["scores"][order]
    same_class = pairwise["same_class"][order]
    x = np.arange(scores.size)

    mated = same_class
    non_mated = ~same_class

    figure, axis = plt.subplots(figsize=(14, 7))
    axis.plot(
        x[mated],
        scores[mated],
        color="#1f77b4",
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )
    axis.scatter(
        x[mated],
        scores[mated],
        color="#1f77b4",
        s=14,
        alpha=0.8,
        label=f"Mated ({int(mated.sum())})",
        edgecolors="none",
        zorder=3,
    )
    axis.plot(
        x[non_mated],
        scores[non_mated],
        color="#d62728",
        linewidth=0.8,
        alpha=0.25,
        zorder=1,
    )
    axis.scatter(
        x[non_mated],
        scores[non_mated],
        color="#d62728",
        s=10,
        alpha=0.45,
        label=f"Non-mated ({int(non_mated.sum())})",
        edgecolors="none",
        zorder=2,
    )

    hd_threshold = -float(evaluation["eer_threshold"])
    axis.axhline(
        hd_threshold,
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label=f"EER threshold HD={hd_threshold:.4f}",
    )

    axis.set_title(title)
    axis.set_xlabel("Comparison")
    axis.set_ylabel("Hamming distance")
    axis.set_ylim(0.0, min(1.0, max(0.6, float(np.nanmax(scores)) + 0.05)))
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.legend(loc="best")
    figure.tight_layout()

    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def output_paths(output_name):
    stem = output_name or "pairwise_hd_scatter"
    if Path(stem).suffix:
        stem = Path(stem).stem

    return {
        "figure": DEFAULT_OUTPUT_DIR / f"{stem}.png",
        "csv": DEFAULT_OUTPUT_DIR / f"{stem}.csv",
        "metadata": DEFAULT_OUTPUT_DIR / f"{stem}.json",
    }


def main():
    parser = ArgumentParser(
        description="Plot pairwise Hamming distances with mated and non-mated comparisons on one scatter chart."
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_format",
        default="auto",
        choices=DATASET_CHOICES,
        help="Dataset folder layout to load.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset directory. If omitted, a known default path is used.",
    )
    parser.add_argument(
        "--filters",
        dest="filters",
        default=None,
        help="Optional Python filters file containing a 'filters' list. Defaults to project filters.py.",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=71,
        help="Number of horizontal offsets to evaluate around zero.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Randomly sample at most this many images before pairwise evaluation.",
    )
    parser.add_argument(
        "--max-id",
        dest="max_identities",
        type=int,
        default=None,
        help="Randomly sample at most this many identities.",
    )
    parser.add_argument(
        "--max-img-per-id",
        dest="max_images_per_identity",
        type=int,
        default=20,
        help="Randomly sample at most this many images per identity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic subset sampling.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("pair", "score", "category", "label"),
        default="pair",
        help="How to arrange comparisons along the x axis.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base output name for PNG/CSV/JSON files under analysis/output/pairwise_hd_scatter.",
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    dataset_name = dataset_output_slug(dataset_format)
    paths = output_paths(args.output_name or f"{dataset_name}_hd_scatter")

    print(f"Using dataset format: {dataset_format}")
    print(f"Dataset name: {dataset_name}")
    print(f"Using dataset path: {dataset_path}")

    images, labels, image_names = load_dataset(dataset_path, dataset_format)
    images, labels, image_names = sample_dataset(
        images,
        labels,
        image_names,
        max_samples=args.max_samples,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        seed=args.seed,
    )

    pre_summary = summarize_label_pairs(labels)
    print(f"Samples: {pre_summary['sample_count']}")
    print(f"Classes: {pre_summary['class_count']}")
    print(f"Mated pairs: {pre_summary['mated_pairs']}")
    print(f"Non-mated pairs: {pre_summary['non_mated_pairs']}")
    if pre_summary["mated_pairs"] == 0 or pre_summary["non_mated_pairs"] == 0:
        raise ValueError(
            "The sampled subset does not contain both mated and non-mated pairs. "
            "Use a larger subset and keep --max-img-per-id at 2 or more."
        )

    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")

    classifier = IrisClassifier(selected_filters)
    (
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        labels,
        image_names,
        skipped,
    ) = precompute_codes(images, labels, image_names, classifier, args.rotation)

    if skipped:
        print(f"Skipped {len(skipped)} images due to segmentation failure.")
        for skipped_index, skipped_name, reason in skipped[:5]:
            print(f"  skipped[{skipped_index}] {skipped_name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... {len(skipped) - 5} more skipped images")

    summary = summarize_label_pairs(labels)
    print(f"Usable samples: {summary['sample_count']}")
    print(f"Usable mated pairs: {summary['mated_pairs']}")
    print(f"Usable non-mated pairs: {summary['non_mated_pairs']}")
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError("After segmentation, the subset no longer contains both pair types.")

    pairwise = compute_pairwise_scores_iriscode(
        labels,
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
    )
    evaluation = evaluate_scores(pairwise["same_class"], pairwise["scores"])
    order = order_pairwise_points(pairwise, labels, args.sort_by)

    figure_path = plot_hd_scatter(
        pairwise,
        evaluation,
        order,
        paths["figure"],
        f"{dataset_format} pairwise Hamming distances",
    )
    csv_path = save_pairwise_csv(paths["csv"], pairwise, labels, image_names, order)

    metadata = {
        "dataset": dataset_format,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "seg_path": os.environ.get("SEG_PATH"),
        "filters": filters_source,
        "filter_count": len(selected_filters),
        "rotation": args.rotation,
        "sort_by": args.sort_by,
        "max_samples": args.max_samples,
        "max_identities": args.max_identities,
        "max_images_per_identity": args.max_images_per_identity,
        "seed": args.seed,
        "skipped_images": len(skipped),
        "sample_count": summary["sample_count"],
        "class_count": summary["class_count"],
        "mated_pairs": summary["mated_pairs"],
        "non_mated_pairs": summary["non_mated_pairs"],
        "eer": float(evaluation["eer"]),
        "eer_hd_threshold": -float(evaluation["eer_threshold"]),
        "roc_auc": float(evaluation["roc_auc"]),
    }
    metadata_path = save_metadata_json(paths["metadata"], metadata)

    print(f"EER: {metadata['eer']:.6f}")
    print(f"EER HD threshold: {metadata['eer_hd_threshold']:.6f}")
    print(f"ROC AUC: {metadata['roc_auc']:.6f}")
    print(f"Saved scatter plot to {figure_path}")
    print(f"Saved pair rows to {csv_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
