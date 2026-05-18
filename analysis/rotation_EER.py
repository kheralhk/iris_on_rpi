from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loaders import DATASET_CHOICES, dataset_output_slug, load_dataset, resolve_dataset, sample_dataset
from filters import filters
from iris import IrisClassifier, get_iris_band, hamming_distances
from pairwise_iris_analysis import evaluate_scores, summarize_label_pairs


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rotation_EER"


def add_figure_metadata(fig, metadata):
    if not metadata:
        return
    text = " | ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    if text:
        fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, family="monospace", wrap=True)


def shift_band_vertically(iris_band, iris_mask, offset):
    offset = int(offset)
    if offset == 0:
        return iris_band.copy(), iris_mask.copy()

    shifted_band = np.zeros_like(iris_band)
    shifted_mask = np.zeros_like(iris_mask)
    if offset > 0:
        shifted_band[offset:, :] = iris_band[:-offset, :]
        shifted_mask[offset:, :] = iris_mask[:-offset, :]
    else:
        offset = -offset
        shifted_band[:-offset, :] = iris_band[offset:, :]
        shifted_mask[:-offset, :] = iris_mask[offset:, :]
    return shifted_band, shifted_mask


def segment_samples(images, image_names):
    segmented = []
    sample_count = len(images)
    for index, image in enumerate(images, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Segmenting images: {index}/{sample_count}")
        iris_band, iris_mask = get_iris_band(image)
        if iris_band is None or iris_mask is None:
            raise RuntimeError(
                f"Segmentation failed for sample index {index - 1}: {image_names[index - 1]}"
            )
        segmented.append((iris_band, iris_mask))
    return segmented


def precompute_horizontal_candidates(segmented_samples, classifier, max_offset):
    band_shape = segmented_samples[0][0].shape
    offsets = np.arange(-max_offset, max_offset + 1, dtype=np.int16)
    order = np.argsort(np.abs(offsets), kind="stable")
    ordered_offsets = offsets[order]
    range_values = np.arange(0, max_offset + 1, dtype=np.int16)
    range_end_indices = np.array(
        [int(np.max(np.flatnonzero(np.abs(ordered_offsets) <= value))) for value in range_values],
        dtype=np.int32,
    )

    base_codes = []
    base_masks = []
    candidate_codes = []
    candidate_masks = []

    sample_count = len(segmented_samples)
    for index, (iris_band, iris_mask) in enumerate(segmented_samples, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Encoding horizontal candidates: {index}/{sample_count}")
        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        codes, masks, _ = classifier.get_iris_codes(iris_band, iris_mask, offsets=offsets)

        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(np.asarray(base_mask, dtype=bool))
        candidate_codes.append(np.asarray(codes[order], dtype=bool))
        candidate_masks.append(np.asarray(masks[order], dtype=bool))

    return {
        "axis": "horizontal",
        "band_shape": tuple(int(v) for v in band_shape),
        "range_values": range_values,
        "range_end_indices": range_end_indices,
        "base_codes": np.stack(base_codes, axis=0),
        "base_masks": np.stack(base_masks, axis=0),
        "candidate_codes": np.stack(candidate_codes, axis=0),
        "candidate_masks": np.stack(candidate_masks, axis=0),
        "ordered_offsets": ordered_offsets,
        "ordered_vertical_offsets": np.zeros_like(ordered_offsets),
    }


def precompute_vertical_candidates(segmented_samples, classifier, horizontal_range, max_vertical_offset):
    band_shape = segmented_samples[0][0].shape
    horizontal_offsets = np.arange(-horizontal_range, horizontal_range + 1, dtype=np.int16)
    vertical_offsets = np.arange(-max_vertical_offset, max_vertical_offset + 1, dtype=np.int16)
    vertical_order = np.argsort(np.abs(vertical_offsets), kind="stable")
    ordered_vertical_offsets = vertical_offsets[vertical_order]
    range_values = np.arange(0, max_vertical_offset + 1, dtype=np.int16)

    ordered_horizontal_offsets = []
    ordered_vertical_offsets_per_candidate = []
    for vertical_offset in ordered_vertical_offsets:
        for horizontal_offset in horizontal_offsets:
            ordered_horizontal_offsets.append(int(horizontal_offset))
            ordered_vertical_offsets_per_candidate.append(int(vertical_offset))
    ordered_horizontal_offsets = np.asarray(ordered_horizontal_offsets, dtype=np.int16)
    ordered_vertical_offsets_per_candidate = np.asarray(
        ordered_vertical_offsets_per_candidate, dtype=np.int16
    )
    range_end_indices = np.array(
        [
            int(np.max(np.flatnonzero(np.abs(ordered_vertical_offsets_per_candidate) <= value)))
            for value in range_values
        ],
        dtype=np.int32,
    )

    base_codes = []
    base_masks = []
    candidate_codes = []
    candidate_masks = []

    sample_count = len(segmented_samples)
    for index, (iris_band, iris_mask) in enumerate(segmented_samples, start=1):
        if index == 1 or index % 100 == 0 or index == sample_count:
            print(f"Encoding vertical candidates: {index}/{sample_count}")
        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(np.asarray(base_mask, dtype=bool))

        image_candidate_codes = []
        image_candidate_masks = []
        for vertical_offset in ordered_vertical_offsets:
            shifted_band, shifted_mask = shift_band_vertically(iris_band, iris_mask, int(vertical_offset))
            codes, masks, _ = classifier.get_iris_codes(
                shifted_band,
                shifted_mask,
                offsets=horizontal_offsets,
            )
            image_candidate_codes.append(np.asarray(codes, dtype=bool))
            image_candidate_masks.append(np.asarray(masks, dtype=bool))

        candidate_codes.append(np.concatenate(image_candidate_codes, axis=0))
        candidate_masks.append(np.concatenate(image_candidate_masks, axis=0))

    return {
        "axis": "vertical",
        "band_shape": tuple(int(v) for v in band_shape),
        "range_values": range_values,
        "range_end_indices": range_end_indices,
        "base_codes": np.stack(base_codes, axis=0),
        "base_masks": np.stack(base_masks, axis=0),
        "candidate_codes": np.stack(candidate_codes, axis=0),
        "candidate_masks": np.stack(candidate_masks, axis=0),
        "ordered_offsets": ordered_horizontal_offsets,
        "ordered_vertical_offsets": ordered_vertical_offsets_per_candidate,
    }


def sweep_eer(labels, precomputed):
    pair_count = len(labels) * (len(labels) - 1) // 2
    score_lists = [[] for _ in precomputed["range_values"]]
    same_class_list = []
    started = time.perf_counter()

    base_codes = precomputed["base_codes"]
    base_masks = precomputed["base_masks"]
    candidate_codes = precomputed["candidate_codes"]
    candidate_masks = precomputed["candidate_masks"]
    range_end_indices = precomputed["range_end_indices"]

    pair_index = 0
    for idx1 in range(len(labels)):
        for idx2 in range(idx1 + 1, len(labels)):
            scores_12 = hamming_distances(
                candidate_codes[idx2],
                base_codes[idx1],
                candidate_masks[idx2],
                base_masks[idx1],
            )
            scores_21 = hamming_distances(
                candidate_codes[idx1],
                base_codes[idx2],
                candidate_masks[idx1],
                base_masks[idx2],
            )
            prefix_best = np.minimum(
                np.minimum.accumulate(scores_12),
                np.minimum.accumulate(scores_21),
            )
            for range_idx, end_index in enumerate(range_end_indices):
                score_lists[range_idx].append(float(prefix_best[end_index]))

            same_class_list.append(bool(labels[idx1] == labels[idx2]))
            pair_index += 1
            if pair_index == 1 or pair_index % 5000 == 0 or pair_index == pair_count:
                elapsed = time.perf_counter() - started
                print(f"Sweeping pairs: {pair_index}/{pair_count} in {elapsed:.1f}s")

    same_class = np.asarray(same_class_list, dtype=bool)
    curve = []
    for range_value, scores in zip(precomputed["range_values"], score_lists):
        evaluation = evaluate_scores(same_class, np.asarray(scores, dtype=np.float32))
        curve.append(
            {
                "offset_range": int(range_value),
                "eer": float(evaluation["eer"]),
                "eer_percent": float(evaluation["eer"] * 100.0),
                "roc_auc": float(evaluation["roc_auc"]),
                "eer_threshold": float(evaluation["eer_threshold"]),
                "band_shape": [int(v) for v in precomputed["band_shape"]],
            }
        )
    return curve


def plot_curve(curve, axis_name, dataset_format, figure_path, metadata=None):
    ranges = np.asarray([item["offset_range"] for item in curve], dtype=np.float64)
    eers = np.asarray([item["eer"] for item in curve], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(9, 6))
    x_values = ranges
    x_label = "Offset Range (pixels)"
    if axis_name == "horizontal":
        title = "Equal Error Rate across Horizontal Offset Ranges"
    else:
        title = "Equal Error Rate across Vertical Offset Ranges"

    axis.plot(x_values, eers, lw=2, color="#1f77b4")
    axis.scatter(x_values, eers, s=18, color="#1f77b4")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Equal Error Rate")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, which="both", alpha=0.3)
    add_figure_metadata(figure, metadata or {})
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    output = Path(figure_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main():
    parser = ArgumentParser(
        description="Sweep EER across horizontal or vertical offset ranges using the current U-Net + Gabor pipeline."
    )
    parser.add_argument(
        "--axis",
        choices=["horizontal", "vertical"],
        required=True,
        help="Which offset family to sweep.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to dataset. If omitted, use configured defaults.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_format",
        default="casia-v3-lamp",
        choices=DATASET_CHOICES,
        help="Dataset layout.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--max-images-per-identity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-offset-range",
        type=int,
        default=40,
        help="Maximum absolute offset range in pixels to test.",
    )
    parser.add_argument(
        "--horizontal-range",
        type=int,
        default=21,
        help="When sweeping vertical offsets, keep horizontal search fixed to this absolute range.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and PNG outputs.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base output name without extension. Defaults to <axis>_<dataset>.",
    )
    args = parser.parse_args()

    if args.max_offset_range < 0:
        raise ValueError("--max-offset-range must be non-negative")
    if args.horizontal_range < 0:
        raise ValueError("--horizontal-range must be non-negative")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    dataset_name = dataset_output_slug(dataset_format)
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
    summary = summarize_label_pairs(labels)
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError("Need both mated and non-mated pairs in the sampled subset.")

    print(f"Filters in use: {len(filters)}")
    classifier = IrisClassifier(filters)
    segmented_samples = segment_samples(images, image_names)
    if args.axis == "horizontal":
        precomputed = precompute_horizontal_candidates(segmented_samples, classifier, args.max_offset_range)
    else:
        precomputed = precompute_vertical_candidates(
            segmented_samples,
            classifier,
            args.horizontal_range,
            args.max_offset_range,
        )

    curve = sweep_eer(labels, precomputed)
    best_point = min(curve, key=lambda item: item["eer"])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.output_name or f"{args.axis}_{dataset_name}"
    json_path = output_dir / f"{base_name}.json"
    figure_path = output_dir / f"{base_name}.png"

    result = {
        "axis": args.axis,
        "dataset_format": dataset_format,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "sampling": {
            "max_samples": args.max_samples,
            "max_identities": args.max_identities,
            "max_images_per_identity": args.max_images_per_identity,
            "seed": int(args.seed),
        },
        "max_offset_range": int(args.max_offset_range),
        "horizontal_range": int(args.horizontal_range),
        "sample_summary": summary,
        "best_point": best_point,
        "curve": curve,
    }
    json_path.write_text(json.dumps(result, indent=2))
    plot_metadata = {
        "dataset": dataset_format,
        "seg_path": os.environ.get("SEG_PATH"),
        "axis": args.axis,
        "max_offset_range": args.max_offset_range,
        "horizontal_range": args.horizontal_range if args.axis == "vertical" else None,
        "samples": summary["sample_count"],
        "classes": summary["class_count"],
        "mated_pairs": summary["mated_pairs"],
        "non_mated_pairs": summary["non_mated_pairs"],
        "max_samples": args.max_samples,
        "max_identities": args.max_identities,
        "max_images_per_identity": args.max_images_per_identity,
        "seed": args.seed,
        "filter_count": len(filters),
    }
    plot_curve(curve, args.axis, dataset_format, figure_path, metadata=plot_metadata)

    print(f"Best {args.axis} offset range: {best_point['offset_range']} px")
    print(f"Best EER: {best_point['eer']:.6f} ({best_point['eer_percent']:.4f}%)")
    print(f"Saved sweep JSON to {json_path}")
    print(f"Saved sweep figure to {figure_path}")


if __name__ == "__main__":
    main()
