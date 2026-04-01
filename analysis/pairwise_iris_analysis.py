# pairwise_iris_analysis.py

from argparse import ArgumentParser
from itertools import combinations
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loaders import load_dataset, resolve_dataset
from filters import filters
from iris import IrisClassifier, get_iris_band


DEFAULT_DATASET_FORMAT = "auto"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pairwise_iris_analysis_output"


def dataset_output_slug(dataset_format):
    return dataset_format.replace("-", "")


def sample_dataset(
    images,
    labels,
    image_names,
    max_samples=None,
    max_identities=None,
    max_images_per_identity=None,
    seed=0,
):
    if max_samples is None and max_identities is None and max_images_per_identity is None:
        return images, labels, image_names

    rng = random.Random(seed)
    label_to_indices = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    selected_labels = list(label_to_indices.keys())
    if max_identities is not None and max_identities < len(selected_labels):
        selected_labels = rng.sample(selected_labels, max_identities)

    selected_indices = []
    for label in selected_labels:
        indices = list(label_to_indices[label])
        if max_images_per_identity is not None and max_images_per_identity < len(indices):
            indices = rng.sample(indices, max_images_per_identity)
        selected_indices.extend(indices)

    if max_samples is not None and max_samples < len(selected_indices):
        selected_indices = rng.sample(selected_indices, max_samples)

    selected_indices.sort()
    sampled_images = [images[index] for index in selected_indices]
    sampled_labels = labels[selected_indices]
    sampled_image_names = image_names[selected_indices]
    return sampled_images, sampled_labels, sampled_image_names


def precompute_codes(images, image_names, classifier, rotation):
    offsets = np.arange(rotation) - rotation // 2
    sample_count = len(images)

    base_codes = []
    base_masks = []
    rotated_codes = []
    rotated_masks = []

    for index, image in enumerate(images, start=1):
        if index == 1 or index % 25 == 0 or index == sample_count:
            print(f"Precomputing iris codes: {index}/{sample_count}")

        iris_band, iris_mask = get_iris_band(image)
        if iris_band is None or iris_mask is None:
            raise RuntimeError(
                f"Segmentation failed for sample index {index - 1}: {image_names[index - 1]}"
            )

        base_code, base_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)
        base_codes.append(np.asarray(base_code, dtype=bool))
        base_masks.append(np.asarray(base_mask, dtype=bool))

        image_rotated_codes = []
        image_rotated_masks = []
        for offset in offsets:
            code, code_mask, _ = classifier.get_iris_code(iris_band, iris_mask, offset=int(offset))
            image_rotated_codes.append(np.asarray(code, dtype=bool))
            image_rotated_masks.append(np.asarray(code_mask, dtype=bool))

        rotated_codes.append(np.stack(image_rotated_codes, axis=0))
        rotated_masks.append(np.stack(image_rotated_masks, axis=0))

    return (
        np.stack(base_codes, axis=0),
        np.stack(base_masks, axis=0),
        np.stack(rotated_codes, axis=0),
        np.stack(rotated_masks, axis=0),
        offsets,
    )


def best_score_against_rotations(base_code, base_mask, candidate_codes, candidate_masks):
    diff = np.bitwise_xor(candidate_codes, base_code)
    combined_mask = np.bitwise_and(candidate_masks, base_mask)
    valid_bits = np.sum(combined_mask, axis=1)
    mismatch_bits = np.sum(np.bitwise_and(diff, combined_mask), axis=1)

    scores = np.full(candidate_codes.shape[0], 2.0, dtype=np.float64)
    valid_rows = valid_bits > 0
    scores[valid_rows] = mismatch_bits[valid_rows] / valid_bits[valid_rows]

    best_index = int(np.argmin(scores))
    return float(scores[best_index]), best_index


def compute_pairwise_scores(labels, base_codes, base_masks, rotated_codes, rotated_masks, offsets):
    idx1_list = []
    idx2_list = []
    score_list = []
    same_class_list = []
    best_offset_list = []
    direction_list = []

    pair_count = len(labels) * (len(labels) - 1) // 2
    started = time.perf_counter()

    for pair_index, (idx1, idx2) in enumerate(combinations(range(len(labels)), 2), start=1):
        score_12, offset_index_12 = best_score_against_rotations(
            base_codes[idx1],
            base_masks[idx1],
            rotated_codes[idx2],
            rotated_masks[idx2],
        )
        score_21, offset_index_21 = best_score_against_rotations(
            base_codes[idx2],
            base_masks[idx2],
            rotated_codes[idx1],
            rotated_masks[idx1],
        )

        if score_12 <= score_21:
            best_score = score_12
            best_offset = int(offsets[offset_index_12])
            direction = 1
        else:
            best_score = score_21
            best_offset = int(offsets[offset_index_21])
            direction = -1

        idx1_list.append(idx1)
        idx2_list.append(idx2)
        score_list.append(best_score)
        same_class_list.append(labels[idx1] == labels[idx2])
        best_offset_list.append(best_offset)
        direction_list.append(direction)

        if pair_index == 1 or pair_index % 5000 == 0 or pair_index == pair_count:
            elapsed = time.perf_counter() - started
            print(f"Scored pairs: {pair_index}/{pair_count} in {elapsed:.1f}s")

    return {
        "idx1": np.array(idx1_list, dtype=np.int32),
        "idx2": np.array(idx2_list, dtype=np.int32),
        "scores": np.array(score_list, dtype=np.float32),
        "same_class": np.array(same_class_list, dtype=bool),
        "best_offset": np.array(best_offset_list, dtype=np.int16),
        "direction": np.array(direction_list, dtype=np.int8),
    }


def summarize_label_pairs(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    sample_count = int(len(labels))
    class_count = int(len(unique_labels))
    total_pairs = sample_count * (sample_count - 1) // 2
    mated_pairs = int(sum(count * (count - 1) // 2 for count in counts))
    non_mated_pairs = int(total_pairs - mated_pairs)

    return {
        "sample_count": sample_count,
        "class_count": class_count,
        "total_pairs": total_pairs,
        "mated_pairs": mated_pairs,
        "non_mated_pairs": non_mated_pairs,
    }


def evaluate_scores(same_class, scores):
    fpr, tpr, thresholds = roc_curve(same_class, -scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    roc_auc = float(auc(fpr, tpr))

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "eer": eer,
        "eer_threshold": float(thresholds[eer_index]),
        "roc_auc": roc_auc,
    }


def save_results(output_path, pairwise, evaluation, labels, image_names, dataset_path, rotation):
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    idx1 = pairwise["idx1"]
    idx2 = pairwise["idx2"]

    np.savez(
        output,
        idx1=idx1,
        idx2=idx2,
        scores=pairwise["scores"],
        same_class=pairwise["same_class"],
        best_offset=pairwise["best_offset"],
        direction=pairwise["direction"],
        label1=labels[idx1],
        label2=labels[idx2],
        image_name1=image_names[idx1],
        image_name2=image_names[idx2],
        fpr=evaluation["fpr"],
        tpr=evaluation["tpr"],
        thresholds=evaluation["thresholds"],
        eer=np.array(evaluation["eer"]),
        eer_threshold=np.array(evaluation["eer_threshold"]),
        roc_auc=np.array(evaluation["roc_auc"]),
        dataset_path=np.array(str(Path(dataset_path).expanduser().resolve())),
        rotation=np.array(rotation),
    )
    return output


def plot_results(scores, same_class, evaluation, figure_path=None):
    mated_scores = scores[same_class]
    non_mated_scores = scores[~same_class]
    fpr = evaluation["fpr"]
    positive_fpr = fpr[fpr > 0.0]
    min_log_fpr = max(float(positive_fpr.min()) / 10.0, 1e-6) if positive_fpr.size else 1e-6
    plot_fpr = np.maximum(fpr, min_log_fpr)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.kdeplot(mated_scores, ax=axes[0], label="Mated", color="#3b5bff", fill=True, alpha=0.55)
    sns.kdeplot(
        non_mated_scores,
        ax=axes[0],
        label="Non-Mated",
        color="#ff4d4f",
        fill=True,
        alpha=0.55,
    )
    axes[0].set_title("Hamming Distance Distribution")
    axes[0].set_xlabel("Hamming Distance")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0.0, 0.6)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        plot_fpr,
        evaluation["tpr"],
        color="#ff8c00",
        lw=2,
        label=f"ROC (AUC = {evaluation['roc_auc']:.4f}, EER = {evaluation['eer']:.4f})",
    )
    chance_fpr = np.logspace(np.log10(min_log_fpr), 0.0, 200)
    axes[1].plot(chance_fpr, chance_fpr, linestyle="--", color="#6c757d", lw=1)
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate (log scale)")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_xscale("log")
    axes[1].set_xlim(min_log_fpr, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="lower right")

    figure.tight_layout()

    if figure_path:
        output = Path(figure_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight")

    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()


def main():
    parser = ArgumentParser(
        description="Compute pairwise iris comparison scores, save them to .npz, and plot distribution/ROC/EER."
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset directory. If omitted, a known default path is used.",
    )
    parser.add_argument(
        "--dataset-format",
        default=DEFAULT_DATASET_FORMAT,
        choices=["auto", "casia-v1", "casia-v3-interval", "casia-v3-lamp", "casia-v3-twins"],
        help="Dataset folder layout to load",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output .npz file for pairwise scores and metrics. Omit to skip saving scores.",
    )
    parser.add_argument(
        "--figure-output",
        default=None,
        help="Output image path for the distribution/ROC figure. Defaults to fltr_ana_<dataset>.png",
    )
    parser.add_argument(
        "--figure-name",
        default=None,
        help="Output filename for the figure inside the default output directory. Example: my_run.png",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=21,
        help="Number of horizontal offsets to evaluate around zero",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Randomly sample at most this many images before pairwise evaluation.",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=None,
        help="Randomly sample at most this many identities.",
    )
    parser.add_argument(
        "--max-images-per-identity",
        type=int,
        default=None,
        help="Randomly sample at most this many images per identity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic subset sampling.",
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.figure_output and args.figure_name:
        raise ValueError("Use either --figure-output or --figure-name, not both.")

    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    dataset_name = dataset_output_slug(dataset_format)
    figure_output = args.figure_output
    if figure_output is None and args.figure_name is not None:
        figure_name = args.figure_name
        if Path(figure_name).suffix == "":
            figure_name = f"{figure_name}.png"
        figure_output = str(DEFAULT_OUTPUT_DIR / figure_name)
    if figure_output is None:
        figure_output = str(DEFAULT_OUTPUT_DIR / f"fltr_ana_{dataset_name}.png")

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
    summary = summarize_label_pairs(labels)
    print(f"Samples: {summary['sample_count']}")
    print(f"Classes: {summary['class_count']}")
    print(f"Total unordered pairs: {summary['total_pairs']}")
    print(f"Mated pairs: {summary['mated_pairs']}")
    print(f"Non-mated pairs: {summary['non_mated_pairs']}")
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError(
            "The sampled subset does not contain both mated and non-mated pairs. "
            "Use a larger subset and prefer --max-images-per-identity 2 or more."
        )
    classifier = IrisClassifier(filters)

    base_codes, base_masks, rotated_codes, rotated_masks, offsets = precompute_codes(
        images,
        image_names,
        classifier,
        args.rotation,
    )
    pairwise = compute_pairwise_scores(labels, base_codes, base_masks, rotated_codes, rotated_masks, offsets)
    evaluation = evaluate_scores(pairwise["same_class"], pairwise["scores"])

    if args.output:
        output_path = save_results(
            args.output,
            pairwise,
            evaluation,
            labels,
            image_names,
            dataset_path,
            args.rotation,
        )
        print(f"Saved pairwise results to {output_path}")
    print(f"EER: {evaluation['eer']:.6f}")
    print(f"EER threshold: {evaluation['eer_threshold']:.6f}")
    print(f"ROC AUC: {evaluation['roc_auc']:.6f}")

    plot_results(pairwise["scores"], pairwise["same_class"], evaluation, figure_path=figure_output)
    print(f"Saved analysis figure to {Path(figure_output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
