# mask_occlusion_tests.py

from argparse import ArgumentParser
from itertools import combinations
import csv
import inspect
import json
from pathlib import Path
import random
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import auc, roc_curve
except ImportError:  # pragma: no cover - optional dependency
    auc = None
    roc_curve = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loaders import load_dataset, resolve_dataset
from filters import filters
import iris as iris_module

IrisClassifier = iris_module.IrisClassifier
get_iris_band = iris_module.get_iris_band
hamming_distances = iris_module.hamming_distances
DEFAULT_SEGMENTATION_BACKEND = getattr(iris_module, "DEFAULT_SEGMENTATION_BACKEND", "wahet")
predict_unet_masks = getattr(iris_module, "predict_unet_masks", None)

try:
    from segmentation_geometry import build_valid_source_mask, clean_component_mask
except ImportError:
    build_valid_source_mask = None
    clean_component_mask = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "mask_occlusion_tests"
GET_IRIS_BAND_SUPPORTS_BACKEND = "backend" in inspect.signature(get_iris_band).parameters


def load_image(path):
    image_path = Path(path).expanduser().resolve()
    image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image '{image_path}'.")
    return image_path, image


def resolve_backend(name):
    return (name or DEFAULT_SEGMENTATION_BACKEND).strip().lower()


def segment_with_optional_backend(image, backend=None):
    if backend is None or GET_IRIS_BAND_SUPPORTS_BACKEND:
        return get_iris_band(image, backend=backend) if GET_IRIS_BAND_SUPPORTS_BACKEND else get_iris_band(image)
    if str(backend).strip().lower() != "wahet":
        raise RuntimeError(f"Segmentation backend '{backend}' is not supported by this branch.")
    return get_iris_band(image)


def segment_image(image, backend=None):
    iris_band, iris_mask = segment_with_optional_backend(image, backend=backend)
    if iris_band is None or iris_mask is None:
        raise RuntimeError("Iris segmentation failed.")
    return iris_band, iris_mask


def get_unet_source_debug(image):
    if predict_unet_masks is None or build_valid_source_mask is None or clean_component_mask is None:
        raise RuntimeError("U-Net source overlays require the unet-segmentation branch/runtime.")
    gray, iris_mask, pupil_mask, eyelash_mask = predict_unet_masks(image)
    iris_mask = clean_component_mask(iris_mask)
    pupil_mask = clean_component_mask(pupil_mask)
    eyelash_mask = clean_component_mask(eyelash_mask, kernel_size=3)
    if np.any(eyelash_mask):
        eyelash_mask = cv.dilate(
            eyelash_mask,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
    valid_mask = build_valid_source_mask(
        iris_mask,
        pupil_mask,
        eyelash_mask,
        source_image=gray,
        oversat_threshold=254,
    )
    annulus = ((iris_mask > 0) & ~(pupil_mask > 0))
    valid = valid_mask > 0
    excluded = annulus & ~valid
    return {
        "gray": gray,
        "annulus": annulus.astype(np.uint8) * 255,
        "pupil": (pupil_mask > 0).astype(np.uint8) * 255,
        "excluded": excluded.astype(np.uint8) * 255,
        "valid": valid.astype(np.uint8) * 255,
    }


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
    sampled_names = image_names[selected_indices]
    return sampled_images, sampled_labels, sampled_names


def build_enlarged_mask(mask, kernel_size, iterations):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be a positive odd integer")
    if iterations < 1:
        raise ValueError("--iterations must be at least 1")

    invalid = (mask != 255).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_invalid = cv.dilate(invalid, kernel, iterations=iterations)
    return np.where(dilated_invalid > 0, 0, 255).astype(np.uint8)


def apply_synthetic_occlusion(iris_band, original_mask, enlarged_mask, fill_mode):
    occluded = iris_band.copy()
    newly_masked = (original_mask == 255) & (enlarged_mask != 255)
    if not np.any(newly_masked):
        return occluded

    if fill_mode == "zero":
        fill_value = 0
    elif fill_mode == "white":
        fill_value = 255
    elif fill_mode == "mean":
        valid_pixels = iris_band[original_mask == 255]
        fill_value = int(np.round(np.mean(valid_pixels))) if valid_pixels.size else 0
    else:
        raise ValueError(f"Unsupported fill mode: {fill_mode}")

    occluded[newly_masked] = fill_value
    return occluded


def compare_with_details(classifier, iris_ref, mask_ref, iris_probe, mask_probe, rotation):
    ref_code, ref_code_mask, _ = classifier.get_iris_code(iris_ref, mask_ref)
    ref_code = np.asarray(ref_code, dtype=bool)
    ref_code_mask = np.asarray(ref_code_mask, dtype=bool)

    if rotation is None or rotation <= 1:
        offsets = np.array([0], dtype=np.int64)
        probe_codes, probe_masks, _ = classifier.get_iris_codes(iris_probe, mask_probe, offsets=offsets)
    else:
        offsets = np.arange(rotation, dtype=np.int64) - rotation // 2
        probe_codes, probe_masks, _ = classifier.get_iris_codes(iris_probe, mask_probe, offsets=offsets)

    probe_codes = np.asarray(probe_codes, dtype=bool)
    probe_masks = np.asarray(probe_masks, dtype=bool)

    scores = hamming_distances(probe_codes, ref_code, probe_masks, ref_code_mask)
    overlap = np.sum(np.bitwise_and(probe_masks, ref_code_mask), axis=1)

    best_index = int(np.argmin(scores))
    code_length = int(ref_code.size)
    valid_bits = int(overlap[best_index])
    return {
        "score": float(scores[best_index]),
        "best_offset": int(offsets[best_index]),
        "valid_bits": valid_bits,
        "valid_fraction": float(valid_bits / code_length),
        "code_length": code_length,
    }


def symmetric_compare_with_details(classifier, iris1, mask1, iris2, mask2, rotation):
    forward = compare_with_details(classifier, iris1, mask1, iris2, mask2, rotation)
    backward = compare_with_details(classifier, iris2, mask2, iris1, mask1, rotation)
    if forward["score"] <= backward["score"]:
        result = dict(forward)
        result["direction"] = "1_to_2"
        return result

    result = dict(backward)
    result["direction"] = "2_to_1"
    return result


def evaluate_scores(same_class, scores):
    if roc_curve is None or auc is None:
        return {}
    if np.sum(same_class) == 0 or np.sum(~same_class) == 0:
        return {}

    fpr, tpr, thresholds = roc_curve(same_class, -scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fnr - fpr)))
    return {
        "roc_auc": float(auc(fpr, tpr)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "eer_threshold": float(thresholds[eer_index]),
    }


def summarize_rows(rows):
    same_class = np.array([row["same_class"] for row in rows], dtype=bool)
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    valid_bits = np.array([row["valid_bits"] for row in rows], dtype=np.int32)
    valid_fraction = np.array([row["valid_fraction"] for row in rows], dtype=np.float64)

    summary = {
        "pair_count": int(len(rows)),
        "genuine_count": int(np.sum(same_class)),
        "impostor_count": int(np.sum(~same_class)),
    }

    if np.any(same_class):
        summary.update(
            {
                "genuine_score_mean": float(np.mean(scores[same_class])),
                "genuine_score_std": float(np.std(scores[same_class])),
                "genuine_valid_bits_mean": float(np.mean(valid_bits[same_class])),
                "genuine_valid_fraction_mean": float(np.mean(valid_fraction[same_class])),
            }
        )

    if np.any(~same_class):
        summary.update(
            {
                "impostor_score_mean": float(np.mean(scores[~same_class])),
                "impostor_score_std": float(np.std(scores[~same_class])),
                "impostor_valid_bits_mean": float(np.mean(valid_bits[~same_class])),
                "impostor_valid_fraction_mean": float(np.mean(valid_fraction[~same_class])),
            }
        )

    if np.any(same_class) and np.any(~same_class):
        summary["score_gap"] = float(np.mean(scores[~same_class]) - np.mean(scores[same_class]))
        summary.update(evaluate_scores(same_class, scores))

    return summary


def build_mask_overlay(iris_band, original_mask, enlarged_mask):
    base = cv.cvtColor(iris_band, cv.COLOR_GRAY2RGB).astype(np.float32)
    original_invalid = original_mask != 255
    enlarged_invalid = enlarged_mask != 255
    newly_masked = enlarged_invalid & ~original_invalid

    overlay = base.copy()
    overlay[original_invalid] = 0.2 * overlay[original_invalid] + 0.8 * np.array([255, 0, 0], dtype=np.float32)
    overlay[newly_masked] = 0.2 * overlay[newly_masked] + 0.8 * np.array([255, 255, 0], dtype=np.float32)

    original_contours, _ = cv.findContours(original_invalid.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    enlarged_contours, _ = cv.findContours(enlarged_invalid.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overlay, original_contours, -1, (255, 0, 0), 2)
    cv.drawContours(overlay, enlarged_contours, -1, (255, 255, 0), 2)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_source_mask_overlay(source_debug):
    gray = source_debug["gray"]
    annulus = source_debug["annulus"] > 0
    excluded = source_debug["excluded"] > 0
    pupil = source_debug["pupil"] > 0

    overlay = cv.cvtColor(gray, cv.COLOR_GRAY2RGB).astype(np.float32)
    overlay[annulus] = 0.4 * overlay[annulus] + 0.6 * np.array([0, 255, 0], dtype=np.float32)
    overlay[excluded] = 0.2 * overlay[excluded] + 0.8 * np.array([110, 40, 150], dtype=np.float32)
    overlay[pupil] = 0.15 * overlay[pupil]

    annulus_contours, _ = cv.findContours(annulus.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    excluded_contours, _ = cv.findContours(excluded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pupil_contours, _ = cv.findContours(pupil.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(overlay, annulus_contours, -1, (0, 255, 0), 2)
    cv.drawContours(overlay, excluded_contours, -1, (110, 40, 150), 2)
    cv.drawContours(overlay, pupil_contours, -1, (0, 0, 0), 2)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_source_overlay_preview(output_path, source_debug):
    overlay = build_source_mask_overlay(source_debug)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].imshow(source_debug["gray"], cmap="gray")
    axes[0].set_title("Original Eye Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Source Mask Overlay")
    axes[1].axis("off")

    axes[2].imshow(source_debug["valid"], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Valid Iris Area")
    axes[2].axis("off")

    axes[3].imshow(source_debug["excluded"], cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Excluded Iris Area")
    axes[3].axis("off")

    figure.suptitle(
        "Overlay colors: green = iris annulus, purple = excluded inside annulus, black = pupil",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_band_overlay_preview(output_path, raw_image, iris_band, original_mask, enlarged_mask):
    original_overlay = build_mask_overlay(iris_band, original_mask, original_mask)
    enlarged_overlay = build_mask_overlay(iris_band, original_mask, enlarged_mask)
    newly_masked = ((original_mask == 255) & (enlarged_mask != 255)).astype(np.uint8) * 255

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    axes[0].imshow(raw_image, cmap="gray")
    axes[0].set_title("Original Eye Image")
    axes[0].axis("off")

    axes[1].imshow(original_overlay)
    axes[1].set_title("Original Band Mask Overlay")
    axes[1].axis("off")

    axes[2].imshow(enlarged_overlay)
    axes[2].set_title("Enlarged Band Mask Overlay")
    axes[2].axis("off")

    axes[3].imshow(iris_band, cmap="gray")
    axes[3].set_title("Original Unwrapped Iris Band")
    axes[3].axis("off")

    axes[4].imshow(original_mask, cmap="gray", vmin=0, vmax=255)
    axes[4].set_title("Original Mask (Band Coordinates)")
    axes[4].axis("off")

    axes[5].imshow(newly_masked, cmap="gray", vmin=0, vmax=255)
    axes[5].set_title("Newly Masked Area")
    axes[5].axis("off")

    figure.suptitle(
        "Band overlay colors: red = originally invalid, yellow = newly masked after enlargement",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_single(args):
    classifier = IrisClassifier(filters)
    image_path, image = load_image(args.image)
    backend = resolve_backend(args.segmentation_backend)
    iris_band, iris_mask = segment_image(image, backend=backend)

    enlarged_mask = build_enlarged_mask(iris_mask, args.kernel_size, args.iterations)
    occluded_iris = apply_synthetic_occlusion(
        iris_band,
        iris_mask,
        enlarged_mask,
        args.fill_mode,
    )

    baseline = symmetric_compare_with_details(
        classifier,
        iris_band,
        iris_mask,
        iris_band,
        iris_mask,
        args.rotation,
    )
    mask_only = symmetric_compare_with_details(
        classifier,
        iris_band,
        iris_mask,
        iris_band,
        enlarged_mask,
        args.rotation,
    )
    occluded = symmetric_compare_with_details(
        classifier,
        iris_band,
        iris_mask,
        occluded_iris,
        enlarged_mask,
        args.rotation,
    )

    original_valid_fraction = float(np.mean(iris_mask == 255))
    enlarged_valid_fraction = float(np.mean(enlarged_mask == 255))
    newly_masked_pixels = int(np.sum((iris_mask == 255) & (enlarged_mask != 255)))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    overlay_path = output_dir / f"{stem}_mask_occlusion_overlay.png"
    if backend == "unet":
        source_debug = get_unet_source_debug(image)
        save_source_overlay_preview(overlay_path, source_debug)
    else:
        save_band_overlay_preview(overlay_path, image, iris_band, iris_mask, enlarged_mask)

    result = {
        "metadata": {
            "image_path": str(image_path),
            "segmentation_backend": backend,
            "rotation": args.rotation,
            "kernel_size": args.kernel_size,
            "iterations": args.iterations,
            "fill_mode": args.fill_mode,
            "original_valid_fraction": original_valid_fraction,
            "enlarged_valid_fraction": enlarged_valid_fraction,
            "newly_masked_pixels": newly_masked_pixels,
        },
        "baseline_original_vs_original": baseline,
        "test_1_mask_only": mask_only,
        "test_2_mask_and_occlusion": occluded,
        "overlay_preview_path": str(overlay_path),
    }

    output_path = output_dir / f"{stem}_mask_occlusion_single.json"
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"Saved single-image analysis to: {output_path}")
    print(f"Saved overlay preview to: {overlay_path}")


def run_control(args):
    classifier = IrisClassifier(filters)
    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
    backend = resolve_backend(args.segmentation_backend)
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

    segmented = []
    skipped = []
    for index, image in enumerate(images, start=1):
        if index == 1 or index % 10 == 0 or index == len(images):
            print(f"Segmenting samples: {index}/{len(images)}")
        iris_band, iris_mask = segment_with_optional_backend(image, backend=backend)
        if iris_band is None or iris_mask is None:
            skipped.append(str(image_names[index - 1]))
            continue

        enlarged_mask = build_enlarged_mask(iris_mask, args.kernel_size, args.iterations)
        occluded_iris = apply_synthetic_occlusion(
            iris_band,
            iris_mask,
            enlarged_mask,
            args.fill_mode,
        )
        segmented.append(
            {
                "name": str(image_names[index - 1]),
                "label": str(labels[index - 1]),
                "iris_band": iris_band,
                "iris_mask": iris_mask,
                "enlarged_mask": enlarged_mask,
                "occluded_iris": occluded_iris,
            }
        )

    if len(segmented) < 2:
        raise RuntimeError("Need at least two segmented samples for the control test.")

    baseline_rows = []
    mask_only_rows = []
    occluded_rows = []
    pair_count = len(segmented) * (len(segmented) - 1) // 2

    for pair_index, (idx1, idx2) in enumerate(combinations(range(len(segmented)), 2), start=1):
        sample1 = segmented[idx1]
        sample2 = segmented[idx2]
        same_class = sample1["label"] == sample2["label"]

        baseline = symmetric_compare_with_details(
            classifier,
            sample1["iris_band"],
            sample1["iris_mask"],
            sample2["iris_band"],
            sample2["iris_mask"],
            args.rotation,
        )
        mask_only = symmetric_compare_with_details(
            classifier,
            sample1["iris_band"],
            sample1["iris_mask"],
            sample2["iris_band"],
            sample2["enlarged_mask"],
            args.rotation,
        )
        occluded = symmetric_compare_with_details(
            classifier,
            sample1["iris_band"],
            sample1["iris_mask"],
            sample2["occluded_iris"],
            sample2["enlarged_mask"],
            args.rotation,
        )

        pair_meta = {
            "same_class": bool(same_class),
            "label_1": sample1["label"],
            "label_2": sample2["label"],
            "image_1": sample1["name"],
            "image_2": sample2["name"],
        }

        baseline_rows.append({**pair_meta, **baseline})
        mask_only_rows.append({**pair_meta, **mask_only})
        occluded_rows.append({**pair_meta, **occluded})

        if pair_index == 1 or pair_index % 100 == 0 or pair_index == pair_count:
            print(f"Scored pairs: {pair_index}/{pair_count}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = dataset_format.replace("-", "")
    base_name = (
        f"mask_occlusion_control_{dataset_slug}_ids"
        f"{args.max_identities if args.max_identities is not None else 'all'}_img"
        f"{args.max_images_per_identity if args.max_images_per_identity is not None else 'all'}_seed{args.seed}"
    )

    baseline_summary = summarize_rows(baseline_rows)
    mask_only_summary = summarize_rows(mask_only_rows)
    occluded_summary = summarize_rows(occluded_rows)

    result = {
        "metadata": {
            "dataset_format": dataset_format,
            "dataset_path": str(dataset_path),
            "rotation": args.rotation,
            "segmentation_backend": backend,
            "kernel_size": args.kernel_size,
            "iterations": args.iterations,
            "fill_mode": args.fill_mode,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "max_identities": args.max_identities,
            "max_images_per_identity": args.max_images_per_identity,
            "segmented_sample_count": len(segmented),
            "skipped_samples": skipped,
        },
        "baseline_original_vs_original": baseline_summary,
        "control_mask_only": mask_only_summary,
        "control_mask_and_occlusion": occluded_summary,
    }

    json_path = output_dir / f"{base_name}.json"
    json_path.write_text(json.dumps(result, indent=2))

    if args.save_pairs:
        write_csv(output_dir / f"{base_name}_baseline.csv", baseline_rows)
        write_csv(output_dir / f"{base_name}_mask_only.csv", mask_only_rows)
        write_csv(output_dir / f"{base_name}_mask_and_occlusion.csv", occluded_rows)

    print(json.dumps(result, indent=2))
    print(f"Saved control-test summary to: {json_path}")


def build_parser():
    parser = ArgumentParser(
        description=(
            "Run mask-occlusion robustness tests for iris recognition. "
            "Includes single-image mask-only and mask+occlusion tests, plus a dataset control test."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON/CSV/image outputs.",
    )
    parser.add_argument(
        "--segmentation-backend",
        default=None,
        help="Optional segmentation backend override, for example 'wahet' or 'unet'.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single",
        help="Run test 1 and test 2 on one image.",
    )
    single.add_argument("image", help="Path to the input iris image.")
    single.add_argument("--rotation", type=int, default=21, help="Number of offsets to evaluate.")
    single.add_argument("--kernel-size", type=int, default=17, help="Odd dilation kernel size in pixels.")
    single.add_argument("--iterations", type=int, default=1, help="Number of dilation iterations.")
    single.add_argument(
        "--fill-mode",
        choices=["zero", "white", "mean"],
        default="zero",
        help="How to overwrite newly masked pixels for test 2.",
    )
    single.add_argument(
        "--segmentation-backend",
        default=None,
        help="Optional segmentation backend override, for example 'wahet' or 'unet'.",
    )
    single.set_defaults(func=run_single)

    control = subparsers.add_parser(
        "control",
        help="Run the dataset-level control test on genuine and impostor pairs.",
    )
    control.add_argument(
        "--dataset-path",
        help="Path to the dataset directory. If omitted, a known default path is used.",
    )
    control.add_argument(
        "--dataset-format",
        default="casia-v3-interval",
        choices=["auto", "casia-v1", "casia-v3-interval", "casia-v3-lamp", "casia-v3-twins"],
        help="Dataset folder layout to load.",
    )
    control.add_argument("--rotation", type=int, default=21, help="Number of offsets to evaluate.")
    control.add_argument("--kernel-size", type=int, default=17, help="Odd dilation kernel size in pixels.")
    control.add_argument("--iterations", type=int, default=1, help="Number of dilation iterations.")
    control.add_argument(
        "--fill-mode",
        choices=["zero", "white", "mean"],
        default="zero",
        help="How to overwrite newly masked pixels for the occlusion test.",
    )
    control.add_argument("--max-samples", type=int, default=None, help="Optional cap on total sampled images.")
    control.add_argument("--max-identities", type=int, default=20, help="Optional cap on sampled identities.")
    control.add_argument(
        "--max-images-per-identity",
        type=int,
        default=2,
        help="Optional cap on sampled images per identity.",
    )
    control.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling.")
    control.add_argument(
        "--save-pairs",
        action="store_true",
        help="Also save per-pair CSV files in addition to the summary JSON.",
    )
    control.add_argument(
        "--segmentation-backend",
        default=None,
        help="Optional segmentation backend override, for example 'wahet' or 'unet'.",
    )
    control.set_defaults(func=run_control)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
