# mask_occlusion_tests.py

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_loaders import load_dataset, resolve_dataset
from filters import filters
from iris import IrisClassifier, get_iris_band, hamming_distances, predict_unet_masks

try:
    from segmentation_geometry import build_valid_source_mask, clean_component_mask
except ImportError:
    build_valid_source_mask = None
    clean_component_mask = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "mask_occlusion_tests"


def load_image(path):
    image_path = Path(path).expanduser().resolve()
    image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image '{image_path}'.")
    return image_path, image


def segment_image(image):
    iris_band, iris_mask = get_iris_band(image)
    if iris_band is None or iris_mask is None:
        raise RuntimeError("Iris segmentation failed.")
    return iris_band, iris_mask


def get_unet_source_debug(image):
    if build_valid_source_mask is None or clean_component_mask is None:
        raise RuntimeError("Source overlays require segmentation geometry helpers.")
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
        "base_title": "Original Eye Image",
        "overlay_title": "Source Mask Overlay",
        "valid_title": "Valid Iris Area",
        "excluded_title": "Excluded Iris Area",
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
    bit_weights = classifier.get_bit_weights(iris_ref.shape)

    if rotation is None or rotation <= 1:
        offsets = np.array([0], dtype=np.int64)
        probe_codes, probe_masks, _ = classifier.get_iris_codes(iris_probe, mask_probe, offsets=offsets)
    else:
        offsets = np.arange(rotation, dtype=np.int64) - rotation // 2
        probe_codes, probe_masks, _ = classifier.get_iris_codes(iris_probe, mask_probe, offsets=offsets)

    probe_codes = np.asarray(probe_codes, dtype=bool)
    probe_masks = np.asarray(probe_masks, dtype=bool)

    scores = hamming_distances(probe_codes, ref_code, probe_masks, ref_code_mask, weights=bit_weights)
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


def build_source_mask_overlay(source_debug):
    gray = source_debug["gray"]
    annulus = source_debug["annulus"] > 0
    excluded = source_debug["excluded"] > 0
    pupil = source_debug["pupil"] > 0
    excluded_color = np.array([170, 90, 220], dtype=np.float32)

    overlay = cv.cvtColor(gray, cv.COLOR_GRAY2RGB).astype(np.float32)
    overlay[annulus] = 0.4 * overlay[annulus] + 0.6 * np.array([0, 255, 0], dtype=np.float32)
    overlay[excluded] = 0.55 * overlay[excluded] + 0.45 * excluded_color
    overlay[pupil] = 0.15 * overlay[pupil]

    annulus_contours, _ = cv.findContours(annulus.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    excluded_contours, _ = cv.findContours(excluded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pupil_contours, _ = cv.findContours(pupil.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(overlay, annulus_contours, -1, (0, 255, 0), 2)
    cv.drawContours(overlay, excluded_contours, -1, tuple(int(v) for v in excluded_color), 2)
    cv.drawContours(overlay, pupil_contours, -1, (0, 0, 0), 2)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_source_overlay_preview(output_path, source_debug):
    overlay = build_source_mask_overlay(source_debug)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].imshow(source_debug["gray"], cmap="gray")
    axes[0].set_title(source_debug.get("base_title", "Original Eye Image"))
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(source_debug.get("overlay_title", "Mask Overlay"))
    axes[1].axis("off")

    axes[2].imshow(source_debug["valid"], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(source_debug.get("valid_title", "Valid Area"))
    axes[2].axis("off")

    axes[3].imshow(source_debug["excluded"], cmap="gray", vmin=0, vmax=255)
    axes[3].set_title(source_debug.get("excluded_title", "Excluded Area"))
    axes[3].axis("off")

    figure.suptitle(
        "Overlay colors: green = valid segmentation, purple = excluded area, black = pupil (if available)",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_overlay_preview(output_path, raw_image, iris_band, iris_mask, kernel_size, iterations):
    source_debug = get_unet_source_debug(raw_image)
    save_source_overlay_preview(output_path, source_debug)


def run_single(args):
    classifier = IrisClassifier(filters)
    image_path, image = load_image(args.image)
    iris_band, iris_mask = segment_image(image)

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

    overlay_path = output_dir / f"{stem}_{args.output_label}.png"
    save_overlay_preview(overlay_path, image, iris_band, iris_mask, args.kernel_size, args.iterations)

    print(f"Image: {image_path}")
    print("Backend: unet")
    print(f"Original valid fraction: {original_valid_fraction:.4f}")
    print(f"Enlarged valid fraction: {enlarged_valid_fraction:.4f}")
    print(f"Newly masked pixels: {newly_masked_pixels}")
    print(
        "Baseline score / mask-only score / mask+occlusion score: "
        f"{baseline['score']:.6f} / {mask_only['score']:.6f} / {occluded['score']:.6f}"
    )
    print(
        "Fill mode options: zero=fill with black (0), white=fill with white (255), "
        "mean=fill with the mean valid iris intensity."
    )
    print(f"Saved overlay preview to: {overlay_path}")


def run_multiple(args):
    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset_format)
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

    skipped = []
    saved = 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images, start=1):
        if index == 1 or index % 10 == 0 or index == len(images):
            print(f"Segmenting samples: {index}/{len(images)}")
        iris_band, iris_mask = get_iris_band(image)
        if iris_band is None or iris_mask is None:
            skipped.append(str(image_names[index - 1]))
            continue

        name = Path(str(image_names[index - 1])).stem
        overlay_path = output_dir / f"{name}_{args.output_label}.png"
        save_overlay_preview(overlay_path, image, iris_band, iris_mask, args.kernel_size, args.iterations)
        saved += 1

    print(f"Saved overlay previews: {saved}")
    if skipped:
        print(f"Skipped samples: {len(skipped)}")
        for name in skipped[:10]:
            print(f"  - {name}")


def build_parser():
    parser = ArgumentParser(
        description=(
            "Run mask-occlusion robustness tests for iris recognition. "
            "Includes a single-image mask/occlusion test and a multiple-image overlay export."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for image outputs.",
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
        "--output-label",
        default="mask_occlusion_overlay",
        help="Filename label after the image stem, for example 'source_mask' -> S1129R05_source_mask.png.",
    )
    single.add_argument(
        "--fill-mode",
        choices=["zero", "white", "mean"],
        default="zero",
        help="How to overwrite newly masked pixels for test 2: zero=0, white=255, mean=mean valid intensity.",
    )
    single.set_defaults(func=run_single)

    multiple = subparsers.add_parser(
        "multiple",
        help="Save a mask overlay image for each sampled dataset image.",
    )
    multiple.add_argument(
        "--dataset-path",
        help="Path to the dataset directory. If omitted, a known default path is used.",
    )
    multiple.add_argument(
        "--dataset-format",
        default="casia-v3-interval",
        choices=["auto", "casia-v1", "casia-v3-interval", "casia-v3-lamp", "casia-v3-twins"],
        help="Dataset folder layout to load.",
    )
    multiple.add_argument("--kernel-size", type=int, default=17, help="Odd dilation kernel size in pixels.")
    multiple.add_argument("--iterations", type=int, default=1, help="Number of dilation iterations.")
    multiple.add_argument("--max-samples", type=int, default=None, help="Optional cap on total sampled images.")
    multiple.add_argument("--max-identities", type=int, default=10, help="Optional cap on sampled identities.")
    multiple.add_argument(
        "--max-images-per-identity",
        type=int,
        default=1,
        help="Optional cap on sampled images per identity.",
    )
    multiple.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling.")
    multiple.add_argument(
        "--output-label",
        default="mask_occlusion_overlay",
        help="Filename label after each image stem, for example 'source_mask' -> S1129R05_source_mask.png.",
    )
    multiple.set_defaults(func=run_multiple)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
