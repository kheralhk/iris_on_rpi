from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from dataset_loaders import DATASET_CHOICES, load_dataset, resolve_dataset, sample_dataset
from iris import get_iris_band


DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "output" / "generated_filters"
CANDIDATE_COLUMNS = ["fx", "fy", "freq", "strength", "relative_strength"]


def dominant_frequency_candidates(patch, top_k=3):
    patch = patch.astype(np.float32)
    patch -= float(np.mean(patch))
    if np.std(patch) < 1e-6:
        return []
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch)))
    h, w = patch.shape
    cy, cx = h // 2, w // 2
    spectrum[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0
    total_strength = float(np.sum(spectrum))
    if total_strength <= 1e-6:
        return []
    flat_indices = np.argpartition(spectrum.ravel(), -top_k)[-top_k:]
    candidates = []
    for flat_index in flat_indices:
        y, x = np.unravel_index(flat_index, spectrum.shape)
        fy = (y - cy) / h
        fx = (x - cx) / w
        freq = float(np.hypot(fx, fy))
        strength = float(spectrum[y, x])
        if freq <= 1e-6:
            continue
        candidates.append([fx, fy, freq, strength, strength / total_strength])
    return candidates


def collect_candidates(images, image_names, patch_size, stride, max_patches):
    candidates = []
    patches = 0
    for index, image in enumerate(images, start=1):
        if index == 1 or index % 100 == 0 or index == len(images):
            print(f"Analyzing image {index}/{len(images)}: {image_names[index - 1]}")
        try:
            iris_band, iris_mask = get_iris_band(image)
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue
        h, w = iris_band.shape
        ph, pw = patch_size
        for y in range(0, max(h - ph + 1, 0), stride):
            for x in range(0, max(w - pw + 1, 0), stride):
                mask_patch = iris_mask[y : y + ph, x : x + pw]
                if mask_patch.shape != (ph, pw) or not np.all(mask_patch == 255):
                    continue
                patch = iris_band[y : y + ph, x : x + pw]
                candidates.extend(dominant_frequency_candidates(patch))
                patches += 1
                if max_patches is not None and patches >= max_patches:
                    return np.asarray(candidates, dtype=np.float32), patches
    return np.asarray(candidates, dtype=np.float32), patches


def candidate_feature_matrix(candidates, cluster_space):
    fx = candidates[:, 0]
    fy = candidates[:, 1]
    freq = candidates[:, 2]
    if cluster_space == "cartesian":
        return candidates[:, :3]
    if cluster_space == "orientation":
        angle = np.arctan2(fy, fx)
        return np.stack((np.cos(2.0 * angle), np.sin(2.0 * angle), freq), axis=1).astype(np.float32)
    raise ValueError(f"Unsupported cluster space: {cluster_space}")


def candidate_sample_weights(candidates, use_strength_weighting):
    if not use_strength_weighting:
        return None
    weights = np.maximum(candidates[:, 4].astype(np.float64), 1e-12)
    return weights / float(np.mean(weights))


def frequency_center_from_cluster(center, cluster_space):
    if cluster_space == "cartesian":
        fx, fy, freq = [float(value) for value in center]
        return fx, fy, freq
    if cluster_space == "orientation":
        cos2, sin2, freq = [float(value) for value in center]
        angle = 0.5 * float(np.arctan2(sin2, cos2))
        fx = float(freq * np.cos(angle))
        fy = float(freq * np.sin(angle))
        return fx, fy, float(freq)
    raise ValueError(f"Unsupported cluster space: {cluster_space}")


def filter_from_cluster(center, band_shape, cluster_space, sigma_scale, rotate_envelope):
    fx, fy, freq = frequency_center_from_cluster(center, cluster_space)
    wavelength = float(np.clip(1.0 / max(freq, 1e-6), 2.0, 64.0))
    theta = float(np.arctan2(fy, fx) + np.pi / 2.0)
    sigma = float(np.clip(wavelength * sigma_scale, 2.0, 12.0))
    size_y = int(max(17, min(65, round(sigma * 6) // 2 * 2 + 1)))
    size_x = int(max(17, min(65, round(sigma * 4) // 2 * 2 + 1)))
    stride_x = 16 if wavelength < 16 else 32
    stride_y = 8
    return {
        "filter": {
            "size": (size_y, size_x),
            "sigma": round(sigma, 4),
            "theta": round(theta, 8),
            "lambd": round(wavelength, 4),
            "psi": 0,
            "gamma": 0.5,
            "rotate_envelope": bool(rotate_envelope),
        },
        "stride": (stride_x, stride_y),
        "start_position": (stride_x // 2, stride_y),
    }


def write_filters(path, filters):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("import numpy as np\n\n")
        handle.write("filters = [\n")
        for item in filters:
            params = item["filter"]
            handle.write(
                "    {"
                f"\"filter\": {{\"size\": {params['size']}, \"sigma\": {params['sigma']}, "
                f"\"theta\": {params['theta']}, \"lambd\": {params['lambd']}, "
                f"\"psi\": {params['psi']}, \"gamma\": {params['gamma']}, "
                f"\"rotate_envelope\": {params['rotate_envelope']}}}, "
                f"\"stride\": {item['stride']}, "
                f"\"start_position\": {item['start_position']}"
                "},\n"
            )
        handle.write("]\n")


def parse_args():
    parser = ArgumentParser(description="Generate a Gabor filter bank from iris texture FFT peaks.")
    parser.add_argument("--dataset", "--dataset-format", dest="dataset_format", default="auto", choices=DATASET_CHOICES)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--filter-count", type=int, default=20)
    parser.add_argument("--output", required=True, help="Output filename or path. Relative names go under output/generated_filters/.")
    parser.add_argument("--max-id", "--max-identities", dest="max_identities", type=int, default=None)
    parser.add_argument("--max-img-per-id", "--max-images-per-identity", dest="max_images_per_identity", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=0.55,
        help="Set sigma as wavelength * sigma_scale before clipping. Default matches the previous heuristic.",
    )
    parser.add_argument(
        "--rotate-envelope",
        action="store_true",
        help="Generate filters whose Gaussian envelope is rotated with theta.",
    )
    parser.add_argument(
        "--cluster-space",
        choices=["orientation", "cartesian"],
        default="orientation",
        help=(
            "Feature space for KMeans. orientation uses [cos(2*angle), sin(2*angle), freq]; "
            "cartesian uses the old [fx, fy, freq] coordinates."
        ),
    )
    parser.add_argument(
        "--no-strength-weighting",
        action="store_true",
        help="Disable FFT peak-strength sample weighting during KMeans.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sigma_scale <= 0:
        raise ValueError("--sigma-scale must be positive")
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
    candidates, patch_count = collect_candidates(
        images,
        image_names,
        patch_size=(args.patch_size, args.patch_size),
        stride=args.patch_stride,
        max_patches=args.max_patches,
    )
    if candidates.shape[0] == 0:
        raise RuntimeError("No valid FFT candidates found.")
    cluster_count = min(args.filter_count, candidates.shape[0])
    cluster_features = candidate_feature_matrix(candidates, args.cluster_space)
    sample_weights = candidate_sample_weights(candidates, not args.no_strength_weighting)
    kmeans = KMeans(n_clusters=cluster_count, random_state=args.seed, n_init=10)
    kmeans.fit(cluster_features, sample_weight=sample_weights)
    filters = [
        filter_from_cluster(center, (64, 512), args.cluster_space, args.sigma_scale, args.rotate_envelope)
        for center in kmeans.cluster_centers_
    ]

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = DEFAULT_OUTPUT_DIR / output
    if output.suffix == "":
        output = output.with_suffix(".py")
    write_filters(output, filters)
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "dataset": dataset_format,
                "patch_count": patch_count,
                "candidate_count": int(candidates.shape[0]),
                "candidate_columns": CANDIDATE_COLUMNS,
                "cluster_space": args.cluster_space,
                "strength_weighting": not args.no_strength_weighting,
                "sigma_scale": args.sigma_scale,
                "rotate_envelope": args.rotate_envelope,
                "strength_summary": {
                    "min": float(np.min(candidates[:, 4])),
                    "mean": float(np.mean(candidates[:, 4])),
                    "max": float(np.max(candidates[:, 4])),
                },
                "filter_count": len(filters),
                "filters": filters,
            },
            indent=2,
        )
    )
    print(f"Generated {len(filters)} filters from {patch_count} patches.")
    print(f"Filters: {output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
