from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from dataset_loaders import load_dataset, resolve_dataset
from filters import filters
from iris import IrisClassifier, get_iris_band, get_segmentation_backend_name, hamming_distances
from pairwise_iris_analysis import (
    MATCHER_IRISCODE,
    compute_pairwise_scores_iriscode,
    evaluate_scores,
    precompute_codes,
    sample_dataset,
    summarize_label_pairs,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "benchmark_pipeline"
DEFAULT_PAIRWISE_CONFIGS = {
    "casia-v1": {
        "max_samples": None,
        "max_identities": None,
        "max_images_per_identity": None,
        "seed": 0,
    },
    "casia-v3-interval": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-v4-interval": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-v3-lamp": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "casia-v3-twins": {
        "max_samples": None,
        "max_identities": 200,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "ubiris-v2": {
        "max_samples": None,
        "max_identities": 250,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "iitd": {
        "max_samples": None,
        "max_identities": 224,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "mmu": {
        "max_samples": None,
        "max_identities": None,
        "max_images_per_identity": 2,
        "seed": 0,
    },
    "mmu2": {
        "max_samples": None,
        "max_identities": None,
        "max_images_per_identity": 2,
        "seed": 0,
    },
}


def format_result(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [format_result(item) for item in value]
    if isinstance(value, list):
        return [format_result(item) for item in value]
    if isinstance(value, dict):
        return {key: format_result(item) for key, item in value.items()}
    return value


def segment_image(image):
    try:
        return get_iris_band(image)
    except Exception:
        return None, None


def benchmark(name, runs, func):
    times = np.empty(runs, dtype=np.float64)
    result = None
    for index in range(runs):
        start = time.perf_counter()
        result = func()
        times[index] = time.perf_counter() - start

    return {
        "name": name,
        "runs": int(runs),
        "mean_seconds": float(times.mean()),
        "last_result": format_result(result),
    }


def find_speed_benchmark_samples(dataset_path, dataset_format, database_size):
    images, labels, image_names = load_dataset(dataset_path, dataset_format)
    label_to_samples = {}

    def has_enough_samples():
        for label, sample_group in label_to_samples.items():
            if len(sample_group) < 2:
                continue
            impostor_count = sum(len(group) for other_label, group in label_to_samples.items() if other_label != label)
            if impostor_count >= max(database_size - 1, 0):
                return True
        return False

    for image, label, name in zip(images, labels, image_names):
        iris_band, iris_mask = segment_image(image)
        if iris_band is None or iris_mask is None:
            continue
        label_to_samples.setdefault(str(label), []).append(
            {
                "raw_image": image,
                "iris_band": iris_band,
                "iris_mask": iris_mask,
                "label": str(label),
                "image_name": str(name),
            }
        )
        if has_enough_samples():
            break

    samples = []
    for sample_group in label_to_samples.values():
        samples.extend(sample_group)
    return samples


def build_database(classifier, samples):
    codes = []
    for sample in samples:
        iris_code, mask_code, _ = classifier.get_iris_code(sample["iris_band"], sample["iris_mask"])
        codes.append(
            np.stack(
                (
                    np.asarray(iris_code, dtype=bool),
                    np.asarray(mask_code, dtype=bool),
                ),
                axis=0,
            )
        )
    return np.stack(codes, axis=1)


def select_benchmark_samples(samples, database_size):
    label_to_samples = {}
    for sample in samples:
        label_to_samples.setdefault(sample["label"], []).append(sample)

    query = None
    compare_sample = None
    for label, sample_group in label_to_samples.items():
        if len(sample_group) >= 2:
            impostor_count = sum(
                len(other_group)
                for other_label, other_group in label_to_samples.items()
                if other_label != label
            )
            if impostor_count >= max(database_size - 1, 0):
                query = sample_group[0]
                compare_sample = sample_group[1]
                break
    if query is None or compare_sample is None:
        raise RuntimeError(
            "Need at least one identity with two valid images and enough impostors for the benchmark database."
        )

    impostor_pool = []
    for label, sample_group in label_to_samples.items():
        if label == query["label"]:
            continue
        impostor_pool.extend(sample_group)

    if len(impostor_pool) < max(database_size - 1, 0):
        raise RuntimeError(
            f"Need {database_size - 1} impostor samples for the benchmark database, found {len(impostor_pool)}."
        )

    database_samples = [compare_sample]
    database_samples.extend(impostor_pool[: max(database_size - 1, 0)])
    return query, compare_sample, database_samples


def enroll_operation(classifier, sample):
    iris_code, mask_code, _ = classifier.get_iris_code(sample["iris_band"], sample["iris_mask"])
    code = np.stack(
        (
            np.asarray(iris_code, dtype=bool),
            np.asarray(mask_code, dtype=bool),
        ),
        axis=0,
    )
    with tempfile.NamedTemporaryFile(suffix=".npy") as handle:
        np.save(handle.name, code[:, np.newaxis, :])
    return code.shape


def compare_image_operation(classifier, sample1, sample2, rotation):
    score, offset = classifier(
        sample1["iris_band"],
        sample2["iris_band"],
        sample1["iris_mask"],
        sample2["iris_mask"],
        rotation=rotation,
    )
    return score, offset


def find_operation(classifier, sample, codes, rotation):
    offsets = np.arange(rotation) - rotation // 2
    iris_codes, mask_codes, _ = classifier.get_iris_codes(
        sample["iris_band"],
        sample["iris_mask"],
        offsets=offsets,
    )
    bit_weights = classifier.get_bit_weights(sample["iris_band"].shape)
    best_match = None
    best_score = float("inf")
    for index in range(codes.shape[1]):
        curr_scores = hamming_distances(
            iris_codes,
            codes[0, index],
            mask_codes,
            codes[1, index],
            weights=bit_weights,
        )
        curr_score = float(np.min(curr_scores))
        if curr_score < best_score:
            best_score = curr_score
            best_match = index
    return best_match, best_score


def compare_iris_codes_operation(query_code, compare_code):
    bit_weights = query_code[2]
    score = hamming_distances(
        query_code[0][None, :],
        compare_code[0],
        query_code[1][None, :],
        compare_code[1],
        weights=bit_weights,
    )[0]
    return float(score)

def build_functional_summary(operations, matcher):
    operation_lookup = {item["name"]: item for item in operations}
    compare_score = float(operation_lookup["compare_image"]["last_result"][0])
    compare_template_score = float(operation_lookup["compare_templates"]["last_result"])
    find_index = int(operation_lookup["find"]["last_result"][0])
    find_score = float(operation_lookup["find"]["last_result"][1])

    mate_found = find_index == 0
    default_threshold = 0.3 if matcher == MATCHER_IRISCODE else None

    if default_threshold is None:
        threshold_checks = {
            "genuine_image_score_below_default_threshold": None,
            "genuine_template_score_below_default_threshold": None,
            "find_score_below_default_threshold": None,
            "find_returns_enrolled_mate": bool(mate_found),
        }
        passed = bool(mate_found)
    else:
        genuine_below_default_threshold = compare_score < default_threshold
        template_score_below_default_threshold = compare_template_score < default_threshold
        find_score_below_default_threshold = find_score < default_threshold
        threshold_checks = {
            "genuine_image_score_below_default_threshold": bool(genuine_below_default_threshold),
            "genuine_template_score_below_default_threshold": bool(template_score_below_default_threshold),
            "find_score_below_default_threshold": bool(find_score_below_default_threshold),
            "find_returns_enrolled_mate": bool(mate_found),
        }
        passed = bool(
            genuine_below_default_threshold
            and template_score_below_default_threshold
            and find_score_below_default_threshold
            and mate_found
        )

    return {
        "is_functional_iris_recognizer": "yes" if passed else "no",
        "matcher": matcher,
        "default_threshold": default_threshold,
        "checks": threshold_checks,
        "details": {
            "compare_image_score": compare_score,
            "compare_template_score": compare_template_score,
            "find_index": find_index,
            "find_score": find_score,
        },
    }

def run_speed_benchmark(dataset_path, dataset_format, rotation, runs, database_size):
    samples = find_speed_benchmark_samples(dataset_path, dataset_format, database_size)
    query, compare_sample, database_samples = select_benchmark_samples(samples, database_size)
    classifier = IrisClassifier(filters)
    codes = build_database(classifier, database_samples)
    query_code = np.stack(
        classifier.get_iris_code(query["iris_band"], query["iris_mask"])[:2],
        axis=0,
    ).astype(bool)
    compare_code = np.stack(
        classifier.get_iris_code(compare_sample["iris_band"], compare_sample["iris_mask"])[:2],
        axis=0,
    ).astype(bool)
    bit_weights = classifier.get_bit_weights(query["iris_band"].shape)

    operations = [
        benchmark("enroll", runs, lambda: enroll_operation(classifier, query)),
        benchmark("compare_image", runs, lambda: compare_image_operation(classifier, query, compare_sample, rotation)),
        benchmark(
            "compare_templates",
            runs,
            lambda: compare_iris_codes_operation((query_code[0], query_code[1], bit_weights), compare_code),
        ),
        benchmark("find", runs, lambda: find_operation(classifier, query, codes, rotation)),
    ]
    feature_extractor = "gabor_radial_weighted"

    return {
        "dataset_format": dataset_format,
        "matcher": MATCHER_IRISCODE,
        "feature_extractor": feature_extractor,
        "segmentation_backend": get_segmentation_backend_name(),
        "query_image": query["image_name"],
        "compare_image": compare_sample["image_name"],
        "database_size": int(database_size),
        "operations": operations,
        "functional_summary": build_functional_summary(operations, MATCHER_IRISCODE),
    }


def run_pairwise_benchmark(dataset_path, dataset_format, rotation, output_dir):
    config = DEFAULT_PAIRWISE_CONFIGS[dataset_format]
    images, labels, image_names = load_dataset(dataset_path, dataset_format)
    images, labels, image_names = sample_dataset(
        images,
        labels,
        image_names,
        max_samples=config["max_samples"],
        max_identities=config["max_identities"],
        max_images_per_identity=config["max_images_per_identity"],
        seed=config["seed"],
    )

    classifier = IrisClassifier(filters)
    (
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        bit_weights,
        kept_labels,
        kept_image_names,
        skipped,
    ) = precompute_codes(
        images,
        labels,
        image_names,
        classifier,
        rotation,
    )
    summary = summarize_label_pairs(kept_labels)
    pairwise = compute_pairwise_scores_iriscode(
        kept_labels,
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        bit_weights=bit_weights,
    )
    feature_extractor = "gabor_radial_weighted"
    evaluation = evaluate_scores(pairwise["same_class"], pairwise["scores"])

    return {
        "dataset_format": dataset_format,
        "matcher": MATCHER_IRISCODE,
        "feature_extractor": feature_extractor,
        "segmentation_backend": get_segmentation_backend_name(),
        "dataset_path": str(dataset_path),
        "sample_summary": summary,
        "sampling": config,
        "rotation": int(rotation),
        "kept_sample_count": int(len(kept_labels)),
        "skipped_sample_count": int(len(skipped)),
        "eer": float(evaluation["eer"]),
        "eer_threshold": float(evaluation["eer_threshold"]),
        "roc_auc": float(evaluation["roc_auc"]),
    }


def main():
    parser = ArgumentParser(
        description=(
            "Evaluate the U-Net + Gabor iris pipeline for pairwise discrimination and key "
            "recognition operations."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "casia-v1",
            "casia-v3-interval",
            "casia-v4-interval",
            "casia-v3-lamp",
            "casia-v3-twins",
            "ubiris-v2",
            "iitd",
            "mmu",
            "mmu2",
        ],
        choices=["casia-v1", "casia-v3-interval", "casia-v4-interval", "casia-v3-lamp", "casia-v3-twins", "ubiris-v2", "iitd", "mmu", "mmu2"],
        help="Datasets to include in the evaluation.",
    )
    parser.add_argument("--rotation", type=int, default=21, help="Rotation count used for scoring.")
    parser.add_argument("--speed-runs", type=int, default=300, help="How many repetitions to use per speed benchmark.")
    parser.add_argument("--database-size", type=int, default=64, help="How many enrolled iris codes to use in the find benchmark.")
    parser.add_argument("--skip-pairwise", action="store_true", help="Skip AUC/EER evaluation.")
    parser.add_argument("--skip-speed", action="store_true", help="Skip speed benchmarks.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON output.")
    parser.add_argument(
        "--output-name",
        default="benchmark_results.json",
        help="Filename for the JSON results inside --output-dir, or an absolute JSON path.",
    )
    args = parser.parse_args()

    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.speed_runs < 1:
        raise ValueError("--speed-runs must be at least 1")
    if args.database_size < 1:
        raise ValueError("--database-size must be at least 1")
    if args.skip_pairwise and args.skip_speed:
        raise ValueError("At least one of pairwise or speed benchmarking must be enabled.")

    output_name = Path(args.output_name).expanduser()
    if output_name.is_absolute():
        output_path = output_name.resolve()
        output_dir = output_path.parent
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "filters_count": int(len(filters)),
        "matcher": MATCHER_IRISCODE,
        "feature_extractor": "gabor_radial_weighted",
        "segmentation_backend": get_segmentation_backend_name(),
        "rotation": int(args.rotation),
        "pairwise": [],
        "speed": None,
    }

    for dataset_format in args.datasets:
        dataset_path, dataset_format = resolve_dataset(None, dataset_format)
        print(f"Evaluating dataset: {dataset_format}")
        if not args.skip_pairwise:
            pairwise_result = run_pairwise_benchmark(
                dataset_path,
                dataset_format,
                args.rotation,
                output_dir,
            )
            results["pairwise"].append(pairwise_result)
            print(f"  pairwise: AUC={pairwise_result['roc_auc']:.4f} EER={pairwise_result['eer']:.4f}")
        output_path.write_text(json.dumps(format_result(results), indent=2))

    if not args.skip_speed:
        speed_dataset_path, speed_dataset_format = resolve_dataset(None, args.datasets[0])
        speed_result = run_speed_benchmark(
            speed_dataset_path,
            speed_dataset_format,
            args.rotation,
            args.speed_runs,
            args.database_size,
        )
        results["speed"] = speed_result
        print(
            f"Speed benchmark dataset: {speed_result['dataset_format']} "
            f"(feature_extractor={speed_result['feature_extractor']}, database_size={speed_result['database_size']}, "
            f"query={speed_result['query_image']})"
        )

    output_path.write_text(json.dumps(format_result(results), indent=2))
    print(f"Saved pipeline results to {output_path}")


if __name__ == "__main__":
    main()
