# benchmark_cli_no_rotation

from argparse import ArgumentParser
from pathlib import Path
import sys
import tempfile
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters import filters
from iris import IrisClassifier, get_iris_band, hamming_distance
from pairwise_iris_analysis import (
    MATCHER_IRISCODE,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / Path(__file__).stem


def load_image(path):
    image_path = Path(path).expanduser().resolve()
    image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image '{image_path}'.")
    return image


def segment_image(image):
    iris_band, iris_mask = get_iris_band(image)
    if iris_band is None or iris_mask is None:
        raise RuntimeError("Iris segmentation failed.")
    return iris_band, iris_mask


def benchmark(name, runs, func):
    times = np.empty(runs, dtype=np.float64)
    result = None
    for index in range(runs):
        start = time.perf_counter()
        result = func()
        times[index] = time.perf_counter() - start

    print(name)
    print(f"  runs: {runs}")
    print(f"  mean: {times.mean():.8f} s")
    print(f"  median: {np.median(times):.8f} s")
    print(f"  min: {times.min():.8f} s")
    print(f"  max: {times.max():.8f} s")
    print(f"  last_result: {format_result(result)}")
    return {
        "name": name,
        "runs": runs,
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "min": float(times.min()),
        "max": float(times.max()),
        "last_result": format_result(result),
    }


def format_result(result):
    if isinstance(result, np.generic):
        return result.item()
    if isinstance(result, tuple):
        return tuple(format_result(value) for value in result)
    if isinstance(result, list):
        return [format_result(value) for value in result]
    return result


def build_database(classifier, image_paths):
    codes = []
    for image_path in image_paths:
        image = load_image(image_path)
        iris_band, iris_mask = segment_image(image)
        iris_code, mask_code, _ = classifier.get_iris_code(iris_band, iris_mask)
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


def enroll_operation(classifier, image):
    iris_band, iris_mask = segment_image(image)
    iris_code, mask_code, _ = classifier.get_iris_code(iris_band, iris_mask)
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


def compare_iris_code_operation(classifier, image, stored_code, rotation):
    iris_band, iris_mask = segment_image(image)
    search_rotation = rotation if rotation and rotation > 1 else None
    return classifier.compare_iris_code_and_iris(
        iris_band,
        stored_code[0, 0],
        iris_mask,
        stored_code[1, 0],
        rotation=search_rotation,
    )
def compare_image_operation(classifier, image1, image2, rotation):
    iris1, mask1 = segment_image(image1)
    iris2, mask2 = segment_image(image2)
    score, offset = classifier(iris1, iris2, mask1, mask2, rotation=rotation if rotation and rotation > 1 else None)
    return score, offset
def find_operation(classifier, query_image, codes, rotation):
    iris_band, iris_mask = segment_image(query_image)
    search_rotation = rotation if rotation and rotation > 1 else 1
    offsets = np.arange(search_rotation, dtype=np.int64) - search_rotation // 2
    iris_codes, mask_codes, _ = classifier.get_iris_codes(
        iris_band,
        iris_mask,
        offsets=offsets,
    )
    bit_weights = classifier.get_bit_weights(iris_band.shape)

    best_match = None
    best_score = float("inf")
    for index in range(codes.shape[1]):
        curr_scores = [
            hamming_distance(
                np.asarray(code, dtype=bool),
                codes[0, index],
                np.asarray(mask, dtype=bool),
                codes[1, index],
                weights=bit_weights,
            )
            for code, mask in zip(iris_codes, mask_codes)
        ]
        curr_score = float(np.min(curr_scores))
        if curr_score < best_score:
            best_score = curr_score
            best_match = index

    return best_match, best_score
def plot_benchmark_results(results, output_name, title):
    labels = [
        "Enroll",
        "Compare Iris Code",
        "Compare Image",
        "Find",
    ]
    means = [result["mean"] for result in results]

    if output_name:
        output = Path(output_name).expanduser()
        if not output.is_absolute():
            output = DEFAULT_OUTPUT_DIR / output
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(labels, means, color="#7c8aa5")
        axis.set_title(title)
        axis.set_ylabel("Mean Time (seconds)")
        axis.set_ylim(0, max(means) * 1.15 if means else 1.0)
        for bar, mean in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.2f}s",
                ha="center",
                va="bottom",
            )
        figure.tight_layout()
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved benchmark plot to {output}")


def main():
    parser = ArgumentParser(description="Benchmark CLI-style operations for the active matcher and rotation setting.")
    parser.add_argument("query_image", help="Query image path")
    parser.add_argument("compare_image", help="Second image path for comparison")
    parser.add_argument(
        "--database-images",
        nargs="+",
        help="Images to enroll into an in-memory database for the find benchmark",
    )
    parser.add_argument("--rotation", type=int, default=1, help="Rotation count used for comparison and find operations.")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per benchmark")
    parser.add_argument(
        "--output-name",
        "--figure-output",
        dest="output_name",
        default=f"{Path(__file__).stem}.png",
        help="Output filename for the benchmark bar chart inside the default output directory, or an absolute path.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    query_image = load_image(args.query_image)
    compare_image = load_image(args.compare_image)
    database_images = args.database_images if args.database_images else [args.query_image, args.compare_image]

    classifier = IrisClassifier(filters)
    stored_database = build_database(
        classifier,
        database_images,
    )
    stored_template = stored_database[:, :1, :]

    results = []
    results.append(benchmark(
        "enroll",
        args.runs,
        lambda: enroll_operation(classifier, query_image),
    ))
    results.append(benchmark(
        "compare_template",
        args.runs,
        lambda: compare_iris_code_operation(classifier, query_image, stored_template, args.rotation),
    ))
    results.append(benchmark(
        "compare_image",
        args.runs,
        lambda: compare_image_operation(classifier, query_image, compare_image, args.rotation),
    ))
    results.append(benchmark(
        "find",
        args.runs,
        lambda: find_operation(classifier, query_image, stored_database, args.rotation),
    ))

    rotation_label = "No Rotation" if args.rotation <= 1 else f"Rotation {args.rotation}"
    title = f"CLI Benchmark ({MATCHER_IRISCODE}, {rotation_label})"
    plot_benchmark_results(results, args.output_name, title)


if __name__ == "__main__":
    main()
