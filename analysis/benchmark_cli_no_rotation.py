# benchmark_cli_no_rotation

from argparse import ArgumentParser
from pathlib import Path
import sys
import tempfile
import time

import cv2 as cv
import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters import filters
from iris import IrisClassifier, get_iris_band, hamming_distance


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "cli_benchmark_output"


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


def compare_iris_code_operation(classifier, image, stored_code):
    iris_band, iris_mask = segment_image(image)
    return classifier.compare_iris_code_and_iris(
        iris_band,
        stored_code[0, 0],
        iris_mask,
        stored_code[1, 0],
        rotation=None,
    )


def compare_image_operation(classifier, image1, image2):
    iris1, mask1 = segment_image(image1)
    iris2, mask2 = segment_image(image2)
    code1, code_mask1, _ = classifier.get_iris_code(iris1, mask1)
    code2, code_mask2, _ = classifier.get_iris_code(iris2, mask2)
    return hamming_distance(
        np.asarray(code1, dtype=bool),
        np.asarray(code2, dtype=bool),
        np.asarray(code_mask1, dtype=bool),
        np.asarray(code_mask2, dtype=bool),
    )


def find_operation(classifier, query_image, codes):
    iris_band, iris_mask = segment_image(query_image)
    iris_code, mask_code, _ = classifier.get_iris_code(iris_band, iris_mask, offset=0)

    iris_code = np.asarray(iris_code, dtype=bool)
    mask_code = np.asarray(mask_code, dtype=bool)

    best_match = None
    best_score = float("inf")
    for index in range(codes.shape[1]):
        curr_score = hamming_distance(iris_code, codes[0, index], mask_code, codes[1, index])
        if curr_score < best_score:
            best_score = curr_score
            best_match = index

    return best_match, best_score


def plot_benchmark_results(results, output_path):
    labels = [
        "Enroll",
        "Compare Iris Code",
        "Compare Image",
        "Find",
    ]
    means = [result["mean"] for result in results]

    if output_path:
        output = Path(output_path).expanduser()
        if not output.is_absolute():
            output = DEFAULT_OUTPUT_DIR / output
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=means,
                    text=[f"{mean:.2f}s" for mean in means],
                    textposition="outside",
                    marker_color="#7c8aa5",
                )
            ]
        )
        figure.update_layout(
            title="CLI Benchmark Without Rotation",
            yaxis_title="Mean Time (seconds)",
            template="plotly",
        )
        figure.write_html(output, include_plotlyjs="cdn")
        print(f"Saved benchmark plot to {output}")


def main():
    parser = ArgumentParser(description="Benchmark CLI-style operations with rotations disabled.")
    parser.add_argument("query_image", help="Query image path")
    parser.add_argument("compare_image", help="Second image path for comparison")
    parser.add_argument(
        "--database-images",
        nargs="+",
        help="Images to enroll into an in-memory database for the find benchmark",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per benchmark")
    parser.add_argument(
        "--figure-output",
        default=str(DEFAULT_OUTPUT_DIR / "benchmark_cli_no_rotation.html"),
        help="Output path for the benchmark bar chart HTML",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    classifier = IrisClassifier(filters)
    query_image = load_image(args.query_image)
    compare_image = load_image(args.compare_image)

    stored_database = build_database(
        classifier,
        args.database_images if args.database_images else [args.query_image, args.compare_image],
    )
    stored_code = stored_database[:, :1, :]

    results = []
    results.append(benchmark(
        "enroll_no_rotation",
        args.runs,
        lambda: enroll_operation(classifier, query_image),
    ))
    results.append(benchmark(
        "compare_iris_code_no_rotation",
        args.runs,
        lambda: compare_iris_code_operation(classifier, query_image, stored_code),
    ))
    results.append(benchmark(
        "compare_image_no_rotation",
        args.runs,
        lambda: compare_image_operation(classifier, query_image, compare_image),
    ))
    results.append(benchmark(
        "find_no_rotation",
        args.runs,
        lambda: find_operation(classifier, query_image, stored_database),
    ))

    plot_benchmark_results(results, args.figure_output)


if __name__ == "__main__":
    main()
