# plot_hamming_distribution.py

from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_DATASET_PATH = Path(
    "/Users/krist/Documents/project/casia/CASIA Version.1/CASIA Iris Image Database (version 1.0)"
)


def load_labels(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    labels = []
    output_files = sorted(path for path in dataset_dir.iterdir() if "_output" in path.name)
    if not output_files:
        raise FileNotFoundError(
            "No CASIA output images were found in the dataset directory. "
            "Expected filenames containing '_output'."
        )

    for image_path in output_files:
        labels.append(image_path.name[:3])

    return np.array(labels)


def load_required_array(path_argument, argument_name):
    if not path_argument:
        raise ValueError(f"{argument_name} is required.")

    array_path = Path(path_argument).expanduser().resolve()
    if not array_path.exists():
        raise FileNotFoundError(f"Could not find required file: {array_path}")

    return np.load(array_path)


def build_pair_scores(labels, idxs1, idxs2, scores):
    idxs1 = np.asarray(idxs1).reshape(-1)
    idxs2 = np.asarray(idxs2).reshape(-1)
    scores = np.asarray(scores).reshape(-1)

    if not (len(idxs1) == len(idxs2) == len(scores)):
        raise ValueError("idx1, idx2, and scores must have the same number of entries.")

    pair_to_score = {}
    for idx1, idx2, score in zip(idxs1, idxs2, scores, strict=False):
        key = tuple(sorted((int(idx1), int(idx2))))
        if key not in pair_to_score or score < pair_to_score[key]:
            pair_to_score[key] = float(score)

    final_scores = []
    same_class = []
    for idx1, idx2 in combinations(range(len(labels)), 2):
        key = (idx1, idx2)
        final_scores.append(pair_to_score.get(key, np.inf))
        same_class.append(labels[idx1] == labels[idx2])

    return np.array(final_scores), np.array(same_class, dtype=bool)


def plot_distribution(mated_scores, non_mated_scores, output_path=None):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(mated_scores, label="Mated", color="#3b5bff", fill=True, alpha=0.55)
    sns.kdeplot(non_mated_scores, label="Non-Mated", color="#ff4d4f", fill=True, alpha=0.55)
    plt.title("Hamming Distance Distribution: Mated vs. Non-Mated")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Density")
    plt.xlim(0.0, 0.6)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    if output_path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    parser = ArgumentParser(description="Plot mated vs non-mated Hamming distance distributions.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH), help="Path to the CASIA V1 dataset")
    parser.add_argument("--idx1", required=True, help="Path to idx1.npy")
    parser.add_argument("--idx2", required=True, help="Path to idx2.npy")
    parser.add_argument("--scores", required=True, help="Path to scores.npy")
    parser.add_argument("--output", help="Optional output PNG path")
    args = parser.parse_args()

    labels = load_labels(args.dataset_path)
    idxs1 = load_required_array(args.idx1, "--idx1")
    idxs2 = load_required_array(args.idx2, "--idx2")
    scores = load_required_array(args.scores, "--scores")

    distances, same_class = build_pair_scores(labels, idxs1, idxs2, scores)
    finite_mask = np.isfinite(distances)
    distances = distances[finite_mask]
    same_class = same_class[finite_mask]

    if len(distances) == 0:
        raise ValueError("No finite pairwise scores were found in the provided arrays.")

    plot_distribution(distances[same_class], distances[~same_class], output_path=args.output)


if __name__ == "__main__":
    main()
