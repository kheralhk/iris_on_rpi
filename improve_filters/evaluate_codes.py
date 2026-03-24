from pathlib import Path
import argparse

import numpy as np

from iris import hamming_distance


OUTPUT_DIR = Path(__file__).resolve().parent


def compute_eer(scores, labels):
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([-np.inf], thresholds, [np.inf]))

    best_eer = None
    best_threshold = None
    best_gap = None

    genuine_count = np.sum(labels)
    impostor_count = np.sum(~labels)

    for threshold in thresholds:
        false_rejects = np.sum((scores > threshold) & labels)
        false_accepts = np.sum((scores <= threshold) & (~labels))

        fnr = false_rejects / genuine_count if genuine_count else 0.0
        fpr = false_accepts / impostor_count if impostor_count else 0.0
        gap = abs(fnr - fpr)
        eer = (fnr + fpr) / 2.0

        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_eer = eer
            best_threshold = threshold

    return best_eer, best_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pairwise iris codes and report EER."
    )
    parser.add_argument(
        "--codes-path",
        default=str(OUTPUT_DIR / "codes.npy"),
        help="Path to codes.npy.",
    )
    parser.add_argument(
        "--labels-path",
        default=str(OUTPUT_DIR / "pair_labels.npy"),
        help="Path to pair_labels.npy.",
    )
    args = parser.parse_args()

    codes = np.load(args.codes_path, allow_pickle=True)
    labels = np.load(args.labels_path).astype(bool)

    if len(codes) != len(labels):
        raise ValueError(
            f"Code count mismatch: found {len(codes)} code pairs and {len(labels)} labels."
        )

    scores = np.empty(len(codes), dtype=np.float64)
    for index, (code1, code2, mask1, mask2) in enumerate(codes):
        scores[index] = hamming_distance(code1, code2, mask1, mask2)

    eer, threshold = compute_eer(scores, labels)

    genuine_scores = scores[labels]
    impostor_scores = scores[~labels]

    print(f"pairs: {len(scores)}")
    print(f"mated pairs: {len(genuine_scores)}")
    print(f"non-mated pairs: {len(impostor_scores)}")
    print(f"mean mated score: {genuine_scores.mean():.6f}")
    print(f"mean non-mated score: {impostor_scores.mean():.6f}")
    print(f"EER: {eer:.6f}")
    print(f"threshold at EER: {threshold:.6f}")


if __name__ == "__main__":
    main()
