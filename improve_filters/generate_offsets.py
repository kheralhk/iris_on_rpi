from pathlib import Path
from itertools import combinations
from multiprocessing import Pool
import argparse

import numpy as np

from dataset import load_casiav1
from filters import filters
from iris import IrisClassifier


OUTPUT_DIR = Path(__file__).resolve().parent
IMAGES = None
MASKS = None
IRIS_CLASSIFIER = None
ROTATION = None


def build_pair_list(image_count):
    return list(combinations(range(image_count), 2))


def init_worker(images, masks, rotation):
    global IMAGES
    global MASKS
    global IRIS_CLASSIFIER
    global ROTATION

    IMAGES = images
    MASKS = masks
    ROTATION = rotation
    IRIS_CLASSIFIER = IrisClassifier(filters[:8])


def get_offset(pair):
    idx_1, idx_2 = pair
    _, offset = IRIS_CLASSIFIER(
        IMAGES[idx_1],
        IMAGES[idx_2],
        MASKS[idx_1],
        MASKS[idx_2],
        rotation=ROTATION,
    )
    return offset


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise iris alignment offsets for filter analysis."
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=20,
        help="Rotation search window passed to IrisClassifier (default: 20).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Number of worker processes (default: 14).",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional path to the CASIA V1 dataset.",
    )
    parser.add_argument(
        "--iris-filter",
        action="store_true",
        help="Use only people selected by good_persons.npy.",
    )
    args = parser.parse_args()

    images, masks, _ = load_casiav1(
        iris_filter=args.iris_filter,
        dataset_path=args.dataset_path,
    )
    pair_list = build_pair_list(len(images))

    with Pool(
        args.workers,
        initializer=init_worker,
        initargs=(images, masks, args.rotation),
    ) as pool:
        offsets = pool.map(get_offset, pair_list)

    output_path = OUTPUT_DIR / f"offsets-{args.rotation}.npy"
    np.save(output_path, offsets)
    print(f"Saved {len(offsets)} offsets to {output_path}")


if __name__ == "__main__":
    main()
