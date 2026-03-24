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
OFFSETS = None
IRIS_CLASSIFIER = None


def build_pair_list(image_count):
    return list(combinations(range(image_count), 2))


def init_worker(images, masks, offsets):
    global IMAGES
    global MASKS
    global OFFSETS
    global IRIS_CLASSIFIER

    IMAGES = images
    MASKS = masks
    OFFSETS = offsets
    IRIS_CLASSIFIER = IrisClassifier(filters[:8])


def get_codes_for_pair(index_pair):
    index, pair = index_pair
    idx_1, idx_2 = pair
    code1, mask1, _ = IRIS_CLASSIFIER.get_iris_code(IMAGES[idx_1], MASKS[idx_1], offset=0)
    code2, mask2, _ = IRIS_CLASSIFIER.get_iris_code(
        IMAGES[idx_2],
        MASKS[idx_2],
        offset=int(OFFSETS[index]),
    )
    return (
        np.asarray(code1, dtype=bool),
        np.asarray(code2, dtype=bool),
        np.asarray(mask1, dtype=bool),
        np.asarray(mask2, dtype=bool),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise iris codes for filter analysis."
    )
    parser.add_argument(
        "--offsets-path",
        default=None,
        help="Path to the offsets .npy file. Defaults to extra_files/offsets-20.npy.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional path to the CASIA V1 dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=14,
        help="Number of worker processes (default: 14).",
    )
    parser.add_argument(
        "--iris-filter",
        action="store_true",
        help="Use only people selected by good_persons.npy.",
    )
    args = parser.parse_args()

    offsets_path = Path(args.offsets_path) if args.offsets_path else OUTPUT_DIR / "offsets-20.npy"
    offsets = np.load(offsets_path, allow_pickle=True)

    images, masks, labels = load_casiav1(
        iris_filter=args.iris_filter,
        dataset_path=args.dataset_path,
    )
    pair_list = build_pair_list(len(images))

    if len(offsets) != len(pair_list):
        raise ValueError(
            f"Offset count mismatch: found {len(offsets)} offsets for {len(pair_list)} image pairs."
        )

    indexed_pairs = list(enumerate(pair_list))
    with Pool(
        args.workers,
        initializer=init_worker,
        initargs=(images, masks, offsets),
    ) as pool:
        codes = pool.map(get_codes_for_pair, indexed_pairs)

    output_path = OUTPUT_DIR / "codes.npy"
    np.save(output_path, np.array(codes, dtype=object))

    same_class = np.array([labels[idx_1] == labels[idx_2] for idx_1, idx_2 in pair_list], dtype=bool)
    pair_labels_path = OUTPUT_DIR / "pair_labels.npy"
    np.save(pair_labels_path, same_class)

    print(f"Saved {len(codes)} code pairs to {output_path}")
    print(f"Saved pair labels to {pair_labels_path}")


if __name__ == "__main__":
    main()
