from argparse import ArgumentParser
from pathlib import Path
import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from dataset_loaders import DATASET_CHOICES, load_dataset, resolve_dataset, sample_dataset
from filter_loader import load_filter_bank
from iris import IrisClassifier, get_segmentation_backend_name
from mask_occlusion_tests import build_source_mask_overlay, get_source_debug
from pairwise_iris_analysis import (
    MATCHER_IRISCODE,
    compute_pairwise_scores_iriscode,
    precompute_codes,
    summarize_label_pairs,
)


DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "output" / "interactive_hd_distribution"


def add_figure_metadata(figure, metadata):
    if not metadata:
        return
    text = " | ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    if text:
        figure.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, family="monospace", wrap=True)


def scalar_from_npz(value, default=None):
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return default


def load_pairwise_npz(path):
    data = np.load(Path(path).expanduser().resolve(), allow_pickle=True)
    required = ["idx1", "idx2", "scores", "same_class", "image_name1", "image_name2"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays in {path}: {', '.join(missing)}")

    pairwise = {
        "idx1": data["idx1"].astype(np.int32),
        "idx2": data["idx2"].astype(np.int32),
        "scores": data["scores"].astype(np.float32),
        "same_class": data["same_class"].astype(bool),
        "best_offset": data["best_offset"].astype(np.int16)
        if "best_offset" in data.files
        else np.zeros(data["scores"].shape, dtype=np.int16),
        "direction": data["direction"].astype(np.int8)
        if "direction" in data.files
        else np.zeros(data["scores"].shape, dtype=np.int8),
    }
    labels = None
    if "label1" in data.files and "label2" in data.files:
        max_index = int(max(pairwise["idx1"].max(initial=0), pairwise["idx2"].max(initial=0)))
        labels = np.empty(max_index + 1, dtype=object)
        labels[pairwise["idx1"]] = data["label1"].astype(object)
        labels[pairwise["idx2"]] = data["label2"].astype(object)

    max_index = int(max(pairwise["idx1"].max(initial=0), pairwise["idx2"].max(initial=0)))
    image_names = np.empty(max_index + 1, dtype=object)
    image_names[pairwise["idx1"]] = data["image_name1"].astype(object)
    image_names[pairwise["idx2"]] = data["image_name2"].astype(object)

    dataset_path = scalar_from_npz(data["dataset_path"], None) if "dataset_path" in data.files else None
    rotation = scalar_from_npz(data["rotation"], None) if "rotation" in data.files else None
    matcher = scalar_from_npz(data["matcher"], MATCHER_IRISCODE) if "matcher" in data.files else MATCHER_IRISCODE
    return pairwise, labels, image_names, dataset_path, rotation, matcher


def compute_pairwise(args):
    dataset_path, dataset_format = resolve_dataset(args.dataset_path, args.dataset)
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
    pre_summary = summarize_label_pairs(labels)
    if pre_summary["mated_pairs"] == 0 or pre_summary["non_mated_pairs"] == 0:
        raise ValueError("The sampled subset needs both mated and non-mated pairs.")

    selected_filters, filters_source = load_filter_bank(args.filters)
    print(f"Using dataset: {dataset_format}")
    print(f"Using dataset path: {dataset_path}")
    print(f"Filters in use: {len(selected_filters)}")
    print(f"Filters source: {filters_source}")

    classifier = IrisClassifier(selected_filters)
    (
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
        labels,
        image_names,
        skipped,
    ) = precompute_codes(images, labels, image_names, classifier, args.rotation)
    if skipped:
        print(f"Skipped {len(skipped)} images due to segmentation failure.")

    summary = summarize_label_pairs(labels)
    if summary["mated_pairs"] == 0 or summary["non_mated_pairs"] == 0:
        raise ValueError("After segmentation failures, the subset needs both mated and non-mated pairs.")

    pairwise = compute_pairwise_scores_iriscode(
        labels,
        base_codes,
        base_masks,
        rotated_codes,
        rotated_masks,
        offsets,
    )
    return pairwise, labels, image_names, str(dataset_path), args.rotation, MATCHER_IRISCODE, dataset_format


def output_paths(output_dir, dataset_format, output_name):
    clean_name = Path(output_name).stem
    base_dir = Path(output_dir).expanduser().resolve() / str(dataset_format) / clean_name
    return base_dir / f"{clean_name}.png", base_dir / f"{clean_name}_scores.npz"


def save_pairwise_cache(output_path, pairwise, labels, image_names, dataset_path, rotation, matcher):
    idx1 = pairwise["idx1"]
    idx2 = pairwise["idx2"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        idx1=idx1,
        idx2=idx2,
        scores=pairwise["scores"],
        same_class=pairwise["same_class"],
        best_offset=pairwise["best_offset"],
        direction=pairwise["direction"],
        label1=labels[idx1] if labels is not None else np.array([""] * len(idx1), dtype=object),
        label2=labels[idx2] if labels is not None else np.array([""] * len(idx2), dtype=object),
        image_name1=image_names[idx1],
        image_name2=image_names[idx2],
        dataset_path=np.array(str(Path(dataset_path).expanduser().resolve())),
        rotation=np.array(rotation),
        matcher=np.array(matcher),
    )
    return output_path


def load_grayscale(dataset_path, image_name):
    path = Path(dataset_path) / str(image_name)
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


class MaskOverlayCache:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self._cache = {}

    def get(self, image_name):
        key = str(image_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        image = load_grayscale(self.dataset_path, key)
        source_debug = get_source_debug(image)
        overlay = build_source_mask_overlay(source_debug)
        result = {
            "gray": source_debug["gray"],
            "overlay": overlay,
            "valid": source_debug["valid"],
            "excluded": source_debug["excluded"],
        }
        self._cache[key] = result
        return result


class InteractiveDistribution:
    def __init__(
        self,
        pairwise,
        labels,
        image_names,
        dataset_path,
        metadata,
        selection_width,
        show_pairs,
        figure_output_path,
        click_output_dir=None,
    ):
        self.pairwise = pairwise
        self.labels = labels
        self.image_names = np.asarray(image_names, dtype=object)
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.metadata = metadata
        self.selection_width = float(selection_width)
        self.show_pairs = int(show_pairs)
        self.figure_output_path = Path(figure_output_path).expanduser().resolve()
        self.mode = "all"
        self.overlay_cache = MaskOverlayCache(self.dataset_path)
        self.click_output_dir = Path(click_output_dir).expanduser().resolve() if click_output_dir else None
        self.click_count = 0
        self.figure = None
        self.distribution_axis = None
        self.status_text = None
        self.click_marker = None

    def class_mask(self):
        same_class = self.pairwise["same_class"]
        if self.mode == "mated":
            return same_class
        if self.mode == "non-mated":
            return ~same_class
        return np.ones(same_class.shape, dtype=bool)

    def set_mode(self, mode):
        self.mode = mode
        self.update_status()

    def update_status(self):
        if self.status_text is None:
            return
        self.status_text.set_text(
            f"Selection mode: {self.mode} | click distribution to inspect pairs | keys: m=mated, n=non-mated, a=all"
        )
        self.figure.canvas.draw_idle()

    def plot(self):
        scores = self.pairwise["scores"]
        same_class = self.pairwise["same_class"]
        mated_scores = scores[same_class]
        non_mated_scores = scores[~same_class]

        sns.set_theme(style="whitegrid")
        self.figure, axis = plt.subplots(figsize=(13, 7))
        self.distribution_axis = axis

        sns.kdeplot(mated_scores, ax=axis, label="Mated", color="#3b5bff", fill=True, alpha=0.55)
        sns.kdeplot(non_mated_scores, ax=axis, label="Non-Mated", color="#ff4d4f", fill=True, alpha=0.55)
        axis.set_title("Hamming Distance Distribution")
        axis.set_xlabel("Hamming Distance")
        axis.set_ylabel("Density")
        axis.set_xlim(0.0, 0.6)
        axis.legend(loc="upper right")

        self.status_text = self.figure.text(0.01, 0.055, "", ha="left", va="bottom", fontsize=8)
        add_figure_metadata(self.figure, self.metadata)
        self.figure.tight_layout(rect=(0, 0.09, 1, 1))
        self.update_status()
        self.figure_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.figure_output_path, dpi=300, bbox_inches="tight")
        print(f"Saved HD distribution plot to {self.figure_output_path}")

        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def on_key(self, event):
        if event.key == "m":
            self.set_mode("mated")
        elif event.key == "n":
            self.set_mode("non-mated")
        elif event.key == "a":
            self.set_mode("all")

    def selected_pair_indices(self, target_hd):
        scores = self.pairwise["scores"]
        mask = self.class_mask() & (np.abs(scores - target_hd) <= self.selection_width)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            mask = self.class_mask()
            indices = np.flatnonzero(mask)
        order = np.argsort(np.abs(scores[indices] - target_hd))
        return indices[order[: self.show_pairs]]

    def on_click(self, event):
        if event.inaxes is not self.distribution_axis or event.xdata is None:
            return
        target_hd = float(event.xdata)
        if self.click_marker is not None:
            self.click_marker.remove()
        self.click_marker = self.distribution_axis.axvline(
            target_hd,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
        self.figure.canvas.draw_idle()

        pair_indices = self.selected_pair_indices(target_hd)
        if pair_indices.size == 0:
            print("No pairs available for the current selection.")
            return
        self.show_pair_window(target_hd, pair_indices)

    def image_label(self, image_index):
        if self.labels is None:
            return ""
        return f"label={self.labels[image_index]}"

    def show_pair_window(self, target_hd, pair_indices):
        rows = len(pair_indices)
        fig, axes = plt.subplots(rows, 4, figsize=(14, max(3.2, 3.1 * rows)))
        if rows == 1:
            axes = np.asarray([axes])

        for row_index, pair_index in enumerate(pair_indices):
            idx1 = int(self.pairwise["idx1"][pair_index])
            idx2 = int(self.pairwise["idx2"][pair_index])
            score = float(self.pairwise["scores"][pair_index])
            same = bool(self.pairwise["same_class"][pair_index])
            best_offset = int(self.pairwise["best_offset"][pair_index])
            direction = int(self.pairwise["direction"][pair_index])
            names = [str(self.image_names[idx1]), str(self.image_names[idx2])]
            debug_a = self.overlay_cache.get(names[0])
            debug_b = self.overlay_cache.get(names[1])
            comparison_type = "MATED COMPARISON" if same else "NON-MATED COMPARISON"
            comparison_color = "#3b5bff" if same else "#ff4d4f"

            panels = [
                (debug_a["gray"], "gray", f"A: {Path(names[0]).name}\n{self.image_label(idx1)}"),
                (debug_a["overlay"], None, "A mask overlay"),
                (debug_b["gray"], "gray", f"B: {Path(names[1]).name}\n{self.image_label(idx2)}"),
                (debug_b["overlay"], None, "B mask overlay"),
            ]
            for column, (image, cmap, title) in enumerate(panels):
                axis = axes[row_index, column]
                axis.imshow(image, cmap=cmap, vmin=0, vmax=255 if cmap == "gray" else None)
                axis.set_title(title, fontsize=9)
                axis.axis("off")

            axes[row_index, 0].set_ylabel(
                f"{comparison_type}\nHD={score:.4f}\noffset={best_offset}\ndir={direction}",
                fontsize=8,
                rotation=0,
                labelpad=48,
                va="center",
                color=comparison_color,
                fontweight="bold",
            )
            for column in range(4):
                for spine in axes[row_index, column].spines.values():
                    spine.set_visible(True)
                    spine.set_color(comparison_color)
                    spine.set_linewidth(2.0)

        selected_same = self.pairwise["same_class"][pair_indices]
        mated_count = int(np.sum(selected_same))
        non_mated_count = int(len(pair_indices) - mated_count)
        fig.suptitle(
            (
                f"Nearest comparisons at HD={target_hd:.4f} "
                f"| shown={len(pair_indices)} | mated={mated_count} | non-mated={non_mated_count} "
                f"| window=+/-{self.selection_width:g}"
            ),
            fontsize=12,
        )
        add_figure_metadata(fig, self.metadata)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))

        if self.click_output_dir is not None:
            self.click_output_dir.mkdir(parents=True, exist_ok=True)
            self.click_count += 1
            output = self.click_output_dir / f"selection_{self.click_count:03d}_hd_{target_hd:.4f}.png"
            fig.savefig(output, dpi=180, bbox_inches="tight")
            print(f"Saved clicked selection to {output}")
        plt.show(block=False)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Interactively inspect the pairwise Hamming-distance distribution. "
            "Click the distribution to see the image pairs and source mask overlays near that HD."
        )
    )
    parser.add_argument("--scores", default=None, help="Optional .npz from pairwise_iris_analysis.py --output.")
    parser.add_argument("--dataset", default="auto", choices=DATASET_CHOICES, help="Dataset format to compute.")
    parser.add_argument("--dataset-path", default=None, help="Dataset root. Required when --scores lacks dataset_path.")
    parser.add_argument("--rotation", type=int, default=21, help="Rotation count used when computing scores.")
    parser.add_argument("--filters", default=None, help="Optional Python filters file containing a 'filters' list.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--max-img-per-id", "--max-images-per-identity", dest="max_images_per_identity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection-width", type=float, default=0.005, help="HD half-window around the clicked x-value.")
    parser.add_argument("--show-pairs", type=int, default=4, help="Maximum pairs to show per click.")
    parser.add_argument("--click-output-dir", default=None, help="Optional directory to save every clicked detail view.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for saved HD plots and score caches.")
    parser.add_argument("--output-name", default="interactive_hd_distribution", help="Name shown in plot metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    if args.show_pairs < 1:
        raise ValueError("--show-pairs must be at least 1")
    if args.selection_width <= 0:
        raise ValueError("--selection-width must be positive")

    if args.scores:
        pairwise, labels, image_names, saved_dataset_path, rotation, matcher = load_pairwise_npz(args.scores)
        dataset_path = args.dataset_path or saved_dataset_path
        if dataset_path is None:
            raise ValueError("--dataset-path is required because the .npz does not contain dataset_path.")
        dataset_format = args.dataset
        rotation = args.rotation if rotation is None else rotation
        save_cache = False
    else:
        pairwise, labels, image_names, dataset_path, rotation, matcher, dataset_format = compute_pairwise(args)
        save_cache = True

    figure_output_path, score_cache_path = output_paths(args.output_dir, dataset_format, args.output_name)
    if save_cache:
        saved_cache = save_pairwise_cache(
            score_cache_path,
            pairwise,
            labels,
            image_names,
            dataset_path,
            rotation,
            matcher,
        )
        print(f"Saved interactive score cache to {saved_cache}")

    metadata = {
        "dataset": dataset_format,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "seg_path": os.environ.get("SEG_PATH"),
        "segmentation_backend": get_segmentation_backend_name(),
        "matcher": matcher,
        "rotation": rotation,
        "selection_width": args.selection_width,
        "output_name": args.output_name,
        "samples": len(image_names),
        "pairs": len(pairwise["scores"]),
    }
    viewer = InteractiveDistribution(
        pairwise=pairwise,
        labels=labels,
        image_names=image_names,
        dataset_path=dataset_path,
        metadata=metadata,
        selection_width=args.selection_width,
        show_pairs=args.show_pairs,
        figure_output_path=figure_output_path,
        click_output_dir=args.click_output_dir,
    )
    viewer.plot()


if __name__ == "__main__":
    main()
