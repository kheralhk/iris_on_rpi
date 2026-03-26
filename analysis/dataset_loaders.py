# dataset_loaders.py

from pathlib import Path

import cv2 as cv
import numpy as np


CASIA_V1_PATH = Path(
    "/Users/krist/Documents/project/casia/CASIA Version.1/CASIA Iris Image Database (version 1.0)"
)
CASIA_V3_INTERVAL_PATH = Path(
    "/Users/krist/Documents/project/casia/CASIA-IrisV3/CASIA-IrisV3-Interval"
)
CASIA_V3_LAMP_PATH = Path(
    "/Users/krist/Documents/project/casia/CASIA-IrisV3/CASIA-IrisV3-Lamp"
)
CASIA_V3_TWINS_PATH = Path(
    "/Users/krist/Documents/project/casia/CASIA-IrisV3/CASIA-IrisV3-Twins"
)


def _load_images_with_labels(dataset_dir, image_paths, label_builder):
    images = []
    labels = []
    names = []

    for image_path in image_paths:
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to load image '{image_path}'.")

        images.append(image)
        labels.append(label_builder(image_path))
        names.append(str(image_path.relative_to(dataset_dir)))

    return images, np.array(labels), np.array(names)


def load_casia_v1(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.bmp"))
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA V1 images were found. Expected files like XXX/S/XXX_S_Y.bmp."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_v3_interval(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV3 Interval images were found. Expected files like XXX/L/S1XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_v3_lamp(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV3 Lamp images were found. Expected files like XXX/L/S2XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def load_casia_v3_twins(dataset_path):
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    image_paths = sorted(path for path in dataset_dir.glob("*/*/*.jpg") if path.name.lower() != "thumbs.db")
    if not image_paths:
        raise FileNotFoundError(
            "No CASIA-IrisV3 Twins images were found. Expected files like XX/1L/S3XXXL01.jpg."
        )

    return _load_images_with_labels(
        dataset_dir,
        image_paths,
        lambda image_path: f"{image_path.parent.parent.name}_{image_path.parent.name}",
    )


def resolve_dataset(dataset_path, dataset_format):
    if dataset_path:
        resolved_path = Path(dataset_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {resolved_path}")

        if dataset_format == "auto":
            if any(resolved_path.glob("*/*/*.bmp")):
                return resolved_path, "casia-v1"
            if any(resolved_path.glob("*/*/*.jpg")):
                name = resolved_path.name.lower()
                if "interval" in name:
                    return resolved_path, "casia-v3-interval"
                if "lamp" in name:
                    return resolved_path, "casia-v3-lamp"
                if "twins" in name:
                    return resolved_path, "casia-v3-twins"
                return resolved_path, "casia-v3-interval"
            raise FileNotFoundError(
                f"Could not infer dataset format from '{resolved_path}'. "
                "Use --dataset-format explicitly."
            )

        return resolved_path, dataset_format

    if dataset_format in ("auto", "casia-v1") and CASIA_V1_PATH.exists():
        return CASIA_V1_PATH, "casia-v1"
    if dataset_format in ("auto", "casia-v3-interval") and CASIA_V3_INTERVAL_PATH.exists():
        return CASIA_V3_INTERVAL_PATH, "casia-v3-interval"
    if dataset_format in ("auto", "casia-v3-lamp") and CASIA_V3_LAMP_PATH.exists():
        return CASIA_V3_LAMP_PATH, "casia-v3-lamp"
    if dataset_format in ("auto", "casia-v3-twins") and CASIA_V3_TWINS_PATH.exists():
        return CASIA_V3_TWINS_PATH, "casia-v3-twins"

    raise FileNotFoundError(
        "Could not find a default dataset path. "
        "Pass --dataset-path explicitly or place the dataset in one of these locations:\n"
        f"{CASIA_V1_PATH}\n"
        f"{CASIA_V3_INTERVAL_PATH}\n"
        f"{CASIA_V3_LAMP_PATH}\n"
        f"{CASIA_V3_TWINS_PATH}"
    )


def load_dataset(dataset_path, dataset_format):
    if dataset_format == "casia-v1":
        return load_casia_v1(dataset_path)
    if dataset_format == "casia-v3-interval":
        return load_casia_v3_interval(dataset_path)
    if dataset_format == "casia-v3-lamp":
        return load_casia_v3_lamp(dataset_path)
    if dataset_format == "casia-v3-twins":
        return load_casia_v3_twins(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")
