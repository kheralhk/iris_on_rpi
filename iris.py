# iris.py

import cv2 as cv
import numpy as np
import os
import subprocess
import tempfile
from profiling import timeit
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from segmentation_geometry import annulus_mask_to_band, semantic_masks_to_band, DEFAULT_BAND_SHAPE
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEGMENTATION_BACKEND = os.environ.get(
    "SEG_BACKEND",
    os.environ.get(
        "IRIS_SEG_BACKEND",
        os.environ.get("IRIS_SEGMENTATION_BACKEND", "onnx"),
    ),
).strip().lower()


def _resolve_segmentation_model_path():
    configured = os.environ.get(
        "SEG_PATH",
        os.environ.get(
            "IRIS_SEG_PATH",
            os.environ.get(
                "IRIS_SEGMENTATION_ONNX_PATH",
                os.environ.get(
                    "IRIS_SEGMENTATION_PATH",
                    os.environ.get("IRIS_UNET_ONNX_PATH"),
                ),
            ),
        ),
    )
    if configured is not None:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        cwd_candidate = path.resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (PROJECT_ROOT / path).resolve()

    default_candidates = [
        PROJECT_ROOT / "models" / "iris_semseg_upp_scse_mobilenetv2.onnx",
        PROJECT_ROOT / "models" / "upp_scse_mobilenetv2.onnx",
        PROJECT_ROOT / "models" / "mysegmenter.onnx",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate
    return default_candidates[0]


def _resolve_wahet_binary_path():
    configured = os.environ.get(
        "WAHET_PATH",
        os.environ.get("IRIS_WAHET_PATH"),
    )
    if configured is not None:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
        cwd_candidate = path.resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (PROJECT_ROOT / path).resolve()
    return PROJECT_ROOT / "wahet"


def _normalize_segmentation_backend_name(name):
    backend_name = str(name).strip().lower()
    if backend_name in {"onnx", "model", "custom", "unet", "annulus"}:
        return "onnx"
    if backend_name == "wahet":
        return "wahet"
    raise ValueError(f"Unsupported segmentation backend: {name}")


def get_segmentation_backend_name(backend=None):
    return _normalize_segmentation_backend_name(backend or DEFAULT_SEGMENTATION_BACKEND)


UNET_ONNX_PATH = _resolve_segmentation_model_path()
WAHET_BINARY = _resolve_wahet_binary_path()
UNET_INPUT_SIZE = (
    int(os.environ.get("IRIS_UNET_INPUT_WIDTH", 480)),
    int(os.environ.get("IRIS_UNET_INPUT_HEIGHT", 640)),
)
UNET_THRESHOLD = float(os.environ.get("IRIS_UNET_THRESHOLD", 0.5))
UNET_BAND_SHAPE = (
    int(os.environ.get("IRIS_UNET_BAND_HEIGHT", DEFAULT_BAND_SHAPE[0])),
    int(os.environ.get("IRIS_UNET_BAND_WIDTH", DEFAULT_BAND_SHAPE[1])),
)
OUTER_IRIS_WEIGHT = 0.25
VERTICAL_PAD_MODE = os.environ.get("IRIS_VERTICAL_PAD_MODE", "reflect").strip().lower()
MASKED_WINDOW_POLICY = os.environ.get("IRIS_MASKED_WINDOW_POLICY", "strict").strip().lower()
MASK_FILL_MODE = os.environ.get("IRIS_MASK_FILL_MODE", "zero").strip().lower()
_UNET_NET = None

@timeit
def hamming_distance(a,b,mask1, mask2, weights=None):
    diff = np.bitwise_xor(a,b)
    mask = np.bitwise_and(mask1, mask2)
    if weights is None:
        total = np.sum(np.bitwise_and(diff, mask))
        n = np.sum(mask)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        weighted_mask = mask.astype(np.float32) * weights
        total = np.sum(np.bitwise_and(diff, mask).astype(np.float32) * weights)
        n = np.sum(weighted_mask)
    if n == 0:
        return 2.0
    return total/n

@timeit
def hamming_distances(a, b, masks_a, mask_b, weights=None):
    diff = np.bitwise_xor(a, b)
    mask = np.bitwise_and(masks_a, mask_b)
    scores = np.full(a.shape[0], 2.0, dtype=np.float64)
    if weights is None:
        total = np.sum(np.bitwise_and(diff, mask), axis=1)
        n = np.sum(mask, axis=1)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1:
            weights = weights[None, :]
        weighted_mask = mask.astype(np.float32) * weights
        total = np.sum(np.bitwise_and(diff, mask).astype(np.float32) * weights, axis=1)
        n = np.sum(weighted_mask, axis=1)
    valid = n > 0
    scores[valid] = total[valid] / n[valid]
    return scores


@timeit
def complex_gabor_kernel(size, sigma, theta, lambd, psi, gamma):
    """Create a complex Gabor kernel."""
    y_size, x_size = size
    x_half_size = x_size // 2
    y_half_size = y_size // 2
    y, x = np.meshgrid(np.linspace(-y_half_size, y_half_size, y_size),
                       np.linspace(-x_half_size, x_half_size, x_size))
    
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    # _x_theta = x * np.cos(0) + y * np.sin(0)
    _x_theta = x
    # _y_theta = -x * np.sin(0) + y * np.cos(0)
    _y_theta = y

    # gaussian = np.exp(-0.5 * (_x_theta**2 + (gamma**2) * _y_theta**2) / sigma**2)
    gaussian = np.exp(-0.5 * (x**2 + (gamma**2) * y**2) / sigma**2)

    complex_sinusoid = np.exp(1j * (2 * np.pi * x_theta / lambd + psi))

    gabor = gaussian * complex_sinusoid
    return gabor

@timeit
def complex_to_bits(z, mask_bit_list):
    real = (z.real >= 0).astype(np.bool)
    imag = (z.imag >= 0).astype(np.bool)
    result = np.empty(z.size*2, dtype=np.bool)
    mask_bits = np.empty(z.size*2, dtype=np.bool)
    result[0::2] = real
    result[1::2] = imag 
    mask_bits[0::2] = mask_bit_list
    mask_bits[1::2] = mask_bit_list
    return result, mask_bits


def _pad_iris_vertically(iris, pad_top, pad_bottom):
    if VERTICAL_PAD_MODE in {"reflect", "mirror"}:
        return np.pad(iris, ((pad_top, pad_bottom), (0, 0)), mode="reflect")
    if VERTICAL_PAD_MODE in {"duplicate", "edge", "replicate"}:
        return np.pad(iris, ((pad_top, pad_bottom), (0, 0)), mode="edge")
    if VERTICAL_PAD_MODE in {"circular", "wrap"}:
        return np.pad(iris, ((pad_top, pad_bottom), (0, 0)), mode="wrap")
    if VERTICAL_PAD_MODE in {"zero", "constant"}:
        return np.pad(iris, ((pad_top, pad_bottom), (0, 0)), mode="constant")
    raise ValueError(
        "Unsupported IRIS_VERTICAL_PAD_MODE. Use one of: reflect, duplicate, circular, zero."
    )


def _fill_masked_pixels_in_band(iris, mask, mode):
    mode = mode.strip().lower()
    if mode in {"zero", "constant"}:
        filled = iris.copy()
        filled[mask != 255] = 0
        return filled

    valid = mask == 255
    filled = iris.copy()
    height, width = iris.shape

    for x in range(width):
        col = filled[:, x]
        valid_col = valid[:, x]
        invalid_rows = np.flatnonzero(~valid_col)
        if invalid_rows.size == 0:
            continue

        valid_rows = np.flatnonzero(valid_col)
        if valid_rows.size == 0:
            col[:] = 0
            filled[:, x] = col
            continue

        first_valid = int(valid_rows[0])
        last_valid = int(valid_rows[-1])

        if mode in {"duplicate", "edge", "replicate"}:
            for y in invalid_rows:
                nearest = valid_rows[np.argmin(np.abs(valid_rows - y))]
                col[y] = col[nearest]
            filled[:, x] = col
            continue

        if mode in {"mirror", "reflect"}:
            for y in invalid_rows:
                if y < first_valid:
                    reflected = first_valid + (first_valid - y - 1)
                    reflected = min(reflected, last_valid)
                    col[y] = col[reflected]
                elif y > last_valid:
                    reflected = last_valid - (y - last_valid - 1)
                    reflected = max(reflected, first_valid)
                    col[y] = col[reflected]
                else:
                    nearest = valid_rows[np.argmin(np.abs(valid_rows - y))]
                    col[y] = col[nearest]
            filled[:, x] = col
            continue

        raise ValueError(
            "Unsupported IRIS_MASK_FILL_MODE. Use one of: zero, duplicate, mirror."
        )

    return filled

def _apply_filter_grid_batch(iris, filter_real, filter_imag, stride, start_positions, mask=None):
    x_stride, y_stride = stride
    iris_h, iris_w = iris.shape
    filter_h, filter_w = filter_real.shape
    num_x = iris_w // x_stride

    start_positions = np.asarray(start_positions, dtype=np.int64)
    x_starts = start_positions[:, 0]
    y_starts = start_positions[:, 1]
    if not np.all(y_starts == y_starts[0]):
        raise ValueError("All start positions must share the same y coordinate.")

    x_half = filter_w // 2
    y_half = filter_h // 2
    y_bottom = filter_h - y_half - 1
    x_right = filter_w - x_half - 1
    num_y = iris_h // y_stride
    y_positions = y_starts[0] + y_stride * np.arange(num_y)
    extra_top = max(0, -int(y_positions.min()))
    extra_bottom = max(0, int(y_positions.max()) - (iris_h - 1))

    iris_for_filtering = iris
    if mask is not None and MASKED_WINDOW_POLICY == "fill":
        iris_for_filtering = _fill_masked_pixels_in_band(iris, mask, MASK_FILL_MODE)

    padded_iris = _pad_iris_vertically(iris_for_filtering, y_half + extra_top, y_bottom + extra_bottom)
    wrapped_iris = np.concatenate(
        (
            padded_iris[:, -x_half:] if x_half else padded_iris[:, :0],
            padded_iris,
            padded_iris[:, :x_right] if x_right else padded_iris[:, :0],
        ),
        axis=1,
    )

    iris_windows = sliding_window_view(wrapped_iris, (filter_h, filter_w))
    sampled_rows = iris_windows[y_positions + extra_top]
    x_positions = (x_starts[:, None] + x_stride * np.arange(num_x)) % iris_w
    sampled_iris = sampled_rows[:, x_positions, :, :]

    result_real = np.einsum("yoxij,ij->yox", sampled_iris, filter_real, optimize=True)
    result_imag = np.einsum("yoxij,ij->yox", sampled_iris, filter_imag, optimize=True)
    results = (result_real + result_imag * 1j).transpose(1, 2, 0).reshape(len(start_positions), -1)

    if mask is None:
        mask_bits = np.ones(results.shape, dtype=np.bool)
        return results, mask_bits

    padded_mask = np.pad(
        mask,
        ((y_half + extra_top, y_bottom + extra_bottom), (0, 0)),
        mode="constant",
    )
    wrapped_mask = np.concatenate(
        (
            padded_mask[:, -x_half:] if x_half else padded_mask[:, :0],
            padded_mask,
            padded_mask[:, :x_right] if x_right else padded_mask[:, :0],
        ),
        axis=1,
    )
    mask_windows = sliding_window_view(wrapped_mask, (filter_h, filter_w))
    sampled_mask_rows = mask_windows[y_positions + extra_top]
    sampled_mask = sampled_mask_rows[:, x_positions, :, :]
    if MASKED_WINDOW_POLICY == "fill":
        mask_bits = np.any(sampled_mask == 255, axis=(3, 4)).transpose(1, 2, 0).reshape(len(start_positions), -1)
    else:
        mask_bits = np.all(sampled_mask == 255, axis=(3, 4)).transpose(1, 2, 0).reshape(len(start_positions), -1)
    return results, mask_bits

def _resolve_dnn_backend(name):
    mapping = {
        "opencv": cv.dnn.DNN_BACKEND_OPENCV,
    }
    if hasattr(cv.dnn, "DNN_BACKEND_CUDA"):
        mapping["cuda"] = cv.dnn.DNN_BACKEND_CUDA
    return mapping.get(name)


def _resolve_dnn_target(name):
    mapping = {
        "cpu": cv.dnn.DNN_TARGET_CPU,
    }
    if hasattr(cv.dnn, "DNN_TARGET_CUDA"):
        mapping["cuda"] = cv.dnn.DNN_TARGET_CUDA
    if hasattr(cv.dnn, "DNN_TARGET_CUDA_FP16"):
        mapping["cuda_fp16"] = cv.dnn.DNN_TARGET_CUDA_FP16
    return mapping.get(name)


def _get_unet_net():
    global _UNET_NET
    if _UNET_NET is not None:
        return _UNET_NET
    if not UNET_ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Segmentation ONNX model not found at '{UNET_ONNX_PATH}'. "
            "Set IRIS_SEGMENTATION_ONNX_PATH or place the model there."
        )
    net = cv.dnn.readNetFromONNX(str(UNET_ONNX_PATH))
    backend_name = os.environ.get("IRIS_UNET_DNN_BACKEND", "opencv").strip().lower()
    target_name = os.environ.get("IRIS_UNET_DNN_TARGET", "cpu").strip().lower()
    backend = _resolve_dnn_backend(backend_name)
    target = _resolve_dnn_target(target_name)
    if backend is not None:
        net.setPreferableBackend(backend)
    if target is not None:
        net.setPreferableTarget(target)
    _UNET_NET = net
    return net


def _sigmoid_if_needed(output):
    output = np.asarray(output, dtype=np.float32)
    if output.min() < 0.0 or output.max() > 1.0:
        return 1.0 / (1.0 + np.exp(-output))
    return output


def _prepare_segmentation_gray(img):
    return img if img.ndim == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def _forward_segmentation_model(img):
    source_gray = _prepare_segmentation_gray(img)
    resized = cv.resize(source_gray, UNET_INPUT_SIZE, interpolation=cv.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = np.repeat(resized[:, :, None], 3, axis=2)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = ((rgb - mean) / std).transpose(2, 0, 1)[None, :, :, :]

    net = _get_unet_net()
    net.setInput(normalized)
    output = net.forward()
    return source_gray, output


def predict_unet_masks(img):
    source_gray, output = _forward_segmentation_model(img)
    if output.ndim != 4 or output.shape[1] < 3:
        raise RuntimeError(f"Unexpected multiclass segmentation output shape: {output.shape}")

    probabilities = _sigmoid_if_needed(output[0])
    image_h, image_w = source_gray.shape
    resized_probs = np.stack(
        [
            cv.resize(probabilities[index], (image_w, image_h), interpolation=cv.INTER_LINEAR)
            for index in range(probabilities.shape[0])
        ],
        axis=0,
    )

    # Worldcoin's published model card lists classes: eyeball, iris, pupil, eyelashes.
    iris_mask = resized_probs[1] >= UNET_THRESHOLD
    pupil_mask = resized_probs[2] >= UNET_THRESHOLD
    eyelash_mask = resized_probs[3] >= UNET_THRESHOLD if resized_probs.shape[0] > 3 else np.zeros_like(iris_mask)
    return source_gray, iris_mask, pupil_mask, eyelash_mask


def predict_annulus_mask(img):
    source_gray, output = _forward_segmentation_model(img)
    if output.ndim != 4:
        raise RuntimeError(f"Unexpected annulus model output shape: {output.shape}")

    image_h, image_w = source_gray.shape
    if output.shape[1] == 1:
        annulus_prob = _sigmoid_if_needed(output[0, 0])
    elif output.shape[1] == 2:
        logits = output[0].astype(np.float32)
        logits = logits - np.max(logits, axis=0, keepdims=True)
        exp_logits = np.exp(logits)
        annulus_prob = exp_logits[1] / np.sum(exp_logits, axis=0)
    else:
        raise RuntimeError(
            f"Binary annulus model must output 1 or 2 channels, got {output.shape[1]}."
        )

    annulus_prob = cv.resize(annulus_prob, (image_w, image_h), interpolation=cv.INTER_LINEAR)
    annulus_mask = annulus_prob >= UNET_THRESHOLD
    return source_gray, annulus_mask


@timeit
def _segment_with_wahet(img):
    if not WAHET_BINARY.exists():
        raise FileNotFoundError(
            f"Wahet executable not found at '{WAHET_BINARY}'. "
            "Set WAHET_PATH or place the binary there."
        )

    with tempfile.TemporaryDirectory(prefix="wahet_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.png"
        output_path = temp_dir / "output.png"
        mask_path = temp_dir / "mask.png"

        if not cv.imwrite(str(input_path), img):
            raise RuntimeError(f"Failed to write temporary wahet input to '{input_path}'.")

        result = subprocess.run(
            [str(WAHET_BINARY), "-i", str(input_path), "-o", str(output_path), "-m", str(mask_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            detail = stderr or stdout or "wahet exited with a non-zero status"
            raise RuntimeError(f"Wahet segmentation failed: {detail}")

        image = cv.imread(str(output_path), cv.IMREAD_GRAYSCALE)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError("Wahet did not produce readable output image and mask.")
        return image, mask


@timeit
def _segment_with_unet(img):
    source_gray, output = _forward_segmentation_model(img)
    if output.ndim != 4:
        raise RuntimeError(f"Unexpected segmentation output shape: {output.shape}")

    if output.shape[1] in {1, 2}:
        image_h, image_w = source_gray.shape
        if output.shape[1] == 1:
            annulus_prob = _sigmoid_if_needed(output[0, 0])
        else:
            logits = output[0].astype(np.float32)
            logits = logits - np.max(logits, axis=0, keepdims=True)
            exp_logits = np.exp(logits)
            annulus_prob = exp_logits[1] / np.sum(exp_logits, axis=0)

        annulus_prob = cv.resize(annulus_prob, (image_w, image_h), interpolation=cv.INTER_LINEAR)
        annulus_mask = annulus_prob >= UNET_THRESHOLD
        return annulus_mask_to_band(
            source_gray,
            annulus_mask=annulus_mask,
            band_shape=UNET_BAND_SHAPE,
            prefer_ellipse=True,
            source_image_for_saturation=source_gray,
        )

    probabilities = _sigmoid_if_needed(output[0])
    image_h, image_w = source_gray.shape
    resized_probs = np.stack(
        [
            cv.resize(probabilities[index], (image_w, image_h), interpolation=cv.INTER_LINEAR)
            for index in range(probabilities.shape[0])
        ],
        axis=0,
    )
    iris_mask = resized_probs[1] >= UNET_THRESHOLD
    pupil_mask = resized_probs[2] >= UNET_THRESHOLD
    eyelash_mask = resized_probs[3] >= UNET_THRESHOLD if resized_probs.shape[0] > 3 else np.zeros_like(iris_mask)
    return semantic_masks_to_band(
        source_gray,
        iris_mask=iris_mask,
        pupil_mask=pupil_mask,
        occlusion_mask=eyelash_mask,
        band_shape=UNET_BAND_SHAPE,
        prefer_ellipse=True,
    )


@timeit
def get_iris_band(img, backend=None):
    backend_name = get_segmentation_backend_name(backend)
    if backend_name == "wahet":
        return _segment_with_wahet(img)
    return _segment_with_unet(img)

class IrisClassifier():
    def __init__(self, filters) -> None:
        self.init_filters(filters)
        
    @timeit
    def init_filters(self, filters):
        self._filters = [] 
        for filter_settings in filters:
            filter = complex_gabor_kernel(**filter_settings["filter"])
            real_filter = np.real(filter)
            imag_filter = np.imag(filter)
            real_filter = real_filter - np.mean(real_filter)
            imag_filter = imag_filter - np.mean(imag_filter)
            self._filters.append((real_filter, imag_filter))
        self._filter_settings = filters
        self._bit_weight_cache = {}

    def get_bit_weights(self, iris_shape):
        iris_shape = tuple(int(v) for v in iris_shape)
        cached = self._bit_weight_cache.get(iris_shape)
        if cached is not None:
            return cached

        iris_h, iris_w = iris_shape
        weight_chunks = []
        for filter_settings in self._filter_settings:
            x_stride, y_stride = filter_settings["stride"]
            start_x, start_y = filter_settings["start_position"]
            num_x = iris_w // x_stride
            num_y = iris_h // y_stride
            y_positions = start_y + y_stride * np.arange(num_y, dtype=np.float32)
            normalized_radius = np.clip(y_positions / max(iris_h - 1, 1), 0.0, 1.0)
            row_weights = 1.0 - (1.0 - OUTER_IRIS_WEIGHT) * normalized_radius
            location_weights = np.tile(row_weights.astype(np.float32), num_x)
            bit_weights = np.empty(location_weights.size * 2, dtype=np.float32)
            bit_weights[0::2] = location_weights
            bit_weights[1::2] = location_weights
            weight_chunks.append(bit_weights)

        weights = np.concatenate(weight_chunks, axis=0)
        self._bit_weight_cache[iris_shape] = weights
        return weights

    @timeit
    def __call__(self, iris1, iris2, mask1, mask2, rotation=6, offset=0):
        bits_1, mask_1, _ = self.get_iris_code(iris1, mask1)
        return self.compare_iris_code_and_iris(iris2, bits_1, mask2, mask_1, rotation=rotation, offset=offset)
    
    def _encode_iris_offsets(self, iris, mask=None, offsets=(0,)):
        offsets = np.asarray(offsets, dtype=np.int64)
        bit_chunks = []
        filter_chunks = []
        mask_chunks = []

        for i, (filter_real, filter_imag) in enumerate(self._filters):
            start_x, start_y = self._filter_settings[i]["start_position"]
            start_positions = np.column_stack(
                (
                    start_x + offsets,
                    np.full(offsets.shape, start_y, dtype=np.int64),
                )
            )
            results, mask_bit_lists = _apply_filter_grid_batch(
                iris,
                filter_real,
                filter_imag,
                self._filter_settings[i]["stride"],
                start_positions,
                mask,
            )

            filter_bits = []
            filter_masks = []
            for result, mask_bit_list in zip(results, mask_bit_lists):
                new_bits, mask_bits = complex_to_bits(result, mask_bit_list)
                filter_bits.append(new_bits)
                filter_masks.append(mask_bits)

            bit_chunks.append(np.stack(filter_bits, axis=0))
            mask_chunks.append(np.stack(filter_masks, axis=0))
            filter_ids = np.full((len(offsets), filter_bits[0].shape[0]), i, dtype=np.uint8)
            filter_chunks.append(filter_ids)

        bits = np.concatenate(bit_chunks, axis=1)
        filters = np.concatenate(filter_chunks, axis=1)
        mask_bits = np.concatenate(mask_chunks, axis=1)
        return bits, mask_bits, filters

    @timeit
    def get_iris_code(self, iris, mask=None, offset=0):
        bits, mask_bits, filters = self._encode_iris_offsets(iris, mask, offsets=(offset,))
        return bits[0], mask_bits[0], filters[0]

    @timeit
    def get_iris_codes(self, iris, mask=None, offsets=(0,)):
        bits, mask_bits, filters = self._encode_iris_offsets(iris, mask, offsets=offsets)
        return bits, mask_bits, filters
    
    @timeit
    def compare_iris_code_and_iris(self, iris, iris_code, iris_mask, iris_code_mask, rotation=None, offset=0):
        bit_weights = self.get_bit_weights(iris.shape)
        if rotation is None:
            bits, mask, _ = self.get_iris_code(iris, iris_mask, offset=offset)
            return (hamming_distance(bits, iris_code, mask, iris_code_mask, weights=bit_weights), 0)
        offsets = np.arange(rotation) - rotation // 2
        bits, masks, _ = self.get_iris_codes(iris, iris_mask, offsets=offsets)
        scores = hamming_distances(bits, iris_code, masks, iris_code_mask, weights=bit_weights)
        return (np.min(scores), np.argmin(scores)-rotation//2)
