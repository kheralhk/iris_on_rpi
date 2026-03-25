# iris.py

import cv2 as cv
import numpy as np
import subprocess
import tempfile
import time
from analysis.profiling import timeit, span
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
tmp = Path(tempfile.gettempdir())
WAHET_BINARY = Path(__file__).resolve().parent / "wahet"

@timeit
def hamming_distance(a,b,mask1, mask2):
    diff = np.bitwise_xor(a,b)
    mask = np.bitwise_and(mask1, mask2)
    total = np.sum(np.bitwise_and(diff, mask))
    n = np.sum(mask)
    if n == 0:
        return 2.0
    return total/n


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
def apply_filter(iris, filter_real, filter_imag, x, y, mask=None):
    """Filters should have 0 DC"""
    h, w = filter_real.shape
    patch = get_patch(iris, x, y, w, h)
    result_real = np.sum(filter_real * patch)
    result_imag = np.sum(filter_imag * patch)
    if mask is not None:
        patch = get_patch(mask, x, y, w, h)
        mask_bit = np.all(patch == 255)
        return result_real+result_imag*1j, mask_bit
    return result_real+result_imag*1j, True
   
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

@timeit
def get_patch(img, x, y, w, h):
    img_h, img_w = img.shape
    patch = np.zeros((h, w), dtype=img.dtype)

    # Wrap x and generate x indices
    x_indices = (np.arange(x - w // 2, x - w // 2 + w) % img_w)

    # Compute y range and valid bounds
    y_start = y - h // 2
    y_stop = y_start + h

    # Determine valid range in image
    y_valid_start = max(y_start, 0)
    y_valid_stop = min(y_stop, img_h)

    # Determine where to place the valid rows in the patch
    patch_y_start = y_valid_start - y_start
    patch_y_stop = patch_y_start + (y_valid_stop - y_valid_start)

    if patch_y_stop > patch_y_start:  # Only fill if there's something valid
        patch[patch_y_start:patch_y_stop, :] = img[y_valid_start:y_valid_stop, :][:, x_indices]

    return patch

@timeit
def apply_filter_to_iris(iris, filter_real, filter_imag, stride, start_position, mask=None):
    x_stride, y_stride = stride
    x_start, y_start = start_position
    iris_h, iris_w = iris.shape
    filter_h, filter_w = filter_real.shape
    num_x = iris_w // x_stride
    num_y = iris_h // y_stride

    x_half = filter_w // 2
    y_half = filter_h // 2
    y_bottom = filter_h - y_half - 1
    x_right = filter_w - x_half - 1
    x_positions = (x_start + x_stride * np.arange(num_x)) % iris_w
    y_positions = y_start + y_stride * np.arange(num_y)
    extra_top = max(0, -int(y_positions.min()))
    extra_bottom = max(0, int(y_positions.max()) - (iris_h - 1))

    padded_iris = np.pad(
        iris,
        ((y_half + extra_top, y_bottom + extra_bottom), (0, 0)),
        mode="constant",
    )
    wrapped_iris = np.concatenate(
        (
            padded_iris[:, -x_half:] if x_half else padded_iris[:, :0],
            padded_iris,
            padded_iris[:, :x_right] if x_right else padded_iris[:, :0],
        ),
        axis=1,
    )

    iris_windows = sliding_window_view(wrapped_iris, (filter_h, filter_w))
    sampled_iris = iris_windows[y_positions + extra_top][:, x_positions]

    result_real = np.einsum("yxij,ij->yx", sampled_iris, filter_real, optimize=True)
    result_imag = np.einsum("yxij,ij->yx", sampled_iris, filter_imag, optimize=True)
    results = (result_real + result_imag * 1j).T.reshape(-1)

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
    sampled_mask = mask_windows[y_positions + extra_top][:, x_positions]
    mask_bits = np.all(sampled_mask == 255, axis=(2, 3)).T.reshape(-1)
    return results, mask_bits


@timeit
def get_iris_band(img):
    cv.imwrite(tmp/"input.png", img)
    with span("wahet"):
        subprocess.run([str(WAHET_BINARY), "-i", tmp/"input.png", "-o", tmp/"output.png", "-m", tmp/"mask.png"], capture_output=True
        )
    image = cv.imread(tmp/"output.png", cv.IMREAD_GRAYSCALE)
    mask = cv.imread(tmp/"mask.png", cv.IMREAD_GRAYSCALE)
    return image, mask

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

    @timeit
    def __call__(self, iris1, iris2, mask1, mask2, rotation=6, offset=0):
        bits_1, mask_1, _ = self.get_iris_code(iris1, mask1)
        return self.compare_iris_code_and_iris(iris2, bits_1, mask2, mask_1, rotation=rotation, offset=offset)
    
    @timeit
    def get_iris_code(self, iris, mask=None, offset=0):
        bit_chunks = []
        filter_chunks = []
        mask_chunks = []

        for i, (filter_real, filter_imag) in enumerate(self._filters):
            t0 = time.perf_counter()
            start_x, start_y = self._filter_settings[i]["start_position"]
            start_pos = (start_x + offset, start_y)
            result, mask_bit_list = apply_filter_to_iris(
                iris,
                filter_real,
                filter_imag,
                self._filter_settings[i]["stride"],
                start_pos,
                mask,
            )
            t1 = time.perf_counter()
            
            new_bits, mask_bits = complex_to_bits(result, mask_bit_list)
            filter_ids = np.full(new_bits.shape, i, dtype=np.uint8)
            bit_chunks.append(new_bits)
            filter_chunks.append(filter_ids)
            mask_chunks.append(mask_bits)
            
        bits = np.concatenate(bit_chunks)
        filters = np.concatenate(filter_chunks)
        mask_bits = np.concatenate(mask_chunks)
        return bits, mask_bits, filters
    
    @timeit
    def compare_iris_code_and_iris(self, iris, iris_code, iris_mask, iris_code_mask, rotation=None, offset=0):
        if rotation is None:
            bits, mask, _ = self.get_iris_code(iris, iris_mask, offset=offset)
            return (hamming_distance(bits, iris_code, mask, iris_code_mask), 0)
        scores = np.empty(rotation)
        for i in range(rotation):
            bits, mask, _ = self.get_iris_code(iris, iris_mask, offset=i-rotation//2)
            scores[i] = hamming_distance(bits, iris_code, mask, iris_code_mask)
        return (np.min(scores), np.argmin(scores)-rotation//2)
