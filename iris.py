import cv2 as cv
import numpy as np
import numba as nb
import subprocess
import tempfile
from pathlib import Path
tmp = Path(tempfile.gettempdir())

@nb.njit
def hamming_distance(a,b,mask1, mask2):
    diff = np.bitwise_xor(a,b)
    mask = np.bitwise_and(mask1, mask2)
    total = np.sum(np.bitwise_and(diff, mask))
    n = np.sum(mask)
    if n == 0:
        return 2.0
    return total/n

def get_patches(img):
    height = img.shape[0]
    patches = []
    width = img.shape[1]
    for i in range(4):
        x0 = height//4*i
        x1 = height//4*(i+1)
        for j in range(32):
            y0 = width//32*j
            y1 = width//32*(j+1)
            patches.append(img[x0:x1, y0:y1])
    return patches

def get_filters(sigma=2, gamma=1):
    kernels = []
    angles = [np.pi/4, -np.pi/4, 0, np.pi/2]
    periods = [4, 8]
    for angle in angles:
        for period in periods:
            kernel = complex_gabor_kernel((15,15), sigma, angle, period, 0, gamma)
            kernels.append(kernel)
    return kernels

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

@nb.njit
def apply_filter(iris, filter, x, y, mask=None):
    """Filters should have 0 DC"""
    # w, h = filter.shape
    h, w = filter.shape
    patch = get_patch(iris, x, y, w, h)
    filter_real = np.real(filter)
    filter_imag = np.imag(filter)
    result_real = np.sum(filter_real * patch)
    result_imag = np.sum(filter_imag * patch)
    if mask is not None:
        patch = get_patch(mask, x, y, w, h)
        mask_bit = np.all(patch == 255)
        return result_real+result_imag*1j, mask_bit
    return result_real+result_imag*1j, True
   
@nb.njit
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

@nb.njit
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

@nb.njit
def apply_filter_to_iris(iris, filter, stride, start_position, mask=None):
    x_stride, y_stride = stride
    x_start, y_start = start_position
    h, w = iris.shape
    results = np.zeros(w//x_stride*h//y_stride, dtype=np.complex128)
    mask_bits = np.zeros(w//x_stride*h//y_stride, dtype=np.bool)
    for i in range(w//x_stride):
        for j in range(h//y_stride):
            x = x_start+x_stride*i 
            y = y_start+y_stride*j
            result, mask_bit = apply_filter(iris, filter, x, y, mask)
            results[i*h//y_stride+j] = result
            mask_bits[i*h//y_stride+j] = mask_bit
    return results, mask_bits

def get_iris_code(img):
    patches = get_patches(img)
    iris_code = ""
    filters = get_filters(2, 0.5)
    for patch in patches:
        for j in range(len(filters)):
            filterr = np.real(filters[j])
            filterr = filterr - np.mean(filterr)
            filteri = np.imag(filters[j])
            
            resultr = np.sum(patch[:15, :15]*filterr)
            resulti = np.sum(patch[:15, :15]*filteri)
            
            s0 = "1" if resultr >= 0 else "0"
            s1 = "1" if resulti >= 0 else "0"
            iris_code = iris_code + s0 + s1
    return iris_code

def scale_iris_band_vertical(img, mask, offset):
    if offset > 0:
        iris_img = cv.resize(img[:-offset,:], (512,64), interpolation=cv.INTER_LANCZOS4)
        iris_mask = cv.resize(mask[:-offset,:], (512,64), interpolation=cv.INTER_NEAREST)
    elif offset < 0:
        iris_img = cv.resize(img[-offset:,:], (512, 64), interpolation=cv.INTER_LANCZOS4)
        iris_mask = cv.resize(mask[-offset:,:], (512,64), interpolation=cv.INTER_NEAREST)
    else:
        return img, mask
    return iris_img, iris_mask

def get_iris_code(iris, _filters, settings, mask=None, offset=0):
    bits = np.array([], dtype=np.bool)
    filters = np.array([], dtype=np.uint8)
    mask_bits = np.array([], dtype=np.bool)
    for i, filter in enumerate(_filters):
        x, y = settings[i]["start_position"]
        result, mask_bit_list = apply_filter_to_iris(iris, filter, settings[i]["stride"], (x+offset,y), mask)
        new_bits, mask_bit = complex_to_bits(result, mask_bit_list)
        filter = np.ones_like(new_bits)*i
        bits = np.concat([bits,new_bits])
        filters = np.concat([filters, filter])
        mask_bits = np.concat([mask_bits, mask_bit])
    return bits, mask_bits, filters

def get_iris_band(img):
    cv.imwrite(tmp/"input.png", img)
    subprocess.run(["./wahet", "-i", tmp/"input.png", "-o", tmp/"output.png", "-m", tmp/"mask.png"], capture_output=True)
    image = cv.imread(tmp/"output.png", cv.IMREAD_GRAYSCALE)
    mask = cv.imread(tmp/"mask.png", cv.IMREAD_GRAYSCALE)
    return image, mask

class IrisClassifier():
    def __init__(self, filters) -> None:
        self.init_filters(filters)
        
    def init_filters(self, filters):
        self._filters = [] 
        for filter_settings in filters:
            filter = complex_gabor_kernel(**filter_settings["filter"])
            real_filter = np.real(filter)
            imag_filter = np.imag(filter)
            real_filter = real_filter - np.mean(real_filter)
            imag_filter = imag_filter - np.mean(imag_filter)
            filter = real_filter + imag_filter*1j
            self._filters.append(filter)
        self._filter_settings = filters

    def __call__(self, iris1, iris2, mask1, mask2, rotation=6, offset=0):
        bits_1, mask_1, _ = self.get_iris_code(iris1, mask1)
        return self.compare_iris_code_and_iris(iris2, bits_1, mask2, mask_1, rotation=rotation, offset=offset)

    def get_iris_code(self, iris, mask=None, offset=0):
        bits = np.array([], dtype=np.bool)
        filters = np.array([], dtype=np.uint8)
        mask_bits = np.array([], dtype=np.bool)
        for i, filter in enumerate(self._filters):
            x, y = self._filter_settings[i]["start_position"]
            result, mask_bit_list = apply_filter_to_iris(iris, filter, self._filter_settings[i]["stride"], (x+offset,y), mask)
            new_bits, mask_bit = complex_to_bits(result, mask_bit_list)
            filter = np.ones_like(new_bits)*i
            bits = np.concat([bits,new_bits])
            filters = np.concat([filters, filter])
            mask_bits = np.concat([mask_bits, mask_bit])
        return bits, mask_bits, filters
    
    def compare_iris_code_and_iris(self, iris, iris_code, iris_mask, iris_code_mask, rotation=None, offset=0):
        if rotation is None:
            bits, mask, _ = self.get_iris_code(iris, iris_mask, offset=offset)
            return (hamming_distance(bits, iris_code, mask, iris_code_mask), 0)
        scores = np.empty(rotation)
        for i in range(rotation):
            bits, mask, _ = self.get_iris_code(iris, iris_mask, offset=i-rotation//2)
            scores[i] = hamming_distance(bits, iris_code, mask, iris_code_mask)
        return (np.min(scores), np.argmin(scores)-rotation//2)
