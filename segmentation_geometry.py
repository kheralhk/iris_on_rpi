from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np


DEFAULT_BAND_SHAPE = (64, 512)


@dataclass(frozen=True)
class EllipseBoundary:
    center_x: float
    center_y: float
    axis_x: float
    axis_y: float
    angle_radians: float

    def sample(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        local_x = self.axis_x * cos_theta
        local_y = self.axis_y * sin_theta
        cos_angle = np.cos(self.angle_radians)
        sin_angle = np.sin(self.angle_radians)
        x = self.center_x + local_x * cos_angle - local_y * sin_angle
        y = self.center_y + local_x * sin_angle + local_y * cos_angle
        return x, y


@dataclass(frozen=True)
class PolarBoundary:
    center_x: float
    center_y: float
    radii: np.ndarray

    def sample(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        tau = 2.0 * np.pi
        wrapped = np.mod(theta, tau)
        base_theta = np.linspace(0.0, tau, len(self.radii), endpoint=False, dtype=np.float32)
        extended_theta = np.concatenate((base_theta, [tau]), axis=0)
        extended_radii = np.concatenate((self.radii, [self.radii[0]]), axis=0)
        radii = np.interp(wrapped, extended_theta, extended_radii)
        x = self.center_x + radii * np.cos(wrapped)
        y = self.center_y + radii * np.sin(wrapped)
        return x, y


def _largest_component(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    if not np.any(mask):
        return mask
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def clean_component_mask(mask, kernel_size=5):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if not np.any(mask):
        return mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
    return _largest_component(opened)


def contour_center(mask):
    component = clean_component_mask(mask)
    if not np.any(component):
        raise ValueError("Cannot compute a center from an empty mask.")

    moments = cv.moments(component)
    if moments["m00"] > 0:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])

    ys, xs = np.nonzero(component)
    return float(xs.mean()), float(ys.mean())


def fit_boundary_from_mask(mask, prefer_ellipse=True):
    component = clean_component_mask(mask)
    if not np.any(component):
        raise ValueError("Cannot fit a boundary to an empty mask.")

    contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Failed to extract contours from mask.")
    contour = max(contours, key=cv.contourArea)

    if prefer_ellipse and len(contour) >= 5:
        (center_x, center_y), (diameter_x, diameter_y), angle_degrees = cv.fitEllipse(contour)
        axis_x = max(float(diameter_x) / 2.0, 1.0)
        axis_y = max(float(diameter_y) / 2.0, 1.0)
        return EllipseBoundary(
            center_x=float(center_x),
            center_y=float(center_y),
            axis_x=axis_x,
            axis_y=axis_y,
            angle_radians=np.deg2rad(angle_degrees),
        )

    (center_x, center_y), radius = cv.minEnclosingCircle(contour)
    radius = max(float(radius), 1.0)
    return EllipseBoundary(
        center_x=float(center_x),
        center_y=float(center_y),
        axis_x=radius,
        axis_y=radius,
        angle_radians=0.0,
    )


def _periodic_smooth(values, kernel_size):
    kernel_size = max(int(kernel_size), 1)
    if kernel_size <= 1:
        return values.astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    pad = kernel_size // 2
    extended = np.concatenate((values[-pad:], values, values[:pad]), axis=0)
    smoothed = np.convolve(extended, kernel, mode="same")[pad : pad + len(values)]
    return smoothed.astype(np.float32)


def fit_polar_boundary_from_mask(
    mask,
    center,
    num_angles=DEFAULT_BAND_SHAPE[1],
    smooth_kernel=9,
    fallback_to_ellipse=True,
):
    component = clean_component_mask(mask)
    if not np.any(component):
        raise ValueError("Cannot fit a boundary to an empty mask.")

    contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Failed to extract contours from mask.")
    contour = max(contours, key=cv.contourArea).reshape(-1, 2).astype(np.float32)

    center_x, center_y = center
    dx = contour[:, 0] - center_x
    dy = contour[:, 1] - center_y
    angles = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    radii = np.hypot(dx, dy)

    angle_to_radius = np.full(num_angles, np.nan, dtype=np.float32)
    angle_indices = np.floor(angles / (2.0 * np.pi) * num_angles).astype(np.int32) % num_angles
    for index, radius in zip(angle_indices, radii):
        if np.isnan(angle_to_radius[index]) or radius > angle_to_radius[index]:
            angle_to_radius[index] = radius

    valid = ~np.isnan(angle_to_radius)
    if valid.sum() < max(num_angles // 4, 16):
        if not fallback_to_ellipse:
            raise ValueError("Not enough contour coverage to fit a polar boundary.")
        ellipse = fit_boundary_from_mask(component, prefer_ellipse=True)
        theta = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False, dtype=np.float32)
        sample_x, sample_y = ellipse.sample(theta)
        angle_to_radius = np.hypot(sample_x - center_x, sample_y - center_y).astype(np.float32)
        return PolarBoundary(center_x=float(center_x), center_y=float(center_y), radii=angle_to_radius)

    valid_indices = np.flatnonzero(valid)
    extended_x = np.concatenate(
        (
            valid_indices.astype(np.float32) - num_angles,
            valid_indices.astype(np.float32),
            valid_indices.astype(np.float32) + num_angles,
        )
    )
    extended_y = np.concatenate(
        (
            angle_to_radius[valid_indices],
            angle_to_radius[valid_indices],
            angle_to_radius[valid_indices],
        )
    )
    filled = np.interp(np.arange(num_angles, dtype=np.float32), extended_x, extended_y)
    smoothed = _periodic_smooth(filled, smooth_kernel)
    return PolarBoundary(center_x=float(center_x), center_y=float(center_y), radii=smoothed)


def build_valid_source_mask(iris_mask, pupil_mask, occlusion_mask=None, source_image=None, oversat_threshold=254):
    iris_mask = clean_component_mask(iris_mask)
    pupil_mask = clean_component_mask(pupil_mask)
    valid = iris_mask.astype(bool) & ~pupil_mask.astype(bool)
    if occlusion_mask is not None:
        valid &= ~(np.asarray(occlusion_mask) > 0)
    if source_image is not None:
        valid &= np.asarray(source_image) < oversat_threshold
    return (valid.astype(np.uint8) * 255)


def normalize_iris_from_boundaries(
    image,
    pupil_boundary,
    iris_boundary,
    valid_source_mask,
    band_shape=DEFAULT_BAND_SHAPE,
):
    if image.ndim != 2:
        raise ValueError("Expected a grayscale source image.")

    band_height, band_width = band_shape
    theta = np.linspace(0.0, 2.0 * np.pi, band_width, endpoint=False, dtype=np.float32)
    radial = np.linspace(0.0, 1.0, band_height, dtype=np.float32)[:, None]

    pupil_x, pupil_y = pupil_boundary.sample(theta)
    iris_x, iris_y = iris_boundary.sample(theta)

    map_x = (1.0 - radial) * pupil_x[None, :] + radial * iris_x[None, :]
    map_y = (1.0 - radial) * pupil_y[None, :] + radial * iris_y[None, :]

    band = cv.remap(
        image.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT_101,
    )
    sampled_mask = cv.remap(
        valid_source_mask.astype(np.uint8),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )

    return np.clip(band, 0, 255).astype(np.uint8), (sampled_mask > 0).astype(np.uint8) * 255


def semantic_masks_to_band(
    image,
    iris_mask,
    pupil_mask,
    occlusion_mask=None,
    band_shape=DEFAULT_BAND_SHAPE,
    prefer_ellipse=True,
):
    if occlusion_mask is not None:
        occlusion_mask = clean_component_mask(occlusion_mask, kernel_size=3)
        occlusion_mask = cv.dilate(
            occlusion_mask,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

    valid_source_mask = build_valid_source_mask(
        iris_mask,
        pupil_mask,
        occlusion_mask,
        source_image=image,
        oversat_threshold=254,
    )
    pupil_ellipse = fit_boundary_from_mask(pupil_mask, prefer_ellipse=prefer_ellipse)
    center = (pupil_ellipse.center_x, pupil_ellipse.center_y)
    pupil_boundary = fit_polar_boundary_from_mask(
        pupil_mask,
        center=center,
        num_angles=band_shape[1],
        smooth_kernel=7,
        fallback_to_ellipse=True,
    )
    iris_boundary = fit_polar_boundary_from_mask(
        iris_mask,
        center=center,
        num_angles=band_shape[1],
        smooth_kernel=17,
        fallback_to_ellipse=True,
    )

    if np.mean(iris_boundary.radii) <= np.mean(pupil_boundary.radii):
        raise ValueError("Iris boundary must be larger than pupil boundary.")

    return normalize_iris_from_boundaries(
        image=image,
        pupil_boundary=pupil_boundary,
        iris_boundary=iris_boundary,
        valid_source_mask=valid_source_mask,
        band_shape=band_shape,
    )
