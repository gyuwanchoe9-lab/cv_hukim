"""Helper utilities for Homework 1.

This file is provided. Do not modify it.
It only handles file input/output and figure preparation; every image
processing operation that the homework asks about is implemented by you.
"""
import os

import cv2
import numpy as np

__all__ = [
    'ensure_dir', 'imread_bgr', 'imread_gray', 'imsave',
    'crop', 'zoom', 'hstack', 'orientation_to_bgr',
]


def ensure_dir(path):
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def imread_bgr(path):
    """Read a colour image as a (H, W, 3) uint8 BGR array."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'cannot read image: {path}')
    return image


def imread_gray(path):
    """Read an image as a (H, W) float64 array in [0, 255]."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'cannot read image: {path}')
    return image.astype(np.float64)


def imsave(path, image):
    """Write an array as 8-bit PNG. Values are clipped to [0, 255]."""
    array = np.asarray(image, dtype=np.float64)
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    ensure_dir(os.path.dirname(path) or '.')
    if not cv2.imwrite(path, array):
        raise IOError(f'cannot write image: {path}')
    return path


def crop(image, top, left, height, width):
    """Return image[top:top+height, left:left+width]."""
    return image[top:top + height, left:left + width]


def zoom(image, factor):
    """Enlarge by an integer factor with pixel replication, for figures only."""
    array = np.asarray(image)
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def hstack(images, gap=8, value=255):
    """Place images side by side with a constant-valued gap between them."""
    images = [np.asarray(i, dtype=np.float64) for i in images]
    images = [i[:, :, None].repeat(3, axis=2) if i.ndim == 2 else i for i in images]
    height = max(i.shape[0] for i in images)
    parts = []
    for index, image in enumerate(images):
        pad_rows = height - image.shape[0]
        if pad_rows:
            image = np.pad(image, ((0, pad_rows), (0, 0), (0, 0)), constant_values=value)
        if index:
            parts.append(np.full((height, gap, 3), value, dtype=np.float64))
        parts.append(image)
    return np.concatenate(parts, axis=1)


def orientation_to_bgr(magnitude, orientation):
    """Colour-code gradient orientation by hue and magnitude by value.

    Args:
        magnitude (numpy.ndarray): (H, W), non-negative.
        orientation (numpy.ndarray): (H, W), radians in [-pi, pi].
    Returns:
        numpy.ndarray: (H, W, 3) uint8 BGR.
    """
    hue = ((np.rad2deg(orientation) % 180.0) / 180.0 * 179.0).astype(np.uint8)
    peak = magnitude.max()
    value = np.zeros_like(hue) if peak <= 0 else \
        np.clip(magnitude / peak * 255.0, 0, 255).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), value], axis=2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
