"""Problem 2. Upsampling by nearest neighbour and bilinear interpolation.

Computer Vision (2026) -- Homework 1

Both methods share one coordinate convention. For an output pixel index d on
one axis, with scale = old_size / new_size, the continuous source coordinate is

    u = (d + 0.5) * scale - 0.5

which aligns the centres of the two grids (Lecture 2, slides 89-93). Source
coordinates are clamped to [0, old_size - 1] at the border. Using u = d * scale
instead shifts the image by half a pixel; the whole problem depends on getting
this right, so do not change it.

Run:
    python p2_upsampling.py --data data --out out
"""
import argparse
import os

import numpy as np

import imutils


def source_coordinates(old_size, new_size):
    """Return the (new_size,) array of continuous source coordinates.

    Args:
        old_size (int): input length along one axis.
        new_size (int): output length along the same axis.
    Returns:
        numpy.ndarray: (new_size,), float64, values in [-0.5, old_size - 0.5].
    """
    scale = old_size / new_size
    return (np.arange(new_size, dtype=np.float64) + 0.5) * scale - 0.5


# -----------------------------------------------------------------------------
# 2-a. Nearest neighbour
# -----------------------------------------------------------------------------
def upsample_nearest(source, target_size):
    """Resize by taking the nearest source pixel.

    Round the continuous coordinate u to the nearest integer, with halves going
    up: index = floor(u + 0.5). Clamp the result to the valid index range.

    Args:
        source (numpy.ndarray): (old_h, old_w, 3).
        target_size (tuple): (new_h, new_w).
    Returns:
        numpy.ndarray: (new_h, new_w, 3), same dtype as source.
    """
    source = np.asarray(source)
    old_h, old_w = source.shape[:2]
    new_h, new_w = int(target_size[0]), int(target_size[1])
    # HINT: build the row and column index arrays, then use numpy fancy indexing
    #       source[rows[:, None], cols[None, :]].
    rows = np.clip(np.floor(source_coordinates(old_h, new_h) + 0.5).astype(int), 0, old_h - 1)
    cols = np.clip(np.floor(source_coordinates(old_w, new_w) + 0.5).astype(int), 0, old_w - 1)
    target = source[rows[:, None], cols[None, :]]
    return target


# -----------------------------------------------------------------------------
# 2-b. Bilinear
# -----------------------------------------------------------------------------
def upsample_bilinear(source, target_size):
    """Resize by bilinear interpolation of the four surrounding pixels.

    With u the continuous source coordinate, u0 = floor(u), u1 = u0 + 1 and
    a = u - u0, the one-dimensional interpolation is (1 - a) * f(u0) + a * f(u1);
    the two-dimensional case applies this along one axis and then the other
    (Lecture 2, slides 96-99). Clamp u0 and u1 to the valid index range, which
    is the same as replicating the border.

    Args:
        source (numpy.ndarray): (old_h, old_w, 3).
        target_size (tuple): (new_h, new_w).
    Returns:
        numpy.ndarray: (new_h, new_w, 3), float64.
    """
    source = np.asarray(source, dtype=np.float64)
    old_h, old_w = source.shape[:2]
    new_h, new_w = int(target_size[0]), int(target_size[1])
    # HINT: compute the four weighted contributions and add them. Keeping the
    #       row and column weights as (new_h, 1, 1) and (1, new_w, 1) arrays
    #       lets numpy broadcast over the colour channels.
    u = np.clip(source_coordinates(old_h, new_h), 0, old_h - 1)
    v = np.clip(source_coordinates(old_w, new_w), 0, old_w - 1)
    r0 = np.floor(u).astype(int)
    c0 = np.floor(v).astype(int)
    r1 = np.minimum(r0 + 1, old_h - 1)
    c1 = np.minimum(c0 + 1, old_w - 1)
    a = (u - r0)[:, None, None]      # row weights    (new_h, 1, 1)
    b = (v - c0)[None, :, None]      # column weights (1, new_w, 1)
    if source.ndim == 2:
        a, b = a[..., 0], b[..., 0]
    target = ((1 - a) * (1 - b) * source[r0[:, None], c0[None, :]]
              + (1 - a) * b * source[r0[:, None], c1[None, :]]
              + a * (1 - b) * source[r1[:, None], c0[None, :]]
              + a * b * source[r1[:, None], c1[None, :]])
    return target


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def main(args):
    out = imutils.ensure_dir(args.out)
    source = imutils.imread_bgr(os.path.join(args.data, f'{args.image}.png'))
    old_h, old_w = source.shape[:2]
    new_h, new_w = args.target_size
    print(f'input : {args.image}.png {old_w}x{old_h}  ->  {new_w}x{new_h}'
          f'  (x{new_w / old_w:.0f})')

    nearest = upsample_nearest(source, (new_h, new_w))
    bilinear = upsample_bilinear(source, (new_h, new_w))
    imutils.imsave(os.path.join(out, 'p2_upsample_nearest.png'), nearest)
    imutils.imsave(os.path.join(out, 'p2_upsample_bilinear.png'), bilinear)

    # 2-c. the same window from both results, side by side
    top, left = args.crop
    size = args.crop_size
    panel = imutils.hstack([
        imutils.crop(nearest, top, left, size, size),
        imutils.crop(bilinear, top, left, size, size),
    ])
    imutils.imsave(os.path.join(out, 'p2_upsample_crop.png'), panel)

    # a number to put next to the two crops
    difference = np.abs(nearest.astype(np.float64) - bilinear)
    print(f'\n  mean |nearest - bilinear| = {difference.mean():7.3f}')
    print(f'  max  |nearest - bilinear| = {difference.max():7.3f}')
    print(f'  crop window: top={top} left={left} size={size}x{size}')
    print(f'\nwrote results to {out}/')


def parse_args():
    parser = argparse.ArgumentParser(description='Problem 2. Upsampling')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--image', type=str, default='cat')
    parser.add_argument('--target_size', type=int, nargs=2, default=(2048, 2048),
                        metavar=('NEW_H', 'NEW_W'))
    parser.add_argument('--crop', type=int, nargs=2, default=(880, 900),
                        metavar=('TOP', 'LEFT'))
    parser.add_argument('--crop_size', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
