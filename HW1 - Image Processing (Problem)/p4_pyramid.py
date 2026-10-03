"""Problem 4. Gaussian pyramid.

Computer Vision (2026) -- Homework 1

Each level of a Gaussian pyramid is the previous level smoothed and then
subsampled by two (Lecture 2, slides 141-151). Because the smoothing is applied
once per level, the coarsest level has been band-limited repeatedly, which is
what makes it different from a single direct subsampling of the original.

Run:
    python p4_pyramid.py --data data --out out
"""
import argparse
import os

import numpy as np

import imutils
from p1_smoothing import create_gaussian_kernel, filtering
from p3_downsampling import downsample_naive


# -----------------------------------------------------------------------------
# 4-a. Gaussian pyramid
# -----------------------------------------------------------------------------
def build_gaussian_pyramid(source, num_levels, ksize=5, sigma=1.0):
    """Build a Gaussian pyramid.

    Level 0 is the input itself. Level l + 1 is level l smoothed with a
    Gaussian and then subsampled by taking every second row and column.

    Args:
        source (numpy.ndarray): (H, W, 3).
        num_levels (int): total number of levels, including level 0.
        ksize (int): smoothing kernel size.
        sigma (float): smoothing standard deviation.
    Returns:
        list: num_levels arrays, from the finest to the coarsest scale.
    """
    levels = [np.asarray(source, dtype=np.float64)]
    # HINT: reuse create_gaussian_kernel and filtering. Subsampling by two is
    #       just slicing: array[::2, ::2].
    kernel = create_gaussian_kernel(ksize, sigma)
    for _ in range(num_levels - 1):
        smoothed = filtering(levels[-1], kernel, mode='replicate')
        levels.append(smoothed[::2, ::2])
    return levels


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def main(args):
    out = imutils.ensure_dir(args.out)
    source = imutils.imread_bgr(os.path.join(args.data, f'{args.image}.png'))
    print(f'input : {args.image}.png {source.shape[1]}x{source.shape[0]}')

    levels = build_gaussian_pyramid(source, args.num_levels,
                                    args.kernel_size, args.sigma)

    # 4-b. one file per level, plus a size table
    print(f'\n[4-b] pyramid (ksize={args.kernel_size}, sigma={args.sigma})')
    print(f'  {"level":>5s}  {"size":>12s}  {"pixels":>10s}')
    for index, level in enumerate(levels):
        height, width = level.shape[:2]
        imutils.imsave(os.path.join(out, f'p4_level_{index}.png'), level)
        print(f'  {index:5d}  {width:5d}x{height:<6d}  {height * width:10d}')

    # 4-c. coarsest level against the naive downsampling of Problem 3
    coarsest = levels[-1]
    naive = downsample_naive(source, coarsest.shape[:2])
    panel = imutils.hstack([
        imutils.zoom(naive, args.zoom),
        imutils.zoom(coarsest, args.zoom),
    ])
    imutils.imsave(os.path.join(out, 'p4_vs_naive.png'), panel)

    difference = np.abs(coarsest - naive.astype(np.float64))
    print(f'\n[4-c] coarsest level {coarsest.shape[1]}x{coarsest.shape[0]} '
          f'versus one-step naive downsampling')
    print(f'  mean |pyramid - naive| = {difference.mean():7.3f}')
    print(f'  max  |pyramid - naive| = {difference.max():7.3f}')
    print(f'  left panel: naive,  right panel: pyramid  (p4_vs_naive.png)')

    print(f'\nwrote results to {out}/')


def parse_args():
    parser = argparse.ArgumentParser(description='Problem 4. Gaussian pyramid')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--image', type=str, default='sonoma')
    parser.add_argument('--num_levels', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--zoom', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
