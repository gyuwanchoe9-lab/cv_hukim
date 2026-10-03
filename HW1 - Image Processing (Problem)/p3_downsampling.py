"""Problem 3. Downsampling and aliasing.

Computer Vision (2026) -- Homework 1

Sampling a high-resolution image directly throws away every frequency the new
grid cannot represent, and those frequencies do not disappear: they reappear as
structure that was never in the scene (Lecture 2, slides 127-139). Smoothing
before sampling removes them first. This problem measures that trade-off.

The coordinate convention is the one fixed in Problem 2, and the nearest
neighbour sampler from that file is reused here.

Run:
    python p3_downsampling.py --data data --out out
"""
import argparse
import os

import numpy as np

import imutils
from p1_smoothing import create_gaussian_kernel, filtering
from p2_upsampling import upsample_nearest


# -----------------------------------------------------------------------------
# 3-a. Sampling with no prefilter
# -----------------------------------------------------------------------------
def downsample_naive(source, target_size):
    """Shrink by nearest neighbour sampling, with no smoothing.

    Args:
        source (numpy.ndarray): (old_h, old_w, 3).
        target_size (tuple): (new_h, new_w).
    Returns:
        numpy.ndarray: (new_h, new_w, 3).
    """
    # HINT: the mapping is identical to Problem 2; only the scale is now > 1.
    target = upsample_nearest(source, target_size)
    return target


# -----------------------------------------------------------------------------
# 3-b. Smooth, then sample
# -----------------------------------------------------------------------------
def downsample_prefiltered(source, target_size, sigma, ksize=None):
    """Smooth with a Gaussian and then sample on the coarse grid.

    Args:
        source (numpy.ndarray): (old_h, old_w, 3).
        target_size (tuple): (new_h, new_w).
        sigma (float): standard deviation of the prefilter, in input pixels.
        ksize (int): kernel size. None selects 2 * ceil(3 * sigma) + 1.
    Returns:
        numpy.ndarray: (new_h, new_w, 3), float64.
    """
    if ksize is None:
        ksize = 2 * int(np.ceil(3.0 * sigma)) + 1
    # HINT: reuse create_gaussian_kernel, filtering and downsample_naive.
    #       Use replicate padding so the border does not darken.
    kernel = create_gaussian_kernel(ksize, sigma)
    smoothed = filtering(source, kernel, mode='replicate')
    target = downsample_naive(smoothed, target_size)
    return target


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def main(args):
    out = imutils.ensure_dir(args.out)
    source = imutils.imread_bgr(os.path.join(args.data, f'{args.image}.png'))
    old_h, old_w = source.shape[:2]
    new_h, new_w = args.target_size
    factor = old_w / new_w
    print(f'input : {args.image}.png {old_w}x{old_h}  ->  {new_w}x{new_h}'
          f'  (1/{factor:.0f})')

    naive = downsample_naive(source, (new_h, new_w))
    imutils.imsave(os.path.join(out, 'p3_naive.png'), naive)

    # 3-b, 3-d. sigma sweep
    print('\n[3-b, 3-d] prefilter sweep')
    print(f'  {"sigma":>5s}  {"ksize":>5s}  {"high-freq energy":>17s}  {"| - naive|":>11s}')
    results = {}
    energy_naive = high_frequency_energy(naive)
    print(f'  {"none":>5s}  {"-":>5s}  {energy_naive:17.3f}  {0.0:11.3f}')
    for sigma in args.sigmas:
        small = downsample_prefiltered(source, (new_h, new_w), sigma)
        imutils.imsave(os.path.join(out, f'p3_prefiltered_s{sigma:g}.png'), small)
        results[sigma] = small
        energy = high_frequency_energy(small)
        gap = float(np.abs(small - naive.astype(np.float64)).mean())
        print(f'  {sigma:5.1f}  {2 * int(np.ceil(3 * sigma)) + 1:5d}  '
              f'{energy:17.3f}  {gap:11.3f}')

    # 3-c. a window where aliasing is visible, enlarged for the report
    top, left = args.crop
    size = args.crop_size
    reference = downsample_prefiltered(source, (new_h, new_w), args.sigmas[-1])
    panel = imutils.hstack([
        imutils.zoom(imutils.crop(naive, top, left, size, size), args.zoom),
        imutils.zoom(imutils.crop(reference, top, left, size, size), args.zoom),
    ])
    imutils.imsave(os.path.join(out, 'p3_alias_crop.png'), panel)
    print(f'\n  crop window: top={top} left={left} size={size}x{size}, '
          f'zoomed x{args.zoom}')
    print(f'  left panel: no prefilter,  right panel: sigma={args.sigmas[-1]:g}')
    print(f'\nwrote results to {out}/')


def high_frequency_energy(image):
    """Mean squared finite difference; a crude measure of high-frequency content.

    Aliasing injects structure that is not in the scene, so a naively sampled
    image usually scores higher here than a prefiltered one.
    """
    array = np.asarray(image, dtype=np.float64)
    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    return float((dx ** 2).mean() + (dy ** 2).mean())


def parse_args():
    parser = argparse.ArgumentParser(description='Problem 3. Downsampling')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--image', type=str, default='sonoma')
    parser.add_argument('--target_size', type=int, nargs=2, default=(135, 240),
                        metavar=('NEW_H', 'NEW_W'))
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=(0.5, 1.0, 2.0))
    parser.add_argument('--crop', type=int, nargs=2, default=(40, 120),
                        metavar=('TOP', 'LEFT'))
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--zoom', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
