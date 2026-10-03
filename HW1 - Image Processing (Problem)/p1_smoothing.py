"""Problem 1. Linear filtering and smoothing.

Computer Vision (2026) -- Homework 1

Implement a normalised Gaussian kernel, two padding modes, and
correlation-based linear filtering. Then observe how the kernel size, the
standard deviation, and the padding mode each change the result.

Run:
    python p1_smoothing.py --data data --out out
"""
import argparse
import os

import numpy as np

import imutils

PAD_MODES = ('zero', 'replicate')


# -----------------------------------------------------------------------------
# 1-a. Gaussian kernel
# -----------------------------------------------------------------------------
def create_gaussian_kernel(ksize, sigma):
    """Return a (ksize, ksize) Gaussian kernel whose entries sum to one.

    The kernel is centred at c = (ksize - 1) / 2:

        w(s, t) = exp( -((s - c)^2 + (t - c)^2) / (2 * sigma^2) )

    and is then divided by its own sum, so that filtering a constant image
    leaves it unchanged.

    Args:
        ksize (int): odd kernel size.
        sigma (float): standard deviation in pixels, sigma > 0.
    Returns:
        numpy.ndarray: (ksize, ksize), float64, sums to 1.
    """
    if ksize % 2 == 0:
        raise ValueError('ksize must be odd')
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    # HINT: numpy.arange and numpy.meshgrid, or the outer product of two 1-D kernels.
    c = (ksize - 1) / 2.0
    axis = np.arange(ksize, dtype=np.float64) - c
    ss, tt = np.meshgrid(axis, axis, indexing='ij')
    kernel = np.exp(-(ss ** 2 + tt ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


# -----------------------------------------------------------------------------
# 1-b. Padding
# -----------------------------------------------------------------------------
def pad(source, pad_size, mode='zero'):
    """Pad the spatial axes of an image by pad_size on every side.

    Two modes, following Lecture 2 (slides 17-18):
        'zero'       ... 0 0 0 | a b c d | 0 0 0
        'replicate'  ... a a a | a b c d | d d d

    Implement this with array slicing. Do not call numpy.pad.

    Args:
        source (numpy.ndarray): (H, W) or (H, W, C).
        pad_size (int): number of pixels added to each side, pad_size >= 0.
        mode (str): one of PAD_MODES.
    Returns:
        numpy.ndarray: (H + 2p, W + 2p) or (H + 2p, W + 2p, C), float64.
    """
    if mode not in PAD_MODES:
        raise ValueError(f'unknown padding mode: {mode}')
    source = np.asarray(source, dtype=np.float64)
    if pad_size == 0:
        return source.copy()
    height, width = source.shape[:2]

    shape = (height + 2 * pad_size, width + 2 * pad_size) + source.shape[2:]
    target = np.zeros(shape, dtype=np.float64)
    target[pad_size:pad_size + height, pad_size:pad_size + width] = source

    # HINT: fill the top and bottom rows first; padding the columns afterwards
    #       from the already filled rows fills the corners for free.
    if mode == 'replicate':
        p = pad_size
        target[:p, p:p + width] = source[0:1]                   # top
        target[p + height:, p:p + width] = source[-1:]          # bottom
        target[:, :p] = target[:, p:p + 1]                      # left (+ corners)
        target[:, p + width:] = target[:, p + width - 1:p + width]  # right (+ corners)
    return target


# -----------------------------------------------------------------------------
# 1-c. Linear filtering
# -----------------------------------------------------------------------------
def filtering(source, kernel, mode='zero'):
    """Correlate source with kernel; the output keeps the input shape.

        g(x, y) = sum_s sum_t w(s, t) * f(x + s, y + t)

    This is correlation, not convolution: the kernel is not flipped. The two
    agree here because a Gaussian kernel is symmetric, but Problem 5 uses
    kernels for which the distinction matters, so keep it in mind.

    Args:
        source (numpy.ndarray): (H, W) or (H, W, C).
        kernel (numpy.ndarray): (k, k), k odd.
        mode (str): padding mode passed to pad().
    Returns:
        numpy.ndarray: same shape as source, float64.
    """
    source = np.asarray(source, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError('kernel must be square and two-dimensional')
    ksize = kernel.shape[0]
    pad_size = ksize // 2
    padded = pad(source, pad_size, mode)
    height, width = source.shape[:2]

    # HINT: a double loop over the k*k kernel taps is enough. Each step adds a
    #       shifted slice of the padded image, so the image itself is never
    #       looped over in Python.
    target = np.zeros(source.shape, dtype=np.float64)
    for s in range(ksize):
        for t in range(ksize):
            target += kernel[s, t] * padded[s:s + height, t:t + width]
    return target


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def main(args):
    out = imutils.ensure_dir(args.out)
    source = imutils.imread_bgr(os.path.join(args.data, f'{args.image}.png'))
    print(f'input : {args.image}.png {source.shape[1]}x{source.shape[0]}')

    # 1-d. kernel size sweep at a fixed sigma
    detail = imutils.crop(source, args.detail[0], args.detail[1],
                          args.detail_size, args.detail_size)
    print('\n[1-d] kernel size sweep (sigma = 4.0)')
    panels = [detail]
    for ksize in (3, 7, 11):
        kernel = create_gaussian_kernel(ksize, 4.0)
        smoothed = filtering(source, kernel, mode='replicate')
        path = imutils.imsave(
            os.path.join(out, f'p1_smooth_k{ksize:02d}_s4.png'), smoothed)
        panels.append(imutils.crop(smoothed, args.detail[0], args.detail[1],
                                   args.detail_size, args.detail_size))
        print(f'  ksize={ksize:2d}  kernel sum={kernel.sum():.6f}  '
              f'centre weight={kernel[ksize // 2, ksize // 2]:.6f}  '
              f'-> {os.path.basename(path)}')
    imutils.imsave(os.path.join(out, 'p1_ksize_sweep.png'),
                   imutils.hstack([imutils.zoom(p, args.zoom) for p in panels]))
    print('  p1_ksize_sweep.png: original, ksize 3, 7, 11 at sigma = 4  '
          f'({args.detail_size}x{args.detail_size} crop, zoomed x{args.zoom})')

    # 1-d. sigma sweep at a fixed kernel size
    print('\n[1-d] sigma sweep (ksize = 11)')
    panels = [detail]
    for sigma in (1.0, 4.0, 9.0):
        kernel = create_gaussian_kernel(11, sigma)
        smoothed = filtering(source, kernel, mode='replicate')
        path = imutils.imsave(
            os.path.join(out, f'p1_smooth_k11_s{int(sigma)}.png'), smoothed)
        panels.append(imutils.crop(smoothed, args.detail[0], args.detail[1],
                                   args.detail_size, args.detail_size))
        # fraction of the ideal Gaussian mass that an 11x11 window actually holds
        wide_size = max(11, 2 * int(np.ceil(4 * sigma)) + 1)
        axis = np.arange(wide_size, dtype=np.float64) - (wide_size - 1) / 2.0
        xx, yy = np.meshgrid(axis, axis)
        wide = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        wide /= wide.sum()
        centre = wide_size // 2
        kept = wide[centre - 5:centre + 6, centre - 5:centre + 6].sum()
        print(f'  sigma={sigma:4.1f}  mass kept by an 11x11 window={kept:.4f}  '
              f'-> {os.path.basename(path)}')
    imutils.imsave(os.path.join(out, 'p1_sigma_sweep.png'),
                   imutils.hstack([imutils.zoom(p, args.zoom) for p in panels]))
    print('  p1_sigma_sweep.png: original, sigma 1, 4, 9 at ksize = 11  '
          f'({args.detail_size}x{args.detail_size} crop, zoomed x{args.zoom})')

    # 1-e. padding comparison on a crop, with a kernel wide enough to show it
    print('\n[1-e] padding comparison (ksize = 25, sigma = 9.0)')
    patch = imutils.crop(source, args.crop[0], args.crop[1], 512, 512)
    kernel = create_gaussian_kernel(25, 9.0)
    panels, borders = [], []
    for mode in PAD_MODES:
        smoothed = filtering(patch, kernel, mode=mode)
        imutils.imsave(os.path.join(out, f'p1_pad_{mode}.png'), smoothed)
        panels.append(imutils.crop(smoothed, 0, 0, 80, 80))
        borders.append((mode, float(np.abs(smoothed[:12] - patch[:12]).mean())))
    panels.insert(0, imutils.crop(patch, 0, 0, 80, 80))
    imutils.imsave(os.path.join(out, 'p1_pad_corner.png'),
                   imutils.hstack([imutils.zoom(p, 3) for p in panels]))
    print('  p1_pad_corner.png: original, zero, replicate '
          '(top-left 80x80 corner, zoomed x3)')
    for mode, error in borders:
        print(f'  {mode:<10s} mean |smoothed - original| over the top 12 rows = {error:7.3f}')

    print(f'\nwrote results to {out}/')


def parse_args():
    parser = argparse.ArgumentParser(description='Problem 1. Smoothing')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--image', type=str, default='sonoma')
    parser.add_argument('--crop', type=int, nargs=2, default=(0, 0),
                        metavar=('TOP', 'LEFT'),
                        help='corner used for the padding comparison')
    parser.add_argument('--detail', type=int, nargs=2, default=(400, 480),
                        metavar=('TOP', 'LEFT'),
                        help='detailed region used for the sweep figures')
    parser.add_argument('--detail_size', type=int, default=240)
    parser.add_argument('--zoom', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
