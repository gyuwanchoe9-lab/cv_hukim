"""Problem 5. Edge detection.

Computer Vision (2026) -- Homework 1

An edge detector is three decisions: how to differentiate, how much to smooth
first, and where to threshold. This problem walks through all three, ending in
a complete Canny detector (Lecture 2, slides 48-75).

Everything here works on a single-channel image of type float64.

Run:
    python p5_edges.py --data data --out out
"""
import argparse
import os
from collections import deque

import numpy as np

import imutils
from p1_smoothing import create_gaussian_kernel, filtering


# -----------------------------------------------------------------------------
# 5-a. Sobel gradient
# -----------------------------------------------------------------------------
def sobel_kernels():
    """Return the 3x3 Sobel kernels (Kx, Ky) used with filtering().

    filtering() correlates, so these are written in correlation form: Kx must
    give a positive response where intensity increases to the right, and Ky
    where it increases downwards.

    Returns:
        tuple: two (3, 3) float64 arrays.
    """
    # HINT: the separable form is [1, 2, 1]^T [-1, 0, 1] for Kx.
    kx = np.outer([1.0, 2.0, 1.0], [-1.0, 0.0, 1.0])
    ky = kx.T.copy()
    return kx, ky


def compute_gradient(image, kernels=None):
    """Return (gx, gy, magnitude, orientation).

    magnitude is the L2 norm sqrt(gx^2 + gy^2) and orientation is
    arctan2(gy, gx) in radians.

    Args:
        image (numpy.ndarray): (H, W), float64.
        kernels (tuple): (Kx, Ky). None uses sobel_kernels().
    Returns:
        tuple: four (H, W) float64 arrays.
    """
    kx, ky = sobel_kernels() if kernels is None else kernels
    # HINT: filtering(image, kx, mode='replicate').
    gx = filtering(image, kx, mode='replicate')
    gy = filtering(image, ky, mode='replicate')
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx)
    return gx, gy, magnitude, orientation


# -----------------------------------------------------------------------------
# 5-b. Derivative of Gaussian
# -----------------------------------------------------------------------------
def create_dog_kernels(ksize, sigma):
    """Return the derivative-of-Gaussian kernels (Dx, Dy).

    Differentiating a noisy image amplifies the noise, so the image is smoothed
    first. Because differentiation and convolution commute, the two steps
    collapse into one kernel (Lecture 2, slides 68, 73-75):

        Dx(s, t) = d/dt G(s, t) = -(t - c) / sigma^2 * G(s, t)

    with c = (ksize - 1) / 2, s the row offset and t the column offset. Sign
    convention: as in sobel_kernels(), Dx responds positively to intensity
    increasing to the right.

    Args:
        ksize (int): odd kernel size.
        sigma (float): standard deviation.
    Returns:
        tuple: two (ksize, ksize) float64 arrays.
    """
    # HINT: start from create_gaussian_kernel(ksize, sigma) and multiply by the
    #       coordinate grid. Do not renormalise: a derivative kernel sums to 0.
    # filtering() correlates, so the kernel is the flipped derivative:
    # +(t - c) instead of -(t - c), which makes Dx positive on a rising edge.
    gauss = create_gaussian_kernel(ksize, sigma)
    c = (ksize - 1) / 2.0
    axis = np.arange(ksize, dtype=np.float64) - c
    ss, tt = np.meshgrid(axis, axis, indexing='ij')
    dx = tt / sigma ** 2 * gauss
    dy = ss / sigma ** 2 * gauss
    return dx, dy


# -----------------------------------------------------------------------------
# 5-c. Non-maximum suppression
# -----------------------------------------------------------------------------
def non_maximum_suppression(magnitude, orientation):
    """Keep only the pixels that are a local maximum along the gradient.

    The gradient direction is rounded to one of four: 0, 45, 90 and 135
    degrees. A pixel survives if its magnitude is at least as large as the two
    neighbours on either side along that direction. Border pixels are set to 0.

    Args:
        magnitude (numpy.ndarray): (H, W), non-negative.
        orientation (numpy.ndarray): (H, W), radians.
    Returns:
        numpy.ndarray: (H, W) float64, thinned magnitude.
    """
    # HINT: quantise (degrees % 180) into four bins, then compare against
    #       shifted copies of the magnitude with numpy slicing.
    magnitude = np.asarray(magnitude, dtype=np.float64)
    angle = np.rad2deg(orientation) % 180.0
    m = magnitude
    centre = m[1:-1, 1:-1]
    a = angle[1:-1, 1:-1]
    # (neighbour on one side, neighbour on the other side) for each direction;
    # rows grow downwards, so 45 degrees points right-down.
    pairs = {
        0: (m[1:-1, :-2], m[1:-1, 2:]),      # left,      right
        45: (m[:-2, :-2], m[2:, 2:]),        # up-left,   down-right
        90: (m[:-2, 1:-1], m[2:, 1:-1]),     # up,        down
        135: (m[:-2, 2:], m[2:, :-2]),       # up-right,  down-left
    }
    bins = {
        0: (a < 22.5) | (a >= 157.5),
        45: (a >= 22.5) & (a < 67.5),
        90: (a >= 67.5) & (a < 112.5),
        135: (a >= 112.5) & (a < 157.5),
    }
    keep = np.zeros(centre.shape, dtype=bool)
    for direction, (before, after) in pairs.items():
        keep |= bins[direction] & (centre >= before) & (centre >= after)
    target = np.zeros_like(m)
    target[1:-1, 1:-1] = np.where(keep, centre, 0.0)
    return target


# -----------------------------------------------------------------------------
# 5-c. Hysteresis thresholding
# -----------------------------------------------------------------------------
def hysteresis_threshold(magnitude, low, high):
    """Two thresholds with connectivity.

    A pixel above `high` is an edge. A pixel between `low` and `high` is an
    edge only if it is connected, through other such pixels, to one above
    `high`. Everything else is discarded (Lecture 2, slide 71).

    Args:
        magnitude (numpy.ndarray): (H, W), the output of non_maximum_suppression.
        low (float): lower threshold.
        high (float): upper threshold.
    Returns:
        numpy.ndarray: (H, W) uint8, 255 on edges and 0 elsewhere.
    """
    if low > high:
        raise ValueError('low must not exceed high')
    strong = magnitude >= high
    weak = (magnitude >= low) & ~strong
    # HINT: start from the strong pixels and walk outwards over weak pixels.
    #       A stack (collections.deque) of coordinates is enough; each weak
    #       pixel is visited at most once.
    height, width = magnitude.shape
    edge = strong.copy()
    stack = deque(zip(*np.nonzero(strong)))
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width
                        and weak[nr, nc] and not edge[nr, nc]):
                    edge[nr, nc] = True
                    stack.append((nr, nc))
    return (edge.astype(np.uint8) * 255)


def canny(image, ksize=5, sigma=1.0, low=80.0, high=200.0):
    """Smooth, differentiate, thin, then threshold with hysteresis."""
    smoothed = filtering(image, create_gaussian_kernel(ksize, sigma), mode='replicate')
    _, _, magnitude, orientation = compute_gradient(smoothed)
    thinned = non_maximum_suppression(magnitude, orientation)
    return hysteresis_threshold(thinned, low, high)


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
def main(args):
    out = imutils.ensure_dir(args.out)
    gray = imutils.imread_gray(os.path.join(args.data, f'{args.image}.png'))
    print(f'input : {args.image}.png {gray.shape[1]}x{gray.shape[0]} (grayscale)')

    # 5-a. Sobel gradient
    gx, gy, magnitude, orientation = compute_gradient(gray)
    imutils.imsave(os.path.join(out, 'p5_sobel_gx.png'), np.abs(gx))
    imutils.imsave(os.path.join(out, 'p5_sobel_gy.png'), np.abs(gy))
    imutils.imsave(os.path.join(out, 'p5_sobel_magnitude.png'), magnitude)
    imutils.imsave(os.path.join(out, 'p5_sobel_orientation.png'),
                   imutils.orientation_to_bgr(magnitude, orientation))
    print(f'\n[5-a] Sobel  max |gx|={np.abs(gx).max():8.2f}  '
          f'max |gy|={np.abs(gy).max():8.2f}  max magnitude={magnitude.max():8.2f}')

    # 5-b. derivative of Gaussian on a clean and on a noisy image
    rng = np.random.default_rng(args.seed)
    noisy = gray + rng.normal(0.0, args.noise, gray.shape)
    imutils.imsave(os.path.join(out, 'p5_noisy_input.png'), noisy)
    print(f'\n[5-b] derivative of Gaussian (noise sigma = {args.noise:g})')
    print(f'  {"sigma":>5s}  {"ksize":>5s}  {"clean max":>10s}  {"noisy max":>10s}'
          f'  {"noisy/clean":>12s}')
    for sigma in args.sigmas:
        ksize = 2 * int(np.ceil(3.0 * sigma)) + 1
        kernels = create_dog_kernels(ksize, sigma)
        _, _, clean_mag, _ = compute_gradient(gray, kernels)
        _, _, noisy_mag, _ = compute_gradient(noisy, kernels)
        imutils.imsave(os.path.join(out, f'p5_dog_clean_s{sigma:g}.png'),
                       clean_mag / max(clean_mag.max(), 1e-12) * 255.0)
        imutils.imsave(os.path.join(out, f'p5_dog_noisy_s{sigma:g}.png'),
                       noisy_mag / max(noisy_mag.max(), 1e-12) * 255.0)
        ratio = float(noisy_mag.mean() / max(clean_mag.mean(), 1e-12))
        print(f'  {sigma:5.1f}  {ksize:5d}  {clean_mag.max():10.3f}'
              f'  {noisy_mag.max():10.3f}  {ratio:12.3f}')

    # 5-c, 5-d. Canny at two threshold settings
    print(f'\n[5-c, 5-d] Canny (ksize={args.kernel_size}, sigma={args.sigma})')
    print(f'  {"low":>6s}  {"high":>6s}  {"edge pixels":>12s}  {"ratio":>7s}')
    total = gray.size
    for low, high in args.thresholds:
        edges = canny(gray, args.kernel_size, args.sigma, low, high)
        imutils.imsave(os.path.join(out, f'p5_canny_{int(low)}_{int(high)}.png'), edges)
        count = int((edges > 0).sum())
        print(f'  {low:6.1f}  {high:6.1f}  {count:12d}  {count / total:7.4f}')

    print(f'\nwrote results to {out}/')


def parse_thresholds(values):
    if len(values) % 2:
        raise argparse.ArgumentTypeError('--thresholds needs pairs of numbers')
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def parse_args():
    parser = argparse.ArgumentParser(description='Problem 5. Edge detection')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--image', type=str, default='cat')
    parser.add_argument('--sigmas', type=float, nargs='+', default=(1.0, 2.0, 4.0))
    parser.add_argument('--noise', type=float, default=15.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[80.0, 200.0, 20.0, 50.0])
    args = parser.parse_args()
    args.thresholds = parse_thresholds(args.thresholds)
    return args


if __name__ == '__main__':
    main(parse_args())
