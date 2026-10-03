"""Self-check for Homework 1.

This script compares your implementation with the OpenCV built-ins on a small
image. It is a debugging aid, not the grader.

    Passing every check does not mean full marks: the written answers and the
    figures carry as many points as the code.
    Failing a check does not mean zero: if your choice is defensible, write the
    reason in the report and you can still receive most of the credit.

The OpenCV functions used here are exactly the ones you may not call in your
own code. They appear in this file only.

Run:
    python check.py --data data
"""
import argparse
import os
import traceback

import cv2
import numpy as np

RESULTS = []


def record(name, detail, passed):
    passed = bool(passed)
    RESULTS.append((name, detail, passed))
    mark = 'PASS' if passed else 'FAIL'
    print(f'  [{mark}] {name:<34s} {detail}')


def skip(name, reason):
    RESULTS.append((name, reason, None))
    print(f'  [SKIP] {name:<34s} {reason}')


def guarded(name):
    """Decorator that turns an unfinished or broken part into a SKIP."""
    def wrapper(function):
        def inner(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except NotImplementedError:
                skip(name, 'not implemented yet')
            except Exception as error:  # noqa: BLE001 - a student aid, not a library
                skip(name, f'raised {type(error).__name__}: {error}')
                if os.environ.get('HW1_TRACEBACK'):
                    traceback.print_exc()
        return inner
    return wrapper


# -----------------------------------------------------------------------------
# Problem 1
# -----------------------------------------------------------------------------
def check_problem1(image):
    import p1_smoothing as p1
    print('\nProblem 1. Linear filtering and smoothing')

    @guarded('1-a kernel matches OpenCV')
    def kernel_check():
        for ksize, sigma in ((3, 1.0), (7, 4.0), (11, 9.0), (25, 9.0)):
            mine = p1.create_gaussian_kernel(ksize, sigma)
            column = cv2.getGaussianKernel(ksize, sigma)
            reference = column @ column.T
            error = float(np.abs(mine - reference).max())
            total = float(mine.sum())
            if error > 1e-9 or abs(total - 1.0) > 1e-9:
                record('1-a kernel matches OpenCV',
                       f'ksize={ksize} sigma={sigma}: max err={error:.2e}, sum={total:.6f}',
                       False)
                return
        record('1-a kernel matches OpenCV', 'max err < 1e-9, sum = 1', True)

    @guarded('1-b padding matches numpy')
    def pad_check():
        rng = np.random.default_rng(0)
        sample = rng.integers(0, 255, (9, 11, 3)).astype(np.float64)
        modes = (('zero', 'constant'), ('replicate', 'edge'))
        worst = 0.0
        for pad_size in (1, 2, 4):
            for mine_mode, numpy_mode in modes:
                width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
                reference = np.pad(sample, width, mode=numpy_mode)
                mine = p1.pad(sample, pad_size, mine_mode)
                if mine.shape != reference.shape:
                    record('1-b padding matches numpy',
                           f'{mine_mode}: shape {mine.shape} != {reference.shape}', False)
                    return
                worst = max(worst, float(np.abs(mine - reference).max()))
        record('1-b padding matches numpy', f'max err={worst:.2e}', worst < 1e-9)

    @guarded('1-c filtering matches OpenCV')
    def filtering_check():
        borders = {'zero': cv2.BORDER_CONSTANT,
                   'replicate': cv2.BORDER_REPLICATE}
        rng = np.random.default_rng(1)
        kernel = rng.random((5, 5))
        kernel /= kernel.sum()
        worst = 0.0
        for mode, border in borders.items():
            mine = p1.filtering(image, kernel, mode)
            reference = cv2.filter2D(image.astype(np.float64), -1, kernel,
                                     borderType=border)
            worst = max(worst, float(np.abs(mine - reference).max()))
        record('1-c filtering matches OpenCV', f'max err={worst:.2e} (2 padding modes)',
               worst < 1e-6)

    @guarded('1-c smoothing matches GaussianBlur')
    def blur_check():
        kernel = p1.create_gaussian_kernel(11, 4.0)
        mine = p1.filtering(image, kernel, 'replicate')
        reference = cv2.GaussianBlur(image.astype(np.float64), (11, 11), 4.0,
                                     borderType=cv2.BORDER_REPLICATE)
        error = float(np.abs(mine - reference).max())
        record('1-c smoothing matches GaussianBlur', f'max err={error:.2e}', error < 1e-6)

    kernel_check()
    pad_check()
    filtering_check()
    blur_check()


# -----------------------------------------------------------------------------
# Problem 2
# -----------------------------------------------------------------------------
def check_problem2(image):
    import p2_upsampling as p2
    print('\nProblem 2. Upsampling')
    target = (image.shape[0] * 8, image.shape[1] * 8)

    @guarded('2-a nearest matches OpenCV')
    def nearest_check():
        mine = p2.upsample_nearest(image, target)
        reference = cv2.resize(image, (target[1], target[0]),
                               interpolation=cv2.INTER_NEAREST_EXACT)
        if mine.shape != reference.shape:
            record('2-a nearest matches OpenCV',
                   f'shape {mine.shape} != {reference.shape}', False)
            return
        ratio = float(np.mean(np.asarray(mine) != reference))
        record('2-a nearest matches OpenCV', f'mismatched pixels={ratio * 100:.4f}%',
               ratio <= 1e-3)

    @guarded('2-b bilinear matches OpenCV')
    def bilinear_check():
        mine = np.asarray(p2.upsample_bilinear(image, target), dtype=np.float64)
        reference = cv2.resize(image, (target[1], target[0]),
                               interpolation=cv2.INTER_LINEAR).astype(np.float64)
        if mine.shape != reference.shape:
            record('2-b bilinear matches OpenCV',
                   f'shape {mine.shape} != {reference.shape}', False)
            return
        error = float(np.abs(mine - reference).max())
        record('2-b bilinear matches OpenCV', f'max err={error:.3f} (uint8 rounding)',
               error <= 1.0)

    nearest_check()
    bilinear_check()


# -----------------------------------------------------------------------------
# Problem 3
# -----------------------------------------------------------------------------
def check_problem3(image):
    import p3_downsampling as p3
    print('\nProblem 3. Downsampling and aliasing')
    target = (image.shape[0] // 8, image.shape[1] // 8)

    def energy(array):
        array = np.asarray(array, dtype=np.float64)
        return float((np.diff(array, axis=1) ** 2).mean()
                     + (np.diff(array, axis=0) ** 2).mean())

    @guarded('3-a naive output shape')
    def naive_check():
        mine = np.asarray(p3.downsample_naive(image, target))
        ok = mine.shape[:2] == target and mine.shape[2] == 3
        record('3-a naive output shape', f'{mine.shape}', ok)

    @guarded('3-b prefilter removes high freq')
    def prefilter_check():
        naive = p3.downsample_naive(image, target)
        smooth = p3.downsample_prefiltered(image, target, 2.0)
        before, after = energy(naive), energy(smooth)
        record('3-b prefilter removes high freq',
               f'edge energy {before:.1f} -> {after:.1f}', after < before)

    naive_check()
    prefilter_check()


# -----------------------------------------------------------------------------
# Problem 4
# -----------------------------------------------------------------------------
def check_problem4(image):
    import p4_pyramid as p4
    print('\nProblem 4. Gaussian pyramid')

    @guarded('4-a level count and sizes')
    def pyramid_check():
        levels = p4.build_gaussian_pyramid(image, 4, 5, 1.0)
        height, width = image.shape[:2]
        expected = [(height >> i, width >> i) for i in range(4)]
        shapes = [tuple(level.shape[:2]) for level in levels]
        ok = len(levels) == 4 and shapes == expected
        record('4-a level count and sizes', f'{shapes}', ok)

    @guarded('4-a level 0 is the input')
    def level0_check():
        levels = p4.build_gaussian_pyramid(image, 3, 5, 1.0)
        error = float(np.abs(np.asarray(levels[0], dtype=np.float64)
                             - image.astype(np.float64)).max())
        record('4-a level 0 is the input', f'max err={error:.2e}', error < 1e-9)

    pyramid_check()
    level0_check()


# -----------------------------------------------------------------------------
# Problem 5
# -----------------------------------------------------------------------------
def check_problem5(gray):
    import p5_edges as p5
    print('\nProblem 5. Edge detection')

    @guarded('5-a Sobel matches OpenCV')
    def sobel_check():
        gx, gy, magnitude, orientation = p5.compute_gradient(gray)
        rx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3,
                       borderType=cv2.BORDER_REPLICATE)
        ry = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3,
                       borderType=cv2.BORDER_REPLICATE)
        error = max(float(np.abs(gx - rx).max()), float(np.abs(gy - ry).max()))
        magnitude_error = float(np.abs(magnitude - np.sqrt(rx ** 2 + ry ** 2)).max())
        record('5-a Sobel matches OpenCV',
               f'max err={error:.2e}, magnitude err={magnitude_error:.2e}',
               error < 1e-6 and magnitude_error < 1e-6)

    @guarded('5-b DoG kernels are derivatives')
    def dog_check():
        dx, dy = p5.create_dog_kernels(9, 1.5)
        sums = abs(float(dx.sum())) + abs(float(dy.sum()))
        centre = 9 // 2
        rising = dx[centre, centre + 1] > 0 and dy[centre + 1, centre] > 0
        antisymmetric = float(np.abs(dx + dx[:, ::-1]).max()) < 1e-12
        record('5-b DoG kernels are derivatives',
               f'|sum|={sums:.2e}, sign ok={rising}, antisymmetric={antisymmetric}',
               sums < 1e-9 and rising and antisymmetric)

    @guarded('5-c NMS thins without adding')
    def nms_check():
        _, _, magnitude, orientation = p5.compute_gradient(gray)
        thinned = p5.non_maximum_suppression(magnitude, orientation)
        subset = bool(np.all((thinned == 0) | (np.abs(thinned - magnitude) < 1e-9)))
        kept = float((thinned > 0).mean())
        record('5-c NMS thins without adding',
               f'kept {kept * 100:.1f}% of pixels, values unchanged={subset}',
               subset and 0.0 < kept < 0.5)

    @guarded('5-c Canny matches OpenCV')
    def canny_check():
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0,
                                   borderType=cv2.BORDER_REPLICATE)
        worst = 1.0
        for low, high in ((80.0, 200.0), (20.0, 50.0)):
            mine = np.asarray(p5.canny(gray, 5, 1.0, low, high)) > 0
            reference = cv2.Canny(blurred.astype(np.uint8), low, high,
                                  apertureSize=3, L2gradient=True) > 0
            hit = float((mine & reference).sum())
            precision = hit / max(mine.sum(), 1)
            recall = hit / max(reference.sum(), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            worst = min(worst, f1)
        record('5-c Canny matches OpenCV', f'worst F1={worst:.3f} over 2 settings',
               worst >= 0.90)

    sobel_check()
    dog_check()
    nms_check()
    canny_check()


# -----------------------------------------------------------------------------
def main(args):
    path = os.path.join(args.data, f'{args.image}.png')
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'cannot read image: {path}')
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    print(f'self-check on {path} ({image.shape[1]}x{image.shape[0]})')

    check_problem1(image)
    check_problem2(image)
    check_problem3(image)
    check_problem4(image)
    check_problem5(gray)

    passed = sum(1 for _, _, state in RESULTS if state is True)
    failed = sum(1 for _, _, state in RESULTS if state is False)
    skipped = sum(1 for _, _, state in RESULTS if state is None)
    print(f'\n{passed} passed, {failed} failed, {skipped} skipped '
          f'out of {len(RESULTS)} checks')
    print('Paste this output into section 2 of your report.')


def parse_args():
    parser = argparse.ArgumentParser(description='Homework 1 self-check')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--image', type=str, default='cat',
                        help='a small image keeps the check fast')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
