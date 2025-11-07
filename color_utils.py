import numpy as np
import cv2
from skimage import color


def read_rgb(image_path: str) -> np.ndarray:
    # Read image as BGR
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise FileNotFoundError(image_path)

    # Convert the BGR colors to the RGB pattern
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Normalize the color data to be between 0 and 1
    # 255 is because we colors can have a value between 0 and 255
    return (img_rgb.astype(np.float32) / 255.0).clip(0, 1)


def write_rgb(path: str, img_rgb01: np.ndarray) -> None:
    # cv2 works with images with values between 0 and 255
    img = (img_rgb01.clip(0, 1) * 255.0).round().astype(np.uint8)

    # cv2 works with BGR as default
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# LAB
# L - Lightness
# A - Red/Green Value
# B - Blue/Yellow Value
def rgb_to_lab(img_rgb01: np.ndarray) -> np.ndarray:
    return color.rgb2lab(img_rgb01)


def lab_to_rgb(img_lab: np.ndarray) -> np.ndarray:
    return np.clip(color.lab2rgb(img_lab), 0, 1)


# deltaE_2000 returns how much a color is different from another one
# ~1 → Almost the same color
# ~2–5 → Visible difference
# > 10 → Complete different colors
# This method returns an array with the difference of color of each pixel, if images are passed as parameters
def deltaE_2000(color_1_lab: np.ndarray, color_2_lab: np.ndarray) -> np.ndarray:
    return color.deltaE_ciede2000(color_1_lab, color_2_lab)


# Convert sRGB to linear RGB
def srgb_to_linear(rgb_01: np.ndarray) -> np.ndarray:
    # a: sRGB standard constant used in the exponential segment
    # thresh: limits separating the linear segment from exponential segment in the sRGB curve.
    a = 0.055
    thresh = 0.04045

    # Array with True where rgb_01 <= thresh
    low_section = rgb_01 <= thresh

    # Invert boolean array
    high_section = ~low_section

    # Create array with same structure as rgb01
    out = np.empty_like(rgb_01, dtype=np.float32)

    # Linear section of sRGB
    out[low_section] = rgb_01[low_section] / 12.92

    # Remove gama
    out[high_section] = ((rgb_01[high_section] + a) / (1 + a)) ** 2.4
    return out


# Convert linear RGB to sRGB
def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    # a: sRGB standard constant used in the exponential segment
    # thresh: threshold of the sRGB standard for the return path (linear→sRGB).
    a = 0.055
    thresh = 0.0031308

    low_section = linear <= thresh
    high_section = ~low_section

    # Create array with same structure as rgb01
    out = np.empty_like(linear, dtype=np.float32)

    # Low linear section
    out[low_section] = linear[low_section] * 12.92

    # Apply gama
    out[high_section] = (1 + a) * np.power(linear[high_section], 1 / 2.4) - a
    return out


# Machado et al. matrix, severity 1.0, linear RGB
MATRIX_PROTAN = np.array([
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
], dtype=np.float32)

MATRIX_DEUTAN = np.array([
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
], dtype=np.float32)

MATRIX_TRITAN = np.array([
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
], dtype=np.float32)

IDENTITY_MATRIX = np.eye(3, dtype=np.float32)


def severity_blend(matrix_def: np.ndarray, severity: float) -> np.ndarray:
    severity_01 = float(np.clip(severity, 0.0, 1.0))

    # Matrix blend
    return (1.0 - severity_01) * IDENTITY_MATRIX + severity_01 * matrix_def


# Apply matrix in linear RGB
def apply_matrix(img_linear: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # img_linear: HxWx3, float linear [0,1], no gama
    height, width, _ = img_linear.shape
    rgb_pixels = img_linear.reshape(-1, 3)

    # rgb_pixels is (N×3), each row is one pixel [R, G, B].
    # The color transform matrix is defined for column vectors, so we use matrix.T
    # to match the row-vector pixel layout. This applies the 3×3 transform to all pixels.
    out = rgb_pixels @ matrix.T
    return out.reshape(height, width, 3)


def simulate_cvd(img_rgb01: np.ndarray, cvd_type: str = "deutan", severity: float = 1.0) -> np.ndarray:
    """
    Simulates colorblind vision.
    cvd_type: "protanus" | "deuteranus" | "tritanus"
    severity: 0.0 (no effect) to 1.0 (severe)
    """

    # sRGB -> linear
    lin = srgb_to_linear(img_rgb01.clip(0, 1).astype(np.float32))

    # Choose matrix based on CVD
    t = cvd_type.lower()
    if t == "protan":
        blend_matrix = severity_blend(MATRIX_PROTAN, severity)
    elif t == "deutan":
        blend_matrix = severity_blend(MATRIX_DEUTAN, severity)
    elif t == "tritan":
        blend_matrix = severity_blend(MATRIX_TRITAN, severity)
    else:
        raise ValueError("cvd_type should be 'protan', 'deutan' or 'tritan'.")

    linear_simulation = apply_matrix(lin, blend_matrix)

    # Clamp + linear -> sRGB
    linear_simulation = np.clip(linear_simulation, 0.0, 1.0)
    srgb_simulation = linear_to_srgb(linear_simulation)
    return np.clip(srgb_simulation, 0.0, 1.0)
