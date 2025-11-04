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
    img = (img_rgb01.clip(0,1) * 255.0).round().astype(np.uint8)

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
#> 10 → Complete different colors
# This method returns an array with the difference of color of each pixel, if images are passed as parameters
def deltaE_2000(color_1_lab: np.ndarray, color_2_lab: np.ndarray) -> np.ndarray:
    return color.deltaE_ciede2000(color_1_lab, color_2_lab)
