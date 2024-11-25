import os
import cv2
import numpy as np

ROOT_DIR = '/data/datasets/beex/2024-02-29--10-25-39_SiteA_revisit_with_rtk_0_fls/'
OUTPUT_DIR = f'{ROOT_DIR}pre_processed/'
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)


def bilateral_filter(image):
    """
    Applies a bilateral filter to an image.

    # d: Diameter of each pixel neighborhood.
    # sigmaColor: Value of \sigma  in the color space. The greater the value, the colors farther to each other will start to get mixed.
    # sigmaSpace: Value of \sigma  in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.


    Args:
        image (PIL.Image or np.ndarray): The input image.

    Returns:
        np.ndarray: The filtered image.
    """
    image = np.array(image)
    return cv2.bilateralFilter(image, 9, 75, 75)


for filename in os.listdir(ROOT_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(ROOT_DIR, filename))
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), bilateral_filter(img))
