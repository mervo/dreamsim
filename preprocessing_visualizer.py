import os
import cv2
import numpy as np

ROOT_DIR = '/data/datasets/beex/2024-02-29--10-25-39_SiteA_revisit_with_rtk_0_fls/scanning_profile/'
# OUTPUT_DIR = f'{ROOT_DIR}pre_processed/'
OUTPUT_DIR = '/data/datasets/beex/2024-02-29--10-25-39_SiteA_revisit_with_rtk_0_fls/scanning_profile_pre_processed_bf_bf_bt/'
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
    return cv2.bilateralFilter(cv2.bilateralFilter(image, 21, 75, 75), 5, 150, 75)


def binary_threshold(image, min_threshold):
    image = np.array(image)
    ret, thresh = cv2.threshold(image, min_threshold, 255, cv2.THRESH_BINARY)
    return thresh


for filename in os.listdir(ROOT_DIR):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(ROOT_DIR, filename))
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename),
                    binary_threshold(bilateral_filter(img), 40))
                    # bilateral_filter(binary_threshold(img, 50)))
