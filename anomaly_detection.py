import shutil
from PIL import Image
from dreamsim import dreamsim
from torchvision import transforms
import torch
import os
import cv2
import numpy as np

# A higher score means more different, lower means more similar.
DISTANCE_THRESHOLD = .16
BINARY_THRESHOLD = 40

INPUT_DIR = '/data/datasets/beex/2024-02-29--10-25-39_SiteA_revisit_with_rtk_0_fls/scanning_profile'
OUTPUT_DIR = f'/data/datasets/beex/2024-02-29--10-25-39_SiteA_revisit_with_rtk_0_fls/anomaly_output_{DISTANCE_THRESHOLD}/'
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)


img_size = 224


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


def pad_to_square(image, padding_value=0):
    width, height = image.size
    max_dim = max(width, height)
    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2
    padding = (pad_width, pad_height, max_dim - width -
               pad_width, max_dim - height - pad_height)
    return transforms.functional.pad(image, padding, fill=padding_value)


t = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((img_size, img_size),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    # helped improve separation from 0.11 to 0.23 for change in sensor settings, reducing false negatives
    transforms.Lambda(lambda img: Image.fromarray(bilateral_filter(img))),
    transforms.Lambda(lambda img: Image.fromarray(
        binary_threshold(img, BINARY_THRESHOLD))),
    transforms.ToTensor()
])


def preprocess(img):
    img = img.convert('RGB')
    return t(img).unsqueeze(0)


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = dreamsim(pretrained=True, device=device)

# Load images
cur_img_ref = ''
cur_img_ref_filename = ''
list_of_anomalies = []
for filename in sorted(os.listdir(INPUT_DIR)):
    file_path = os.path.join(INPUT_DIR, filename)
    if os.path.isfile(file_path):
        if len(cur_img_ref) == 0:  # First image
            cur_img_ref = preprocess(Image.open(file_path)).to(device)
            cur_img_ref_filename = filename
            list_of_anomalies.append(filename)
            print(f'cur_img_ref: {cur_img_ref_filename}')
            shutil.copyfile(file_path, os.path.join(OUTPUT_DIR, filename))
            continue  # no need to compare to itself

        img = preprocess(Image.open(file_path)).to(device)
        distance = model(cur_img_ref, img)
        if distance > DISTANCE_THRESHOLD:
            anomaly_log = f'Anomaly detected, updating new reference image: {filename} vs ref image {cur_img_ref_filename}, distance = {distance}'
            print(anomaly_log)
            with open(os.path.join(OUTPUT_DIR, 'anomaly_distance.log'), 'a') as f:
                f.write(anomaly_log + '\n')
            cur_img_ref = img
            cur_img_ref_filename = filename
            list_of_anomalies.append(filename)
            shutil.copyfile(file_path, os.path.join(OUTPUT_DIR, filename))

print(f'List of {len(list_of_anomalies)} anomalies: {list_of_anomalies}')
