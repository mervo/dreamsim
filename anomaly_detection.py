import shutil
from PIL import Image
from dreamsim import dreamsim
from torchvision import transforms
import torch
import os
import cv2
import numpy as np

# A higher score means more different, lower means more similar.
DISTANCE_THRESHOLD = .12
BINARY_THRESHOLD = 50
FRAMES_SINCE_ANOMALY_TO_REFRESH = 30
PROPORTION_OF_IMAGE_TO_KEEP_FROM_CENTER = 1/2
VISUALIZE_TRANSFORMATIONS = False

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


def crop_center_of_image(image, proportion_of_image):
    width, height = image.size
    crop_width = width * proportion_of_image
    left = (width - crop_width) // 2
    right = left + crop_width
    return image.crop((left, 0, right, height))


t = transforms.Compose([
    transforms.Lambda(lambda img: crop_center_of_image(img,
                                                       PROPORTION_OF_IMAGE_TO_KEEP_FROM_CENTER)),
    transforms.Lambda(lambda img: pad_to_square(img)),
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
    img = t(img)

    if VISUALIZE_TRANSFORMATIONS:
        # Convert PyTorch tensor back to NumPy array
        # Permute C x H x W to H x W x C
        numpy_img = img.permute(1, 2, 0).numpy()
        # Scale values from [0, 1] (default in ToTensor) to [0, 255] for OpenCV
        numpy_img = (numpy_img * 255).astype(np.uint8)
        # Convert RGB (Pillow/torch) to BGR (OpenCV) for display
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow("Transformed Image", numpy_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img.unsqueeze(0)


# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = dreamsim(pretrained=True, device=device)

# Load images
cur_img_ref = ''
cur_img_ref_filename = ''
frames_since_anomaly = 0
list_of_anomalies = []
for filename in sorted(os.listdir(INPUT_DIR)):
    frames_since_anomaly += 1
    file_path = os.path.join(INPUT_DIR, filename)
    if os.path.isfile(file_path):
        if len(cur_img_ref) == 0:  # First image
            cur_img_ref = preprocess(Image.open(file_path)).to(device)
            cur_img_ref_filename = filename
            frames_since_anomaly = 0  # reset counter
            list_of_anomalies.append(filename)
            print(f'cur_img_ref: {cur_img_ref_filename}')
            shutil.copyfile(file_path, os.path.join(OUTPUT_DIR, filename))
            continue  # no need to compare to itself

        img = preprocess(Image.open(file_path)).to(device)

        if frames_since_anomaly >= FRAMES_SINCE_ANOMALY_TO_REFRESH:
            frames_since_anomaly = 0
            cur_img_ref = img
            cur_img_ref_filename = filename
            print(f'Updating cur_img_ref to: {filename}')

        distance = model(cur_img_ref, img)
        if distance > DISTANCE_THRESHOLD:
            anomaly_log = f'Anomaly detected, updating new reference image: {filename} vs ref image {cur_img_ref_filename}, distance = {distance}'
            print(anomaly_log)
            with open(os.path.join(OUTPUT_DIR, 'anomaly_distance.log'), 'a') as f:
                f.write(anomaly_log + '\n')
            cur_img_ref = img
            frames_since_anomaly = 0  # reset counter
            cur_img_ref_filename = filename
            list_of_anomalies.append(filename)
            shutil.copyfile(file_path, os.path.join(OUTPUT_DIR, filename))

print(f'List of {len(list_of_anomalies)} anomalies: {list_of_anomalies}')
