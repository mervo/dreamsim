# ffmpeg -framerate 25 -start_number 444 -i frame_%06d.png -frames:v 759 -c:v libx264 -crf 25 -movflags +faststart output.mp4

import os
import cv2
import shutil

from PIL import Image, ImageDraw

ROOT_DIR = '/data/datasets/beex/2024-02-29--10-25-39_SiteA_revisit_with_rtk_0_fls/anomaly_output_0.14/'
BACKUP_DIR = os.path.join(ROOT_DIR, 'backup')
os.makedirs(BACKUP_DIR, exist_ok=True)
for filename in os.listdir(ROOT_DIR):
    if not os.path.isfile(os.path.join(ROOT_DIR, filename)):
        continue
    shutil.copy2(os.path.join(ROOT_DIR, filename), os.path.join(BACKUP_DIR, filename))


def add_red_border(image_path):
    img = cv2.imread(image_path)
    cv2.rectangle(
        img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 0, 255), 15)
    cv2.imwrite(image_path, img)


for filename in os.listdir(ROOT_DIR):
    print(filename)
    if filename.endswith(".png") or filename.endswith(".jpg"):
        add_red_border(os.path.join(ROOT_DIR, filename))