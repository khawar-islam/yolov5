import torch
import cv2
from PIL import Image

from utils.plots import Annotator, colors, save_one_box
import os
import glob
import numpy as np
# Load Ours Custom Model
model = torch.hub.load('.', 'custom', path='/media/coremax/CM_1/yolov5/runs/train/exp4/weights/last.pt', source='local')

# Files extension
img_Extension = ['jpg', 'jpeg', 'png']

# Load all testing images
my_path = "/home/coremax/Documents/hide_info_test_dataset/testing_images/"

# Save images into array
files = []
[files.extend(glob.glob(my_path + '*.' + e)) for e in img_Extension]

# Iteration on all images
images = [cv2.imread(file) for file in files]

total_images = 1

# Taking only image name to save with save name
image_file_name = ''

for img in glob.glob(my_path + '*.*'):
    img_bgr_rgb = cv2.imread(img)
    file_Name = os.path.basename(img)
    detections = model(img_bgr_rgb[:, :, ::-1])
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    if len(results) == 0:
        cv2.imwrite(os.path.join("/home/coremax/Documents/hide_info_test_dataset/detected/", file_Name), img_bgr_rgb)
    else:
        for result in results:
            print(result['class'])
            con = result['confidence']
            cs = result['class']
            x1 = int(result['xmin'])
            y1 = int(result['ymin'])
            x2 = int(result['xmax'])
            y2 = int(result['ymax'])
            imagee = cv2.rectangle(img_bgr_rgb, (x1, y1), (x2, y2), (255, 87, 51), -1)
            cv2.imwrite(os.path.join("/home/coremax/Documents/hide_info_test_dataset/detected/", file_Name), img_bgr_rgb)




