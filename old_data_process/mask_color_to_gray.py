import os
import shutil

import cv2
import numpy as np

src = "/mnt/algo_storage_server/UNet/Dataset/mask/"
dst = "/mnt/algo_storage_server/UNet/Dataset/gray_mask/"
if os.path.exists(dst):
    print('Output directory already exists: ', dst)
    shutil.rmtree(dst)
os.makedirs(dst)

for p in os.listdir(src):
    mask_path = os.path.join(src, p)
    print(mask_path)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.array(mask)
    mask[mask == 0] = 129
    mask[mask == 38.0] = 0
    mask[mask == 75.0] = 255
    cv2.imwrite(os.path.join(dst, p), mask)
