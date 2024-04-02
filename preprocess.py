import numpy as np
import PIL.Image
import cv2

COLORS = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
          (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
          (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
          (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
          (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
          (0, 64, 128), (128, 64, 12)]


def image_preprocessing(image):
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    return image


def mask_preprocessing(mask):
    mask = np.array(mask)
    mask[mask == 38.0] = 1.0
    mask[mask == 75.0] = 2.0
    return mask


def preprocessing(image, image_size):
    image = image.resize((image_size, image_size),
                         PIL.Image.BILINEAR)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    return image


def pre_process(image_path, mask_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    # image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
    # img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # produce a pseudocolored image. 伪彩色
    image = np.array(image, np.float32)
    image = image / 127.5 - 1

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask[mask == 129.0] = 1.0
    mask[mask == 192.0] = 2.0
    mask[mask == 255.0] = 3.0
    mask[mask == 64.0] = 0.0
    # print(mask)
    return image, mask

def pre_process2(image_path, mask_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    # image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
    # img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # produce a pseudocolored image. 伪彩色
    image = np.array(image, np.float32)
    image = image / 127.5 - 1

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask[mask == 64.0] = 1.0
    mask[mask == 129.0] = 2.0
    mask[mask == 192.0] = 3.0
    mask[mask == 255.0] = 4.0
    # print(mask)
    return image, mask