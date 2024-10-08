import PIL.Image
import cv2
import numpy as np

COLORS = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
          (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
          (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
          (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
          (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
          (0, 64, 128), (128, 64, 12)]


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
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    # image = image / 255.0

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask[mask == 64.0] = 1.0
    mask[mask == 129.0] = 2.0
    mask[mask == 192.0] = 3.0
    mask[mask == 255.0] = 4.0
    mask[mask == 32.0] = 4.0

    return image, mask
