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


def pre_process(image_path, depth_path, mask_path, image_size):
    image = cv2.imread(image_path)
    if image is None:
        print(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    # image = image / 255.0

    if depth_path is None:
        depth = np.ones((image_size, image_size), np.float32)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        depth = np.array(depth, np.float32)
        depth[depth > 0] = (depth[depth > 0] - 80) / 40

        # depth = cv2.GaussianBlur(depth, (3, 3), 0)
        # depth = np.array(depth, np.float32)
        # depth[depth > 0] = (depth[depth > 0]) / np.max(depth)

        depth[depth > 1] = 1
        depth[depth < 0] = 0
        # print(np.min(depth), np.max(depth))

    if mask_path is None:
        mask = np.zeros((image_size, image_size), np.float32)
    else:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        # mask[mask == 64.0] = 1.0
        mask[mask == 129.0] = 1.0
        mask[mask == 192.0] = 2.0
        mask[mask == 255.0] = 3.0
    return image, depth, mask
