import cv2
import torch
import numpy as np
from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

import PIL.Image

torch.manual_seed(3407)

from preprocess import pre_process, pre_process2


def preprocessing(image, mask, image_size):
    image = image.resize((image_size, image_size),
                         PIL.Image.BILINEAR)
    mask = mask.resize((image_size, image_size),
                       PIL.Image.NEAREST)
    # 使用双边滤波处理图像
    # image = image.filter(ImageFilter.SMOOTH_MORE)
    # image.show()
    # image1.show()
    # exit(0)
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    mask = np.array(mask)
    # print(mask)
    # mask[mask == 38.0] = 1.0
    # mask[mask == 75.0] = 2.0
    mask[mask >= 2] = 2
    return image, mask


transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class MyDataset(Dataset):
    def __init__(self, path, image_size):
        # self.images = torch.tensor(np.load(image_path))  # 加载npy数据
        # self.masks = torch.tensor(np.load(mask_path))
        self.path = path
        self.image_size = image_size
        self.image_paths = os.listdir(path + '/image')
        self.mask_paths = os.listdir(path + '/mask')

        # self.transforms = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.path, 'image', self.image_paths[index])
        mask_path = os.path.join(self.path, 'mask', self.mask_paths[index])

        # image = PIL.Image.open(image_path).convert('RGB')
        # mask = PIL.Image.open(mask_path)
        # image, mask = preprocessing(image, mask, image_size=self.image_size)

        image, mask = pre_process(image_path, mask_path, self.image_size)

        # if self.transforms is not None:
        #     image = self.transforms(image)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        # mask = torch.Tensor(mask)
        # mask = self.transforms(mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)


def compute_weights(masks_path):
    # calculate the weights of different classes based on train samples
    masks = np.load(masks_path)
    sum_0 = (np.sum(masks == 0))
    sum_1 = (np.sum(masks == 1))
    sum_2 = (np.sum(masks == 2))
    print(sum_0, " ", sum_1, " ", sum_2)
    sum_total = sum_0 + sum_1 + sum_2
    weights_list = [(1 / sum_0) * sum_total / 2.0, (1 / sum_1) * sum_total / 2.0,
                    (1 / sum_2) * sum_total / 2.0]
    print('Weights for different classes are:', weights_list)


def main():
    dataset = MyDataset("data/train", 192)
    data = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, j in data:
        print(i.shape, j.shape)
        print(np.min(i.numpy()), np.max(i.numpy()))
        print(np.min(j.numpy()), np.max(j.numpy()))
    # compute_weights("data/train/masks.npy")


if __name__ == '__main__':
    main()
    # print(torch.cuda.is_bf16_supported())
