import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import os


torch.manual_seed(1)  # reproducible




class MyDataset(Dataset):
    def __init__(self, image_path, mask_path):
        self.images = torch.tensor(np.load(image_path))  # 加载npy数据
        self.masks = torch.tensor(np.load(mask_path))
        # self.transforms = transform  # 转为tensor形式

    def __getitem__(self, index):

        image = self.images[index, :, :, :]  # 读取每一个npy的数据
        image = image.permute(2, 0, 1)
        mask = self.masks[index, :, :]
        # mask=self.transforms(mask)
        return image, mask

    def __len__(self):
        return self.images.shape[0]


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
    data = np.load("data/train/masks.npy")
    print(data.shape)
    # dataset = MyDataset("data/train/images.npy", "data/train/masks.npy")
    # data = DataLoader(dataset, batch_size=1, shuffle=True)
    # for i,j in data:
    #     print(i.shape)
    # compute_weights("data/train/masks.npy")


if __name__ == '__main__':
    main()
