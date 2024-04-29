import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(3407)

from preprocess import pre_process

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


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    dataset = MyDataset(args.data_path + "/augmentation_test", args.image_size)
    data = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, j in data:
        print(i.shape, j.shape)
        print(np.min(i.numpy()), np.max(i.numpy()))
        print(np.min(j.numpy()), np.max(j.numpy()))
    # print(torch.cuda.is_bf16_supported())
