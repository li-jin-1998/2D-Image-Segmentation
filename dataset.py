import cv2
import torch
import numpy as np
from PIL import ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os

torch.manual_seed(3407)

from preprocess import pre_process


class MyDataset(Dataset):
    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size
        self.image_paths = sorted([os.path.join(self.path, p) for p in os.listdir(path) if 'image' in p])
        self.mask_paths = sorted([os.path.join(self.path, p) for p in os.listdir(path) if 'mask' in p])
        self.depth_paths = sorted([os.path.join(self.path, p) for p in os.listdir(path) if 'depth' in p])
        assert len(self.image_paths) == len(self.mask_paths) == len(self.depth_paths)

    def __getitem__(self, index):
        image, depth, mask = pre_process(self.image_paths[index], self.depth_paths[index],
                                         self.mask_paths[index], self.image_size)
        # image = np.concatenate((image, depth[..., np.newaxis]), axis=2)

        depth = torch.Tensor(depth[..., np.newaxis])
        depth = depth.permute(2, 0, 1)

        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        return image, depth, mask

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    dataset = MyDataset(args.data_path + "/test", args.image_size)
    data = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, j, k in data:
        # print(i.shape, j.shape)
        # print(np.min(i.numpy()), np.max(i.numpy()))
        print(np.min(j.numpy()), np.max(j.numpy()))
    # print(torch.cuda.is_bf16_supported())
