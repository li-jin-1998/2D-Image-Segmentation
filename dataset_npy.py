import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(3407)  # reproducible


class MyDataset(Dataset):
    def __init__(self, npy_dir):
        self.npy_dir = npy_dir
        self.image_files = sorted(
            [f for f in os.listdir(self.npy_dir) if f.startswith('images_batch_') and f.endswith('.npy')])
        self.mask_files = sorted(
            [f for f in os.listdir(self.npy_dir) if f.startswith('masks_batch_') and f.endswith('.npy')])
        self.images, self.masks = self.load_all_batches()
        # self.images = torch.tensor(np.load(image_path))  # 加载npy数据
        # self.masks = torch.tensor(np.load(mask_path))
        # self.transforms = transform

    def load_all_batches(self):
        all_images = []
        all_masks = []
        for i, (img_file, mask_file) in enumerate(zip(self.image_files, self.mask_files)):
            start_time = time.time()  # Start timing
            images = np.load(os.path.join(self.npy_dir, img_file))
            masks = np.load(os.path.join(self.npy_dir, mask_file))
            load_time = time.time() - start_time  # Calculate elapsed time
            print(f"Loaded batch {i} in {load_time:.4f} seconds")

            all_images.append(images)
            all_masks.append(masks)
        all_images = np.concatenate(all_images, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        return all_images, all_masks

    def __getitem__(self, index):
        image = self.images[index, :, :, :]  # 读取每一个npy的数据
        image = np.array(image, np.float32)
        image = image / 127.5 - 1
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)

        mask = self.masks[index, :, :]
        # mask=self.transforms(mask)
        return image, mask

    def __len__(self):
        return len(self.images)


def compute_weights(masks_path):
    # calculate the weights of different classes based on train samples
    masks = np.load(masks_path)
    max_value = np.max(masks)
    sum_total = 0
    sum = np.zeros(max_value)
    for i in range(max_value):
        sum[i] = np.sum(masks == i)
        sum_total += sum[i]
    print(sum)
    weights_list = []
    for s in sum:
        weights_list.append((1 / s) * sum_total)
    print('Weights for different classes are:', weights_list)


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 8)

    for image, target in data_loader:
        plot_images = []
        plot_masks = []
        for i in range(plot_num):
            img = image[i].numpy().transpose(1, 2, 0)
            img = (img + 1) * 127.5
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            plot_masks.append(target[i].item())
            plot_images.append(img)

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.flatten()
        for img, lab, ax in zip(plot_images, plot_masks, axes):
            ax.imshow(img)
            ax.axis("off")
            ax.imshow(lab)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()

    # images_npy_path = os.path.join(args.data_path, 'augmentation_test', 'images.npy')
    # masks_npy_path = os.path.join(args.data_path, 'augmentation_test', 'masks.npy')
    # data = np.load(masks_npy_path)
    # print(data.shape)
    # compute_weights(masks_npy_path)


    dataset = MyDataset(os.path.join(args.data_path, 'augmentation_test'))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    plot_data_loader_image(data_loader)
    # for image, target in data_loader:
    #     print(image.shape, target.shape)

