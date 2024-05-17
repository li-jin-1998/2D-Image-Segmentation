import os

import numpy as np
import tqdm

from preprocess import pre_process


def png_npy(input_dir, image_size=224):
    images = []
    masks = []
    for path in tqdm.tqdm(os.listdir(os.path.join(input_dir, 'image'))):
        image_path = os.path.join(input_dir, 'image', path)
        mask_path = os.path.join(input_dir, 'mask', path)
        image, mask = pre_process(image_path, mask_path, image_size=image_size)
        images.append(image)
        masks.append(mask)
    print('Prepare augmented sample from:', input_dir)
    np.save(os.path.join(input_dir, 'images.npy'), images)
    np.save(os.path.join(input_dir, 'masks.npy'), masks)


if __name__ == '__main__':
    from parse_args import parse_args
    args = parse_args()

    train_input_dir = os.path.join(args.data_path, 'augmentation_train')
    test_input_dir = os.path.join(args.data_path, 'augmentation_test')

    png_npy(train_input_dir, args.image_size)
    png_npy(test_input_dir, args.image_size)
