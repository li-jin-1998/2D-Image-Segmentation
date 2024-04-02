import argparse
import os

import numpy as np
import PIL.Image
import tqdm


def preprocessing(image, mask, image_size):
    image = image.resize((image_size, image_size),
                         PIL.Image.BILINEAR)
    mask = mask.resize((image_size, image_size),
                       PIL.Image.NEAREST)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    mask = np.array(mask)
    mask[mask >= 2] = 2
    return image, mask


def png_npy(input_dir):
    images = []
    masks = []
    for path in tqdm.tqdm(os.listdir(os.path.join(input_dir, 'image'))):
        image = PIL.Image.open(
            os.path.join(os.path.join(input_dir, 'image'), path))
        mask = PIL.Image.open(
            os.path.join(os.path.join(input_dir, 'mask'), path))
        image, mask = preprocessing(image, mask, image_size=192)
        images.append(image)
        masks.append(mask)
    print('Prepare augmented sample from:', input_dir)
    np.save(os.path.join(input_dir, 'images.npy'), images)
    np.save(os.path.join(input_dir, 'masks.npy'), masks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input directory', default="/home/lij/PycharmProjects/Seg/data/")
    args = parser.parse_args()

    test_input_dir = os.path.join(args.input_dir, 'test')
    train_input_dir = os.path.join(args.input_dir, 'train')

    png_npy(test_input_dir)
    png_npy(train_input_dir)
