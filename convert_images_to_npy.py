import os

import numpy as np
import tqdm

from preprocess import pre_process


def save_batch(images, masks, batch_index, output_dir):
    images = np.array(images)
    masks = np.array(masks)

    try:
        np.save(os.path.join(output_dir, f'images_batch_{batch_index}.npy'), images)
        np.save(os.path.join(output_dir, f'masks_batch_{batch_index}.npy'), masks)
        print(f'Batch {batch_index} saved successfully.')
    except IOError as e:
        print(f"Error saving data: {e}")


def png_npy(input_dir, image_size=224, batch_size=10000):
    images = []
    masks = []
    batch_index = 0

    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    image_dir = os.path.join(input_dir, 'image')
    mask_dir = os.path.join(input_dir, 'mask')

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Image or mask directory does not exist.")
        return

    # 获取所有图像文件名
    image_paths = sorted(os.listdir(image_dir))

    for i, path in enumerate(tqdm.tqdm(image_paths)):
        image_path = os.path.join(image_dir, path)
        mask_path = os.path.join(mask_dir, path.replace("IMAGE", "MASK"))

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping {path}, corresponding mask or image not found.")
            continue

        # 处理图像和掩码
        image, mask = pre_process(image_path, mask_path, image_size=image_size)
        images.append(image)
        masks.append(mask)

        # 每 batch_size 张图像保存为一个 .npy 文件
        if len(images) == batch_size:
            save_batch(images, masks, batch_index, input_dir)
            images = []
            masks = []
            batch_index += 1

    # 保存最后一批图像
    if images:
        save_batch(images, masks, batch_index, input_dir)


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()

    train_input_dir = os.path.join(args.data_path, 'augmentation_train')
    test_input_dir = os.path.join(args.data_path, 'augmentation_test')

    png_npy(train_input_dir, args.image_size)
    png_npy(test_input_dir, args.image_size)
