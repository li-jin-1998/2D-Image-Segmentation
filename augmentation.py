import argparse
import os
import random
import sys
import shutil
from PIL import ImageEnhance
import numpy as np
import PIL.Image
import tqdm

CLASS_NUMBER = 2
INPUT_SHAPE = [192, 192, 3]


def preprocessing(image, mask):
    mask = PIL.Image.fromarray(np.array(mask))
    image = image.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]),
                         PIL.Image.BILINEAR)
    mask = mask.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]),
                       PIL.Image.NEAREST)
    image = np.array(image, np.float32)
    image = image / 127.5 - 1
    mask = np.array(mask)
    mask[mask >= CLASS_NUMBER] = CLASS_NUMBER
    return image, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input directory', default="/home/lij/PycharmProjects/Seg/data/")
    args = parser.parse_args()

    # test_input_dir = os.path.join(args.input_dir, 'test')
    train_input_dir = os.path.join(args.input_dir, 'train')
    # train_input_dir = "/home/lij/PycharmProjects/Seg/Final_test"
    augmentation_output_dir = os.path.join(args.input_dir, 'augmentation3')

    if os.path.exists(augmentation_output_dir):
        shutil.rmtree(augmentation_output_dir)
    os.makedirs(augmentation_output_dir)
    os.makedirs(os.path.join(augmentation_output_dir, 'image'))
    os.makedirs(os.path.join(augmentation_output_dir, 'mask'))

    # if not (os.path.exists(os.path.join(test_input_dir, 'image_list.txt')) and
    #         os.path.exists(os.path.join(train_input_dir, 'image_list.txt'))):
    #     print('List files do not exist!')
    #     sys.exit(1)

    with open(os.path.join(train_input_dir, 'image_list.txt'), 'r',
              encoding='utf8') as f:
        lines = f.readlines()
        images = []
        masks = []
        for i in tqdm.tqdm(range(len(lines))):
            name = lines[i].split()[0]
            base = os.path.splitext(name)[0]
            image = PIL.Image.open(
                os.path.join(os.path.join(train_input_dir, 'image'), name))
            mask = PIL.Image.open(
                os.path.join(os.path.join(train_input_dir, 'mask'), name))
            image_flr = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            mask_flr = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            image_ftp = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            mask_ftp = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            # 随机旋转图像
            # seed = random.randint(0, 360)
            # image_rot = image.rotate(seed)
            # mask_rot = mask.rotate(seed)
            # image_rot.save(os.path.join(augmentation_output_dir, 'image',
            #                             base + '[ROTATE].png'))
            # mask_rot.save(os.path.join(augmentation_output_dir, 'mask',
            #                            base + '[ROTATE].png'))
            # 随机缩放图像
            # scale = random.uniform(0.8, 1.2)
            # image_scale = image.resize((int(image.width * scale), int(image.height * scale)))
            # mask_scale = mask.resize((int(image.width * scale), int(image.height * scale)))
            # image_scale.save(os.path.join(augmentation_output_dir, 'image',
            #                               base + '[SCALE].png'))
            # mask_scale.save(os.path.join(augmentation_output_dir, 'mask',
            #                              base + '[SCALE].png'))

            # 随机调整亮度
            brightness = ImageEnhance.Brightness(image)
            seed2 = random.uniform(0.8, 1.2)
            bright_image = brightness.enhance(seed2)
            bright_mask = mask
            bright_image.save(os.path.join(augmentation_output_dir, 'image',
                                           base + '[BRIGHT].png'))
            bright_mask.save(os.path.join(augmentation_output_dir, 'mask',
                                          base + '[BRIGHT].png'))
            # 随机调整对比度
            seed3 = random.uniform(0.8, 1.2)
            contrast = ImageEnhance.Contrast(image)
            contrast_image = contrast.enhance(seed3)
            contrast_mask = mask
            contrast_image.save(os.path.join(augmentation_output_dir, 'image',
                                             base + '[ENHANCE].png'))
            contrast_mask.save(os.path.join(augmentation_output_dir, 'mask',
                                            base + '[ENHANCE].png'))
            # image_90 = image.transpose(PIL.Image.Transpose.ROTATE_90)
            # mask_90 = mask.transpose(PIL.Image.Transpose.ROTATE_90)
            # image_180 = image.transpose(PIL.Image.Transpose.ROTATE_180)
            # mask_180 = mask.transpose(PIL.Image.Transpose.ROTATE_180)
            # image_270 = image.transpose(PIL.Image.Transpose.ROTATE_270)
            # mask_270 = mask.transpose(PIL.Image.Transpose.ROTATE_270)
            image.save(os.path.join(augmentation_output_dir, 'image',
                                    base + '[ORIGIN].png'))
            mask.save(os.path.join(augmentation_output_dir, 'mask',
                                   base + '[ORIGIN].png'))
            image_flr.save(os.path.join(augmentation_output_dir, 'image',
                                        base + '[FLIP_LEFT_RIGHT].png'))
            mask_flr.save(os.path.join(augmentation_output_dir, 'mask',
                                       base + '[FLIP_LEFT_RIGHT].png'))
            image_ftp.save(os.path.join(augmentation_output_dir, 'image',
                                        base + '[FLIP_TOP_BOTTOM].png'))
            mask_ftp.save(os.path.join(augmentation_output_dir, 'mask',
                                       base + '[FLIP_TOP_BOTTOM].png'))
            # image_90.save(os.path.join(augmentation_output_dir, 'image/data',
            #                            base + '[ROTATE_90].png'))
            # mask_90.save(os.path.join(augmentation_output_dir, 'mask/data',
            #                           base + '[ROTATE_90].png'))
            # image_180.save(os.path.join(augmentation_output_dir, 'image/data',
            #                             base + '[ROTATE_180].png'))
            # mask_180.save(os.path.join(augmentation_output_dir, 'mask/data',
            #                            base + '[ROTATE_180].png'))
            # image_270.save(os.path.join(augmentation_output_dir, 'image/data',
            #                             base + '[ROTATE_270].png'))
            # mask_270.save(os.path.join(augmentation_output_dir, 'mask/data',
            #                            base + '[ROTATE_270].png'))
            image, mask = preprocessing(image, mask)
            images.append(image)
            masks.append(mask)
            # print('Prepare augmented train sample from:', name)
        # np.save(os.path.join(train_input_dir, 'images.npy'), images)
        # np.save(os.path.join(train_input_dir, 'masks.npy'), masks)

    # with open(os.path.join(test_input_dir, 'image_list.txt'), 'r',
    #           encoding='utf8') as f:
    #     lines = f.readlines()
    #     images = []
    #     masks = []
    #     for i in tqdm.tqdm(range(len(lines))):
    #         name = lines[i].split()[0]
    #         base = os.path.splitext(name)[0]
    #         image = PIL.Image.open(
    #             os.path.join(os.path.join(test_input_dir, 'image'), name))
    #         mask = PIL.Image.open(
    #             os.path.join(os.path.join(test_input_dir, 'mask'), name))
    #         # image_flr = image.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
    #         # mask_flr = mask.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
    #         # image_ftp = image.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)
    #         # mask_ftp = mask.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)
    #         # image_90 = image.transpose(PIL.Image.Transpose.ROTATE_90)
    #         # mask_90 = mask.transpose(PIL.Image.Transpose.ROTATE_90)
    #         # image_180 = image.transpose(PIL.Image.Transpose.ROTATE_180)
    #         # mask_180 = mask.transpose(PIL.Image.Transpose.ROTATE_180)
    #         # image_270 = image.transpose(PIL.Image.Transpose.ROTATE_270)
    #         # mask_270 = mask.transpose(PIL.Image.Transpose.ROTATE_270)
    #         # image.save(os.path.join(augmentation_output_dir, 'image/data',
    #         #                         base + '[ORIGIN].png'))
    #         # mask.save(os.path.join(augmentation_output_dir, 'mask/data',
    #         #                        base + '[ORIGIN].png'))
    #         # image_flr.save(os.path.join(augmentation_output_dir, 'image/data',
    #         #                             base + '[FLIP_LEFT_RIGHT].png'))
    #         # mask_flr.save(os.path.join(augmentation_output_dir, 'mask/data',
    #         #                            base + '[FLIP_LEFT_RIGHT].png'))
    #         # image_ftp.save(os.path.join(augmentation_output_dir, 'image/data',
    #         #                             base + '[FLIP_TOP_BOTTOM].png'))
    #         # mask_ftp.save(os.path.join(augmentation_output_dir, 'mask/data',
    #         #                            base + '[FLIP_TOP_BOTTOM].png'))
    #         # image_90.save(os.path.join(augmentation_output_dir, 'image/data',
    #         #                            base + '[ROTATE_90].png'))
    #         # mask_90.save(os.path.join(augmentation_output_dir, 'mask/data',
    #         #                           base + '[ROTATE_90].png'))
    #         # image_180.save(os.path.join(augmentation_output_dir, 'image/data',
    #         #                             base + '[ROTATE_180].png'))
    #         # mask_180.save(os.path.join(augmentation_output_dir, 'mask/data',
    #         #                            base + '[ROTATE_180].png'))
    #         # image_270.save(os.path.join(augmentation_output_dir, 'image/data',
    #         #                             base + '[ROTATE_270].png'))
    #         # mask_270.save(os.path.join(augmentation_output_dir, 'mask/data',
    #         #                            base + '[ROTATE_270].png'))
    #         image, mask = preprocessing(image, mask)
    #         images.append(image)
    #         masks.append(mask)
    #         # print('Prepare test sample from:', name)
    # np.save(os.path.join(test_input_dir, 'images.npy'), images)
    # np.save(os.path.join(test_input_dir, 'masks.npy'), masks)


if __name__ == '__main__':
    main()
