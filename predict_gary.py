import glob
import os
import shutil
import time

import cv2
import numpy as np
import torch
import tqdm

from others.open_dir import open_directory
from parse_args import parse_args, get_model, get_best_weight_path, get_device

CONVERT_COLOR = True

args = parse_args()
device = get_device()

# create model
model = get_model(args)
# load weights
weights_path = get_best_weight_path(args)
model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
model.to(device)
model.eval()  # 进入验证模式


def gray_to_color_mask(gray_image):
    custom_colormap = np.zeros((6, 3), dtype=np.uint8)

    custom_colormap[0] = [0, 0, 0]
    custom_colormap[1] = [64, 64, 64]
    custom_colormap[2] = [129, 129, 129]
    custom_colormap[3] = [64, 255, 64]
    custom_colormap[4] = [255, 255, 255]
    custom_colormap[5] = [255, 129, 64]

    color_image = custom_colormap[gray_image]
    # color_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    # color_image = cv2.addWeighted(origin_image, 1, color_image, 1-alpha, 0)
    return color_image


def mask_postprocessing(mask, w, h, convert_color=CONVERT_COLOR):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if convert_color:
        mask = gray_to_color_mask(mask)
    else:
        mask[mask == 1] = 64
        mask[mask == 2] = 129
        mask[mask == 3] = 192
        mask[mask == 4] = 255
        mask[mask == 5] = 32
    return mask


def predict_gray(predict_image_names, result_path, test_id=None):
    start_time = time.time()
    with torch.no_grad():
        for img_path in tqdm.tqdm(predict_image_names):
            original_img = cv2.imread(img_path)
            if original_img is None:
                print("load image error:", img_path)
                continue
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            image = cv2.resize(original_img, (args.image_size, args.image_size),
                               interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1
            image = torch.Tensor(image)
            image = image.permute(2, 0, 1)
            image = torch.unsqueeze(image, dim=0)

            output = model(image.to(device))
            if isinstance(output, dict):
                output = output['out']
                # output = output['aux_output0']
            prediction = output.argmax(1).squeeze(0)

            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            predict_result = mask_postprocessing(prediction, original_img.shape[1], original_img.shape[0])

            dst = os.path.join(result_path,
                               str(test_id) + "_" + os.path.splitext(os.path.basename(img_path))[0] + "_predict.png")

            cv2.imwrite(dst, predict_result)
            shutil.copy(str(img_path), dst.replace('predict', 'image'))

            if 'test' in img_path:
                origin_mask = str(img_path).replace('image/', 'mask/').replace("IMAGE", "MASK")
                if os.path.exists(origin_mask):
                    shutil.copy(origin_mask, dst.replace('predict', 'mask'))
    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(predict_image_names) / total_time))


def test_one(file_name, test_id):
    src = r"/mnt/algo-storage-server/Dataset/2D/外部收集/01_修复/{}/{}/*_7_*".format(file_name, test_id)
    print(">" * 10, src)
    predict_image_names = glob.glob(src)[::10]
    predict_image_names.sort()

    result_path = './visualization/{}'.format(test_id)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    predict_gray(predict_image_names, result_path, test_id)

def test_one_file():
    # test_id = 20240422114631
    # src = r"/mnt/algo-storage-server/Dataset/2D/外部收集/01_修复/2024Q1第二批/{}/*_7_*".format(test_id)
    # print(">" * 10, src)
    #
    # predict_image_names = glob.glob("/mnt/local_storage/lijin/Segmentation/dataset/data/augmentation_test/image/*.png")[::10]
    # predict_image_names = glob.glob(src)[::10]
    # predict_image_names = glob.glob("/mnt/algo-storage-server/Workspaces/lijin/implant_test/*")
    # predict_image_names = glob.glob("/mnt/algo-storage-server/Workspaces/lijin/images/*")[::5]
    predict_image_names = glob.glob(r"/home/lj/PycharmProjects/Data/metal_add/*image.png")
    # predict_image_names = glob.glob(args.data_path + "/test/image/*")[::4]
    predict_image_names.sort()

    result_path = './visualization'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    predict_gray(predict_image_names, result_path)
    open_directory(result_path)

if __name__ == '__main__':

    ids = ['20240129153006', '20240422092025', '20240422092449', '20240422092939', '20240422093459',
           '20240422094617', '20240422095714', '20240422100226', '20240422101247', '20240422104015',
           '20240422104531', '20240422110605', '20240422112902', '20240422113710', '20240422114631',
           '20240422115942', '20240422120640', '20240422121306', '20240422123642']
    ids2 = ['20231105142801', '20231204111852', '20240116133815', '20240329092513',
            '20240329100520', '20240329103029', '20240329103649', '20240329110428']



    file_name = '2024Q1第二批'
    file_name2 = '2024Q1'

    result_path = "./visualization"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)

    # for test_id in ids:
    #     test_one(file_name, test_id)

    # for test_id in ids2:
    #     test_one(file_name2, test_id)

    file_name3 = '2023Q1'
    ids3 = os.listdir(r"/mnt/algo-storage-server/Dataset/2D/外部收集/01_修复/{}".format(file_name3))
    for test_id in ids3:
        test_one(file_name3, test_id)

    # test_one_file()
