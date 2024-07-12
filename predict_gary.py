import glob
import os
import shutil
import time

import cv2
import numpy as np
import torch
import tqdm

from parse_args import parse_args, get_model, get_best_weight_path, get_device

CONVERT_COLOR = True


def create_custom_colormap():
    custom_colormap = np.zeros((5, 3), dtype=np.uint8)

    custom_colormap[0] = [0, 0, 0]
    custom_colormap[1] = [64, 64, 64]
    custom_colormap[2] = [129, 129, 129]
    custom_colormap[3] = [245, 224, 64]
    custom_colormap[4] = [255, 255, 255]

    return custom_colormap


def gray_to_color_mask( gray_image):
    custom_colormap = create_custom_colormap()

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
    return mask


def predict_gray(result_path):
    args = parse_args()
    device = get_device()

    # create model
    model = get_model(args)
    # load weights
    weights_path = get_best_weight_path(args)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # torch.save(model.state_dict(), "save_weights/{}_predict_model.pth".format(args.arch))
    predict_image_names = glob.glob(
        "/mnt/algo-storage-server/Workspaces/fangqi/AS connect内部下载数据/种植杆数据/scan转化ok/240529704003976420352/*")[
                          :10]
    # predict_image_names = glob.glob("/mnt/algo-storage-server/Workspaces/lijin/implant_test/*")[::5]
    # predict_image_names = glob.glob("/mnt/algo-storage-server/Workspaces/lijin/images/*")[::5]
    # predict_image_names = glob.glob(r"/home/lj/PycharmProjects/Data/checked/*].png")
    # predict_image_names = glob.glob(args.data_path + "/test/image/*")[::4]
    predict_image_names.sort()

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_path in tqdm.tqdm(predict_image_names):
            # if '004' not in img_path:
            #     continue
            # load image
            original_img = cv2.imread(img_path)
            if original_img is None:
                print("load image error:", img_path)
                continue
            original_img2 = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            image = cv2.resize(original_img2, (args.image_size, args.image_size),
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

            dst = os.path.join(result_path, os.path.splitext(os.path.basename(img_path))[0] + "_predict.png")

            cv2.imwrite(dst, predict_result)
            shutil.copy(str(img_path), dst.replace('predict', 'image'))

            if 'test' in img_path:
                origin_mask = str(img_path).replace('image/', 'mask/').replace("IMAGE", "MASK")
                if os.path.exists(origin_mask):
                    shutil.copy(origin_mask, dst.replace('predict', 'mask'))
    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(predict_image_names) / total_time))


if __name__ == '__main__':
    from others.open_dir import open_directory

    result_path = './visualization'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    predict_gray(result_path)
    open_directory(result_path)
