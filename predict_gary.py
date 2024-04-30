import glob
import os
import shutil
import time

import cv2
import numpy as np
import torch
import tqdm

from parse_args import parse_args, get_model, get_best_weight_path, get_device
from preprocess import pre_process


def mask_postprocessing(mask, w, h):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # mask[mask == 1] = 64
    mask[mask == 1] = 129
    mask[mask == 2] = 192
    mask[mask == 3] = 255
    return mask


def predict_gray():
    args = parse_args()
    device = get_device()

    # create model
    model = get_model(args)
    # load weights
    weights_path = get_best_weight_path(args)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # torch.save(model.state_dict(), "save_weights/{}_predict_model.pth".format(args.arch))

    predict_image_names = glob.glob(args.data_path + "/test/*image.*")[::10]
    # predict_image_names = glob.glob("/mnt/algo_storage_server/UNet/Dataset/implant2/*.*")[::4]
    # predict_image_names = glob.glob("/mnt/algo_storage_server/UNet/Dataset/image/*.*")[::20]
    predict_image_names.sort()
    result_path = './visualization'
    # if os.path.exists(result_path):
    #     shutil.rmtree(result_path)
    # os.mkdir(result_path)

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for image_path in tqdm.tqdm(predict_image_names):
            # load image
            depth_path = image_path.replace('image', 'depth')
            image, depth,_ = pre_process(image_path, depth_path, None, args.image_size)
            original_img = cv2.imread(image_path)
            # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            #
            # image = cv2.resize(original_img, (args.image_size, args.image_size),
            #                    interpolation=cv2.INTER_LINEAR)
            # image = image / 127.5 - 1
            image = torch.Tensor(image)
            image = image.permute(2, 0, 1)
            image = torch.unsqueeze(image, dim=0)

            # depth = cv2.imread(image_path.replace('image', 'depth'), cv2.IMREAD_UNCHANGED)
            # depth = cv2.resize(depth, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
            # depth = np.array(depth, np.float32)
            # depth[depth > 0] = (depth[depth > 0] - 100) / 30
            depth = torch.Tensor(depth[..., np.newaxis])
            depth = depth.permute(2, 0, 1)
            depth = torch.unsqueeze(depth, dim=0)

            output = model(image.to(device), depth.to(device))
            if isinstance(output, dict):
                output = output['out']
            prediction = output.argmax(1).squeeze(0)

            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            predict_result = mask_postprocessing(prediction, original_img.shape[1], original_img.shape[0])

            dst = os.path.join(result_path, os.path.splitext(os.path.basename(image_path))[0] + "_predict2.png")
            cv2.imwrite(dst, predict_result)
            # shutil.copy(str(image_path), dst.replace('predict', ' image'))

    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(predict_image_names) / total_time))


if __name__ == '__main__':
    predict_gray()
