import glob
import os
import shutil
import time
import cv2
import torch
import tqdm
import numpy as np

from parse_args import parse_args, get_model


def mask_postprocessing(mask, w, h):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask[mask == 1] = 129
    mask[mask == 2] = 192
    mask[mask == 3] = 255
    return mask


def mask_postprocessing2(mask, w, h):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask[mask == 1] = 64
    mask[mask == 2] = 129
    mask[mask == 3] = 192
    mask[mask == 4] = 255
    return mask


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = get_model(args)
    weights_path = "./save_weights/{}_best_model.pth".format(args.arch)
    # load weights
    print(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    torch.save(model.state_dict(), "save_weights/{}_predict_model.pth".format(args.arch))
    # predict_image_names = glob.glob(args.data_path + "/augmentation/image/*.*")[x:x + 200]
    predict_image_names = glob.glob("/mnt/algo_storage_server/UNet/Dataset/implant2/*.*")[::3]
    # predict_image_names = glob.glob("/mnt/algo_storage_server/UNet/Dataset/1/*.*")
    result_path = './visualization'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.mkdir(result_path)

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_path in tqdm.tqdm(predict_image_names):
            if 'image' not in img_path and 'implant' in img_path and '_7_' not in img_path:
                continue
            # load image
            original_img = cv2.imread(img_path)
            original_img2 = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            original_height, original_width, _ = original_img2.shape
            original_img2 = cv2.resize(original_img2, (args.image_size, args.image_size),
                                       interpolation=cv2.INTER_LINEAR)
            original_img2 = original_img2 / 127.5 - 1
            img2 = torch.Tensor(original_img2)
            img2 = img2.permute(2, 0, 1)
            img2 = torch.unsqueeze(img2, dim=0)

            output = model(img2.to(device))
            prediction = output['out'].argmax(1).squeeze(0)

            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            predict_result = mask_postprocessing(prediction, original_width, original_height)
            dst = os.path.join(result_path, os.path.splitext(os.path.basename(img_path))[0] + "_predict.png")
            cv2.imwrite(dst, predict_result)

            shutil.copy(str(img_path), dst.replace('predict', ' image'))

    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(predict_image_names) / total_time))


if __name__ == '__main__':
    main()
