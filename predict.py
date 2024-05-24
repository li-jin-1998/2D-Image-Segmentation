import copy
import glob
import os
import random
import shutil
import time

import PIL
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image

from parse_args import parse_args, get_model, get_device
from utils.preprocess import preprocessing, COLORS


def main():
    args = parse_args()

    device = get_device()

    # create model
    model = get_model(args)
    weights_path = "./save_weights/{}_best_model.pth".format(args.arch)
    # load weights
    print(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    torch.save(model.state_dict(), "save_weights/{}_predict_model.pth".format(args.arch))
    x = random.randint(1, 1000)
    print(x)
    predict_image_names = glob.glob("/mnt/algo_storage_server/UNet/Dataset/image/*.*")[x:x + 200]
    # predict_image_names = glob.glob("/mnt/algo_storage_server/ScanSceneClassification/dataset/test/intra/*.*")[x:x + 200]
    if os.path.exists("./visualization"):
        shutil.rmtree("./visualization")
    os.mkdir("./visualization")

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_path in tqdm.tqdm(predict_image_names):
            # load image
            original_img = Image.open(img_path).convert('RGB')
            original_predict_image = copy.deepcopy(original_img)
            original_width, original_height = original_img.size
            original_img = preprocessing(original_img, args.image_size)
            img = torch.Tensor(original_img)
            img = img.permute(2, 0, 1)
            img = torch.unsqueeze(img, dim=0)

            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)

            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            predict_result = cv2.resize(prediction,
                                        (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST)

            predict_result_image = np.reshape(
                np.array(COLORS, np.uint8)[np.reshape(predict_result, [-1])],
                [original_height, original_width, -1])

            predict_result_image = PIL.Image.fromarray(predict_result_image)
            predict_result_image = PIL.Image.blend(original_predict_image,
                                                   predict_result_image, 0.4)
            predict_result_image.save(os.path.join("./visualization",
                                                   os.path.splitext(os.path.basename(img_path))[0] + "_predict.png"))

            mask_path = img_path.replace("image", "mask")
            mask = cv2.imread(mask_path)
            mask_result_image = PIL.Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            mask_result_image = PIL.Image.blend(original_predict_image,
                                                mask_result_image, 0.4)
            mask_result_image.save(os.path.join("./visualization",
                                                os.path.splitext(os.path.basename(img_path))[0] + "_mask.png"))
    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, 100 / total_time))


if __name__ == '__main__':
    main()
