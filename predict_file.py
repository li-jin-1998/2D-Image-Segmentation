import glob
import os
import random
import shutil
import time
import copy
import PIL
import cv2
import torch
import tqdm
import numpy as np
from PIL import Image

from preprocess import preprocessing, COLORS
from parse_args import parse_args, get_model


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
    x = random.randint(1, 1000)
    print(x)
    predict_image_names = glob.glob("data/test/image/*.*")
    result_path="./data/test/predict"
    shutil.rmtree(result_path)
    os.mkdir(result_path)

    start_time = time.time()
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for img_path in tqdm.tqdm(predict_image_names):
            # load image
            original_img = Image.open(img_path).convert('RGB')
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
            predict_result_image.save(img_path.replace('image','predict'))

    total_time = time.time() - start_time
    print("time {}ms, fps {}".format(total_time, 1000 / total_time))


if __name__ == '__main__':
    main()
