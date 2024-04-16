import os
import shutil
import time
import cv2

import onnxruntime
import numpy as np
import tqdm

# Load ONNX model
onnx_file_name = "./save_weights/efficientnet_b1_best_model.onnx"
# onnx_file_name = "./save_weights/no_encrypted_strm_new.onnx"

session = onnxruntime.InferenceSession(onnx_file_name)

source_path = "/mnt/algo_storage_server/UNet/wyh_all/"
result_path = '/mnt/algo_storage_server/UNet/onnx_predict3/'

# if os.path.exists(result_path):
#     shutil.rmtree(result_path)
# os.mkdir(result_path)


for file in os.listdir(source_path):
    print(file)
    paths = os.listdir(os.path.join(source_path, file))
    save_path = os.path.join(result_path, file)
    if os.path.exists(save_path):
        continue
        # shutil.rmtree(save_path)
    os.mkdir(save_path)
    start_time = time.time()

    i = 0
    for p in tqdm.tqdm(paths):
        i = i + 1
        if i % 2 == 0:
            pass
        # Compute ONNX model output
        path = os.path.join(source_path, file, p)
        origin_image = cv2.imread(path)
        original_height, original_width, i = origin_image.shape
        image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = np.array(image, np.float32)
        image = image / 127.5 - 1

        image = np.expand_dims(image, 0)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: image})

        # Compare ONNX model output with golden image
        output_image = result[0].argmax(3).squeeze(0)
        output_image[output_image == 1] = 64
        output_image[output_image == 2] = 129
        output_image[output_image == 3] = 192
        output_image[output_image == 4] = 255
        # output_image[output_image == 1] = 129
        # output_image[output_image == 2] = 192
        # output_image[output_image == 3] = 255
        prediction = output_image.astype(np.uint8)

        predict_result = cv2.resize(prediction, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(save_path, os.path.splitext(os.path.basename(path))[0] + "_origin.png"),
                    origin_image)
        cv2.imwrite(os.path.join(save_path, os.path.splitext(os.path.basename(path))[0] + "_predict.png"),
                    predict_result)

    total_time = time.time() - start_time
    print("time {}s, fps {}".format(total_time, len(paths) / total_time))
