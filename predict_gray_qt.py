import sys
import os
import time
import shutil
import glob
import json

import cv2
import torch
import numpy as np
import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, \
    QLabel, QLineEdit, QCheckBox, QTextEdit, QHBoxLayout

from parse_args import parse_args, get_model, get_best_weight_path, get_device
from preprocess import pre_process

CONFIG_FILE = 'config.json'

def mask_postprocessing(mask, w, h):
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask[mask == 1] = 129
    mask[mask == 2] = 192
    mask[mask == 3] = 255
    return mask

class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Prediction App")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Directory input
        self.directory_layout = QHBoxLayout()
        self.directory_label = QLabel("Data Directory:", self)
        self.directory_layout.addWidget(self.directory_label)

        self.directory_input = QLineEdit(self)
        self.directory_layout.addWidget(self.directory_input)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.browse_directory)
        self.directory_layout.addWidget(self.browse_button)

        self.layout.addLayout(self.directory_layout)

        # With Depth checkbox
        self.with_depth_checkbox = QCheckBox("With Depth", self)
        self.layout.addWidget(self.with_depth_checkbox)

        # Predict button
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        # Output console
        self.output_console = QTextEdit(self)
        self.output_console.setReadOnly(True)
        self.layout.addWidget(self.output_console)

        # Load config
        self.load_config()

    def log_output(self, message):
        self.output_console.append(message)
        self.output_console.ensureCursorVisible()

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", self.last_open_dir)
        if directory:
            self.directory_input.setText(directory)
            self.last_open_dir = directory

    def predict(self):
        data_directory = self.directory_input.text()
        with_depth = self.with_depth_checkbox.isChecked()

        if not data_directory:
            self.log_output("Please select a data directory.")
            return

        args = parse_args()
        args.data_path = data_directory
        args.with_depth = int(with_depth)

        device = get_device()

        # create model
        model = get_model(args)
        # load weights
        weights_path = get_best_weight_path(args)
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)

        predict_image_names = glob.glob(args.data_path + "/*image.*")[:300]
        predict_image_names.sort()
        result_path = './visualization'

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        start_time = time.time()
        model.eval()  # 进入验证模式
        with torch.no_grad():
            for image_path in tqdm.tqdm(predict_image_names, file=sys.stdout):
                if args.with_depth:
                    depth_path = image_path.replace('image', 'depth')
                else:
                    depth_path = None

                image, depth, mask = pre_process(image_path, depth_path, None, args.image_size)
                original_img = cv2.imread(image_path)

                image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
                depth = torch.Tensor(depth[..., np.newaxis]).permute(2, 0, 1).unsqueeze(0)

                output = model(image.to(device), depth.to(device))
                if isinstance(output, dict):
                    output = output['out']
                prediction = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                predict_result = mask_postprocessing(prediction, original_img.shape[1], original_img.shape[0])

                dst = os.path.join(result_path, os.path.splitext(os.path.basename(image_path))[0] + "_predict.png")
                dst_image_path = os.path.join(result_path, os.path.splitext(os.path.basename(image_path))[0] + "_image.png")
                if not os.path.exists(dst_image_path):
                    shutil.copy(str(image_path), dst_image_path)
                cv2.imwrite(dst, predict_result)

                self.log_output(f"Processed: {os.path.basename(image_path)}")

        total_time = time.time() - start_time
        self.log_output(f"Prediction completed in {total_time:.2f} seconds.")
        self.save_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                self.last_open_dir = config.get('last_open_dir', '')
                self.directory_input.setText(self.last_open_dir)
        else:
            self.last_open_dir = ''
            self.directory_input.setText(self.last_open_dir)

    def save_config(self):
        config = {
            'last_open_dir': self.last_open_dir
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
