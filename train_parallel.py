import os

from train import train

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    os.makedirs("./save_weights", exist_ok=True)
    os.makedirs("./log", exist_ok=True)
    train()