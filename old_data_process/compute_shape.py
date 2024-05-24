import os

from PIL import Image
from tqdm import tqdm

data_path = "/mnt/algo_storage_server/UNet/Dataset/image"

i = 0
j = 0
k = 0
for path in tqdm(os.listdir(data_path)):
    image = Image.open(os.path.join(data_path, path))
    if image.size == (272, 240):
        i = i + 1
    if image.size == (204, 204):
        j = j + 1
    k = k + 1
print(i, j, i + j, k)
