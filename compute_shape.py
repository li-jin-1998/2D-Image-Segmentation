import os

from PIL import Image

data_path = "./data/train/image"

i = 0
j = 0
k = 0
for path in os.listdir(data_path):
    image = Image.open(os.path.join(data_path, path))
    if image.size == (272, 240):
        i = i + 1
    if image.size == (204, 204):
        j = j + 1
    k = k + 1
print(i, j, i + j, k)
