import glob
import os
import shutil

src = r"/mnt/algo-storage-server/UNet/Dataset14/data/augmentation_train/*.npy"
dst = r"./data/augmentation_train"

os.makedirs(dst, exist_ok=True)

paths = glob.glob(src)
print(len(paths))
for path in paths:
    print(path, os.path.join(dst, os.path.basename(path)))
    shutil.copy(path, os.path.join(dst, os.path.basename(path)))
