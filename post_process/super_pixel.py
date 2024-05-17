import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries

src = r'../visualization'
image_paths = [p for p in os.listdir(src) if 'image.' in p]

image_path = os.path.join(src, image_paths[20])
print(image_path)
# 加载图像
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 生成超像素
segments = slic(image, n_segments=200, compactness=10, sigma=1, start_label=1)

# 初始分割
mask = cv2.imread(image_path.replace(' image', 'predict'))
initial_segments = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


# 合并超像素基于初始分割
def merge_superpixels(segments, initial_segments):
    new_labels = np.zeros_like(segments)
    for region in regionprops(segments):
        coords = region.coords
        majority_label = np.argmax(np.bincount(initial_segments[coords[:, 0], coords[:, 1]]))
        new_labels[coords[:, 0], coords[:, 1]] = majority_label
    return new_labels


# 合并后的分割结果
merged_segments = merge_superpixels(segments, initial_segments)

# 标记超像素边界
boundary_image = mark_boundaries(image, merged_segments, color=(0.1, 0.1, 0), mode='thick')

# 使用Matplotlib显示结果
fig, ax = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)

ax[0][0].imshow(image)
ax[0][0].set_title('Original Image')
ax[0][0].axis('off')

ax[0][1].imshow(mask)
ax[0][1].set_title('Original Mask')
ax[0][1].axis('off')

ax[1][0].imshow(mark_boundaries(image, segments))
ax[1][0].set_title('SLIC Superpixels')
ax[1][0].axis('off')

ax[1][1].imshow(boundary_image)
ax[1][1].set_title('Post-processed Segmentation')
ax[1][1].axis('off')

plt.tight_layout()
plt.savefig('superpixel.png', dpi=100)
plt.show()
