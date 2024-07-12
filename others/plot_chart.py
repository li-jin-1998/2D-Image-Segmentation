import matplotlib.pyplot as plt

# 示例数据
models = [
    'DSNet', 'DSNet-Base', 'OCRNet', 'HRNetV2-W48', 'HRNetV2-W40', 'PIDNet-M',
    'DeepLabv3+', 'PSPNet', 'DeepLabv3', 'SegNext-T-Seg100', 'DDRNet23',
    'RTFormer-B', 'SFNet(ResNet-18)', 'AFFormer-B-Seg100', 'RegSeg'
]
params = [10, 20, 60, 50, 40, 30, 50, 55, 65, 12, 25, 18, 15, 10, 8]
accuracy = [80.5, 82.0, 81.5, 81.0, 80.0, 79.8, 78.5, 79.0, 77.0, 79.6, 78.9, 79.5, 79.0, 78.0, 78.5]

# 各种图标的样式
styles = ['r*', 'r*', 'yD', 'mp', 'mp', 'bo', 'rx', 'g^', 'rx', 'k^', 'gs', 'b+', 'rD', 'y^', 'c^']

plt.figure(figsize=(12, 8))

for model, param, acc, style in zip(models, params, accuracy, styles):
    plt.plot(param, acc, style, label=model)

# 为 DSNet 和 DSNet-Base 添加连线
plt.plot([10, 20], [80.5, 82.0], 'r-', linewidth=2)

# 添加标题和标签
plt.title('Params vs Accuracy')
plt.xlabel('Params (M)')
plt.ylabel('Accuracy (mIoU %)')

# 添加图例
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper right')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
