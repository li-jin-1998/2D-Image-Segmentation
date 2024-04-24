from PIL import Image

image = Image.open('./test.png')
image.show()

def enhance_h(image, hue=6):
    # 将图像转换为 HSV 模式
    image = image.convert('HSV')
    print(image)
    # 调整图像的色相
    # hue = -6  # 调整值范围为 0-360，0 表示原始色相，180 表示相反色相
    image = image.point(lambda x: (x + hue)% 360 )
    # # 获取图像的像素数据
    # pixels = image.load()
    #
    # # 改变色相
    # for x in range(image.width):
    #     for y in range(image.height):
    #         pixel = pixels[x, y]
    #         h, s, v = pixel
    #         # 修改 H 值以改变色相
    #         new_h = h + hue  # 例如，将色相增加 10
    #         pixels[x, y] = (new_h, s, v)

    # 将图像转换回 RGB 模式
    image = image.convert('RGB')
    return image


image = enhance_h(image, 8)
# 保存修改后的图像
image.save('adjusted_image.jpg')

# 显示修改后的图像
image.show()
