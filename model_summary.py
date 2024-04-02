from torchsummary import summary
from parse_args import parse_args, getModel

args = parse_args()
model = getModel(args)
summary(model, (3, args.image_size, args.image_size))
# args = parse_args()
# model = getModel(args)

# print("*"*50)
# print("需要冻结的网络层为 {}".format(freeze_layer))
# for k, v in model.named_parameters():
#     print("当前参数名称 {}".format(k))
#     v.requires_grad = True  # train all layers
#     if k.startswith("backbone"):
#         print('freezing {}'.format(k))
#         v.requires_grad = False
# print("*"*50)
# print("检查模型参数是否冻结成功")
# # 检查是否冻结成功
# for k, v in model.named_parameters():
#     if v.requires_grad:
#         print("可训练的模型参数名称 {}".format(k))
#     else:
#         print("已被冻结的模型参数名称 {}".format(k))
#
# print("*"*50)

