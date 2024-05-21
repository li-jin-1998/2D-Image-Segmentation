import argparse

import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


def get_best_weight_path(args):
    weights_path = "save_weights/{}_best_model.pth".format(args.arch)
    print("best weight: ", weights_path)
    return weights_path


def get_latest_weight_path(args):
    weights_path = "save_weights/{}_latest_model.pth".format(args.arch)
    # print(weights_path)
    return weights_path


efficientnet_dict = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                     'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                     'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s']


def get_model(args):
    print('**************************')
    print(f'model:{args.arch}\n'
          f'epoch:{args.epochs}\n'
          f'batch size:{args.batch_size}\n'
          f'image size:{args.image_size}')
    print('**************************')
    device = get_device()
    if args.arch == 'unet':
        from network.UNet import UNet
        model = UNet(in_channels=3, num_classes=args.num_classes, base_c=32).to(device)
    if args.arch == 'mobilenet':
        from network.mobilenet_unet import MobileV3UNet
        model = MobileV3UNet(num_classes=args.num_classes, pretrain_backbone=True).to(device)
    if args.arch == 'efficientnet' or args.arch in efficientnet_dict:
        from network.efficientnet_unet import EfficientUNet
        model = EfficientUNet(num_classes=args.num_classes, pretrain_backbone=True,
                              model_name=args.arch, deep_supervision=args.deep_supervision).to(device)
    # if args.arch == 'efficientnet2':
    #     from efficientunet import get_efficientunet_b1
    #     model = get_efficientunet_b1(out_channels=args.num_classes, concat_input=True, pretrained=True).to(device)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1',
                        help='unet/u2net/deeplab/mobilenet/efficientnet/efficientnet_v2_s')
    parser.add_argument("--data_path", default="/mnt/algo_storage_server/UNet/Dataset14/data", help="root")
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")
    # Optimizer options
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume', default=1, help='resume from checkpoint')
    parser.add_argument('--deep_supervision', default=False, help='deep_supervision training')
    parser.add_argument('--multi_scale', default=False, help='multi-scale training')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best metric weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args
