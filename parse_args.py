import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientnet_dict = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                     'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                     'efficientnet_b6', 'efficientnet_b7']


def getModel(args):
    print('**************************')
    print(f'model:{args.arch}\nepoch:{args.epochs}\nbatch size:{args.batch_size}\nimage size:{args.image_size}')
    print('**************************')
    if args.arch == 'unet':
        from network.UNet import UNet
        model = UNet(in_channels=3, num_classes=args.num_classes + 1, base_c=32).to(device)
    if args.arch == 'lraspp':
        from network.lraspp import lraspp_mobilenetv3_large
        model = lraspp_mobilenetv3_large(num_classes=args.num_classes + 1, pretrain_backbone=True).to(device)
    if args.arch == 'u2net':
        from network.U2Net import u2net_lite, u2net_full
        model = u2net_lite(args.num_classes + 1).to(device)
    if args.arch == 'deeplab':
        from network.deeplab_v3 import deeplabv3_mobilenetv3_large, deeplabv3_resnet50, deeplabv3_resnet101
        model = deeplabv3_mobilenetv3_large(aux=False, num_classes=args.num_classes + 1, pretrain_backbone=True).to(
            device)
    if args.arch == 'stdcnet':
        from network.BiseNet import BiSeNet
        model = BiSeNet('STDCNet813', n_classes=args.num_classes + 1).to(device)
    if args.arch == 'mobilenet':
        from network.mobilenet_unet import MobileV3UNet, MobileV2UNet
        model = MobileV3UNet(num_classes=args.num_classes + 1, pretrain_backbone=True).to(device)
    if args.arch == 'bisenetv2':
        from network.bisenet_v2 import BiSeNetV2
        model = BiSeNetV2(n_classes=args.num_classes + 1).to(device)
    if args.arch == 'resunet':
        from network.resunet import ResUNet
        model = ResUNet(num_classes=args.num_classes + 1).to(device)
    if args.arch == 'liteseg':
        from network.liteseg.liteseg import liteSeg
        model = liteSeg(backbone_network="darknet", n_classes=args.num_classes + 1).to(device)
    if args.arch == 'shufflenet':
        from network.shufflenet_unet import ShuffleUNet
        model = ShuffleUNet(num_classes=args.num_classes + 1, pretrain_backbone=True).to(device)
    if args.arch == 'efficientnet' or args.arch in efficientnet_dict:
        # args.arch = 'efficientnet_b0'
        from network.efficientnet_unet import EfficientUNet
        model = EfficientUNet(num_classes=args.num_classes + 1, pretrain_backbone=True,
                              model_name=args.arch).to(device)
    if args.arch == 'efficientnet2':
        from efficientunet import get_efficientunet_b0, get_efficientunet_b1
        model = get_efficientunet_b1(out_channels=args.num_classes + 1, concat_input=True, pretrained=True).to(device)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1',
                        help='unet/lraspp/u2net/deeplab/stdcnet/mobilenet/bisenetv2/resunet/efficientnet')
    parser.add_argument("--data_path", default="/mnt/algo_storage_server/UNet/Dataset9/data", help="root")
    # exclude background
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--resume', default=0, help='resume from checkpoint')
    parser.add_argument('--multi_scale', default=False, help='multi-scale training')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args
