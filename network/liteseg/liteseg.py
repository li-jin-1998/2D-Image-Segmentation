"""https://github.com/tahaemara/LiteSeg/blob/master"""

from network.liteseg import liteseg_shufflenet as shufflenet
from network.liteseg import liteseg_darknet as darknet
from network.liteseg import liteseg_mobilenet as mobilenet


def liteSeg(backbone_network, n_classes, pretrained=False):
    if backbone_network == 'darknet':
        net = darknet.RT(n_classes=n_classes, pretrained=pretrained)
    elif backbone_network == 'shufflenet':
        net = shufflenet.RT(n_classes=n_classes, pretrained=pretrained)
    elif backbone_network == 'mobilenet':
        net = mobilenet.RT(n_classes=n_classes, pretrained=pretrained)
    else:
        raise NotImplementedError

    print("Using LiteSeg with", backbone_network)
    return net
