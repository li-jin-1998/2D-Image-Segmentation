import torch
import time
from timm import create_model
import network.RepNeXt
import utils
torch.autograd.set_grad_enabled(False)

T0 = 5
T1 = 10

def throughput(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    while time.time() - start < T0:
        model(inputs)
    timing = []
    torch.cuda.synchronize()
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        else:
            replace_batchnorm(child)

device = "cuda:0"

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--model', default='repnext_m1', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--batch-size', default=2048, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    resolution = args.resolution
    torch.cuda.empty_cache()
    inputs = torch.randn(batch_size, 3, resolution,
                            resolution, device=device)
    model = create_model(model_name, num_classes=1000)
    replace_batchnorm(model)
    model.to(device)
    model.eval()
    throughput(model_name, model, device, batch_size, resolution=resolution)