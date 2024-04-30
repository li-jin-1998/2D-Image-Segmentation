import sys

import numpy as np
import torch
import tqdm
from torch.nn.functional import cross_entropy

import utils.distributed_utils as utils
from utils.loss import dice_loss, build_target


def criterion(inputs, target, loss_weight=None, num_classes: int = 3, label_smoothing: float = 0.1):
    losses = {}
    if not isinstance(inputs, dict):
        inputs = {'out': inputs}
    # loss_weight = torch.as_tensor([1, 2, 2, 2, 1], device="cuda")
    for name, x in inputs.items():
        target = build_target(target, num_classes)
        a = 0.3
        loss = (1 - a) * cross_entropy(x, target, weight=loss_weight, label_smoothing=label_smoothing)
        + a * dice_loss(x, target, multiclass=True)
        # Flooding
        # b = 0.28
        # loss = (loss - b).abs() + b
        losses[name] = loss

    return losses['out']


def evaluate(epoch_num, model, data_loader, device, num_classes):
    model.eval()

    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes)
    val_loss = []
    with torch.no_grad():
        data_loader = tqdm.tqdm(data_loader, file=sys.stdout)
        for image, depth, target in data_loader:
            image, depth, target = image.to(device), depth.to(device), target.to(device)
            output = model(image, depth)
            # output = model(image)
            loss = criterion(output, target, num_classes=num_classes)
            if isinstance(output, dict):
                output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
            val_loss.append(loss.item())
            data_loader.desc = f"[val epoch {epoch_num}] loss: {np.mean(val_loss):.4f}"
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item(), np.mean(val_loss), confmat.get_miou()


def train_one_epoch(epoch_num, model, optimizer, data_loader, device, num_classes,
                    lr_scheduler, scaler=None):
    model.train()

    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes)

    train_loss = []
    data_loader = tqdm.tqdm(data_loader, file=sys.stdout)
    for image, depth, target in data_loader:
        image, depth, target = image.to(device), depth.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, depth)
            # output = model(image)
            loss = criterion(output, target, num_classes=num_classes)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        if isinstance(output, dict):
            output = output['out']
        confmat.update(target.flatten(), output.argmax(1).flatten())
        dice.update(output, target)
        train_loss.append(loss.item())

        data_loader.desc = f"[train epoch {epoch_num}] loss: {np.mean(train_loss):.4f} "
    lr = optimizer.param_groups[0]["lr"]
    # lr_scheduler.step()
    return np.mean(train_loss), dice.value.item(), confmat.get_miou(), lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
