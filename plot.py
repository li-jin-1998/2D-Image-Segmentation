import os

import matplotlib.pyplot as plt
import numpy as np


def loss_plot(args, train_loss, val_loss):
    num = len(train_loss)
    x = [i for i in range(1, num + 1)]
    plot_save_path = r'log/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path + str(args.arch) + '_' + str(
        args.epochs) + '_loss.png'
    plt.figure()
    plt.xlim([1, num + 1])
    plt.ylim([np.min(val_loss) / 2, np.max(val_loss) + 0.1])
    plt.plot(x, train_loss, 'r', label='train loss')
    plt.plot(x, val_loss, 'g', label='val loss')
    plt.legend()
    plt.savefig(save_loss, dpi=300)


def metrics_plot(arg, name, metrics_value):
    num = len(metrics_value)
    x = [i for i in range(1, num + 1)]
    plot_save_path = r'log/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + str(arg.arch) + '_' + str(
        arg.epochs) + '_' + name + '.png'
    plt.figure()
    plt.plot(x, metrics_value, 'b', label=str(name))
    plt.xlim([1, num + 1])
    plt.ylim([np.min(metrics_value) - 0.01, np.max(metrics_value) + 0.01])
    plt.legend()
    plt.savefig(save_metrics, dpi=300)


if __name__ == '__main__':
    from parse_args import parse_args

    args = parse_args()
    args.epochs = 5
    loss_plot(args, [0, 0.6, 0.8, 0.8, 0.92], [0, 0.3, 0.7, 0.86, 0.97])
    metrics_plot(args, "dice", [0.82, 0.86, 0.88, 0.92, 0.95])
