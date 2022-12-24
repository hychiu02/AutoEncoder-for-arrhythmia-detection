import time

import pandas as pd
# pytorch packages
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from Dataset import VA_Dataset
from Module import Autoencoder
from Trainer import VATrainer


def main() -> None:
    data_path = '../data/'
    bs = 1
    # num_workers = len(os.sched_getaffinity(0))
    num_workers = 0
    epochs = 100
    print("Start loading training data")
    
    train_set = VA_Dataset(data_path + 'train')
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    print("Finish loading training data")

    print("Start loading validation data")
    val_set = VA_Dataset(data_path + 'val')
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    print("Finish loading training data")
    print("Finish loading validation data")

    print("Start loading net and config")
    net = Autoencoder()

    optimizer = optim.SGD(net.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    print("Finish loading net")
    print("Start initial trainer")
    trainer = VATrainer(net, optimizer, criterion, "torch_test")

    print("Start training")
    start_t = time.time()
    trainer.train(epochs, train_loader, val_loader)
    end_t = time.time()
    print("Training time: {}".format(end_t - start_t))

    print('Start inference')
    start_t = time.time()
    trainer.inference(train_loader)
    end_t = time.time()
    print("Inference time: {}".format(end_t - start_t))

if __name__ == '__main__':
    torch.manual_seed(123)

    main()
