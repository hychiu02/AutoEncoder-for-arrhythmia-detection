import matplotlib.pyplot as plt
from typing import Callable, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
from torch.utils.data import DataLoader


class ProgressBar:
    last_length = 0

    @staticmethod
    def show(prefix: str, postfix: str, current: int, total: int, newline: bool = False) -> None:
        progress = (current + 1) / total
        if current == total:
            progress = 1

        current_progress = progress * 100
        progress_bar = '=' * int(progress * 20)

        message = ''

        if len(prefix) > 0:
            message += f'{prefix}, [{progress_bar:<20}]'

            if not newline:
                message += f' {current_progress:6.2f}%'

        if len(postfix) > 0:
            message += f', {postfix}'

        print(f'\r{" " * ProgressBar.last_length}', end='')
        print(f'\r{message}', end='')

        if newline:
            print()
            ProgressBar.last_length = 0
        else:
            ProgressBar.last_length = len(message) + 1


class BaseTrainer:
    def __init__(self) -> None:
        self.device = 'cpu'

    def train(self) -> None:
        raise NotImplementedError('train not implemented')

    def test(self) -> None:
        raise NotImplementedError('test not implemented')

    @property
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


class VATrainer(BaseTrainer):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer, criterion: Callable, model_name) -> None:
        super(VATrainer, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_name = model_name

        self.net = self.net.to(self.device)

    def _step(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        outputs = self.net(x)

        running_acc = (outputs.argmax(1) == y).type(torch.float).sum()

        running_loss = self.criterion(outputs, y)

        return running_loss, running_acc

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader = None, scheduler: Any = None) -> None:
        epoch_length = len(str(epochs))

        stale = 0

        best_acc = 0.0
        
        plt_train_loss = []
        plt_train_acc = []
        plt_val_loss = []
        plt_val_acc = []

        for epoch in range(epochs):
            self.net.train()

            loss = 0.0
            acc = 0.0

            for i, batch in enumerate(train_loader):
                input_data, labels = batch

                self.optimizer.zero_grad()

                running_loss, running_acc = self._step(x=input_data, y=labels)

                running_loss.backward()
                self.optimizer.step()

                loss += running_loss.item()
                acc += running_acc.item()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = f'loss: {running_loss.item():.3f}'
                # ProgressBar.show(prefix, postfix, i, len(train_loader))

            loss /= len(train_loader)
            acc /= len(train_loader.dataset)

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = f'loss: {loss:.3f}, acc: {acc:.3f}'
            # ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), newline=True)

            if val_loader:
                val_loss, val_acc = self.test(val_loader)

                # plt_val_loss.append(val_loss)
                # plt_val_acc.append(val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc
                    stale = 0
                else:
                    stale += 1
                    # if stale > 50:
                    #     break

            if scheduler:
                scheduler.step()
            
            # plt_train_loss.append(loss)
            
            # plt_train_acc.append(acc)
            
        # plt.plot(plt_train_loss)
        # plt.plot(plt_val_loss)
        # plt.title('Loss')
        # plt.legend(['train', 'valation'])
        # plt.savefig('loss.png')
        # plt.close()
        
        # plt.plot(plt_train_acc)
        # plt.plot(plt_val_acc)
        # plt.title('Accuracy')
        # plt.legend(['train', 'valation'])
        # plt.savefig('Accuracy.png')
        # plt.close()

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> None:
        self.net.eval()

        loss = 0.0
        acc = 0.0

        for i, batch in enumerate(test_loader):
            input_data, labels= batch
            running_loss, running_acc = self._step(x=input_data, y=labels)

            loss += running_loss.item()
            acc += running_acc.item()

            prefix = 'Test'
            postfix = f'loss: {running_loss.item():.3f}'
            # ProgressBar.show(prefix, postfix, i, len(test_loader))

        loss /= len(test_loader)
        acc /= len(test_loader.dataset)

        prefix = 'Test'
        postfix = f'loss: {loss:.3f}, acc: {acc:.3f}'
        # ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)

        return loss, acc
    
    @torch.no_grad()
    def inference(self, data_loader: DataLoader) -> None:
        self.net.eval()
        
        preds = []
        
        for i, batch in enumerate(data_loader):
            input_data, _ = batch
            
            input_data = input_data.to(self.device)
            
            pred = self.net(input_data)

            pred = pred > 0.5

            preds += pred.numpy().tolist()
        #     ProgressBar.show('Inference', '', i, len(data_loader))

        # ProgressBar.show('Inference', 'done', len(data_loader), len(data_loader), newline=True)
        
        return preds

    @property
    @torch.no_grad()
    def weights(self) -> dict:
        return {'net': self.net}
