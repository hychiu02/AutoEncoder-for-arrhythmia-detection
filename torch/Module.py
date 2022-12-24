import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=1250, out_features=512),
            nn.Sigmoid(),
            nn.Linear(in_features=512, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=32),
            nn.Sigmoid(),
            nn.Linear(in_features=32, out_features=8),
            nn.Sigmoid(),
            nn.Linear(in_features=8, out_features=1)
        )

    def forward(self, input):
        out = self.fc(input)

        return out
