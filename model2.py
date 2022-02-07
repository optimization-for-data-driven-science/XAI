from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    ## This _init_weights function is not necessary
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            # print(m)
            # print(type(m))
            # nn.init.xavier_normal_(m.weight)
            # nn.init.constant_(m.bias, 1 / m.bias.numel())

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x