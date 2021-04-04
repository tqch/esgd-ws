import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.classifier(x)
        return out
