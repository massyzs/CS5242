import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes,config):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*29*29, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        self.norm1=nn.BatchNorm2d(6)
        self.norm2=nn.BatchNorm2d(16)
        self.norm3=nn.BatchNorm1d(16*29*29)
        self.norm4=nn.BatchNorm1d(120)
        self.norm5=nn.BatchNorm1d(84)
        self.config=config

    def forward(self, x):
        out = self.conv1(x)
        if self.config["norm"]==1:
            out=self.norm1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        if self.config["norm"]==2:
            out=self.norm2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        if self.config["norm"]==3:
            out=self.norm3(out)
        out=self.fc1(out)
        out = F.relu(out)
        if self.config["norm"]==4:
            out=self.norm4(out)
        out=self.fc2(out)
        if self.config["norm"]==5:
            out=self.norm5(out)
        out = F.relu(out)
        out = self.fc3(out)

        # return out, latent1, latent2, latent3
        return out