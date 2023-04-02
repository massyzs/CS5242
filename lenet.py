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

        self.config=config

        if self.config["norm_type"]=="BN":
            self.norm1=nn.BatchNorm2d(6)
            self.norm2=nn.BatchNorm2d(16)
            self.norm3=nn.BatchNorm1d(16*29*29)
            self.norm4=nn.BatchNorm1d(120)
            self.norm5=nn.BatchNorm1d(84)
        elif self.config["norm_type"]=="LN":
            self.norm1=nn.LayerNorm([6,124,124])
            self.norm2=nn.LayerNorm([16,58,58])
            self.norm3=nn.LayerNorm(16*29*29)
            self.norm4=nn.LayerNorm(120)
            self.norm5=nn.LayerNorm(84)
        
        if self.config["activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.config["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif self.config["activation"] == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.config["activation"] == "tanh":
            self.activation = nn.Tanh()
        

    def forward(self, x):
        # breakpoint()
        out = self.conv1(x)
        if self.config["norm"]==1:
            out=self.norm1(out)
        out = self.activation(out)
        out = F.max_pool2d(out, 2)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after max pooling
        out = self.conv2(out)
        if self.config["norm"]==2:
            out=self.norm2(out)
        out = self.activation(out)
        out = F.max_pool2d(out, 2)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after max pooling
        out = out.view(out.size(0), -1)
        if self.config["norm"]==3:
            out=self.norm3(out)
        out=self.fc1(out)
        out = self.activation(out)
        if self.config["norm"]==4:
            out=self.norm4(out)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after ReLU
        out=self.fc2(out)
        if self.config["norm"]==5:
            out=self.norm5(out)
        out = self.activation(out)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after ReLU
        out = self.fc3(out)

        return out