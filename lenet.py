import torch.nn as nn
import torch.nn.functional as F
import torch



class LeNet(nn.Module):
    '''
    Input: (512, 512, 3)
    Conv1: (512, 512, 32) -> MaxPool: (256, 256, 32)
    Conv2: (256, 256, 64) -> MaxPool: (128, 128, 64)
    Conv3: (128, 128, 128) -> MaxPool: (64, 64, 128)
    Conv4: (64, 64, 256) -> MaxPool: (32, 32, 256)
    Conv5: (32, 32, 512) -> MaxPool: (16, 16, 512)
    '''
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
            # self.norm3=nn.BatchNorm1d(16*29*29)
            self.norm3=nn.BatchNorm1d(120)
            self.norm4=nn.BatchNorm1d(84)
        elif self.config["norm_type"]=="LN":
            self.norm1=nn.LayerNorm([6,124,124])
            self.norm2=nn.LayerNorm([16,58,58])
            # self.norm3=nn.LayerNorm(16*29*29)
            self.norm3=nn.LayerNorm(120)
            self.norm4=nn.LayerNorm(84)
        
        if self.config["activation"] == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif self.config["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif self.config["activation"] == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.config["activation"] == "tanh":
            self.activation = nn.Tanh()
        
        self.dropout = nn.Dropout(p=0.5)
        
    def get_emb(self,x):
        out = self.conv1(x)
        tmp=out
        if self.config["norm"]==1:
            out=self.norm1(out)
            return tmp,out
    def forward(self, x):
        # breakpoint()
        
        out = self.conv1(x)
      
        if self.config["norm"]==1:
            out=self.norm1(out)
            
        out = self.activation(out)
        out = F.max_pool2d(out, 2)
        # if self.config["dropout"]:
        #     out = self.dropout(out) # Applying dropout after max pooling

        out = self.conv2(out)
        if self.config["norm"]==2:
            out=self.norm2(out)
        out = self.activation(out)
        out = F.max_pool2d(out, 2)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after max pooling
        
        

        out = out.view(out.size(0), -1)

        out=self.fc1(out)
        out = self.activation(out)
        if self.config["norm"]==3:
            out=self.norm3(out)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after ReLU

        out=self.fc2(out)
        if self.config["norm"]==4:
            out=self.norm4(out)
        out = self.activation(out)
        # if self.config["dropout"]:
        #     out = self.dropout(out) # Applying dropout after ReLU

        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out