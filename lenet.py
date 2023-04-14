import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn import MultiheadAttention as attention

class LeNet(nn.Module):
   
    def __init__(self, num_classes,config):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(13456, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        # self.attention = nn.MultiheadAttention(embed_dim=13456, num_heads=8, dropout=config["dropout_rate"])
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
        
        self.dropout1 = nn.Dropout(p=config["dropout_rate"])
        self.dropout2 = nn.Dropout(p=config["dropout_rate"])
        self.dropout3 = nn.Dropout(p=config["dropout_rate"])
        self.dropout4 = nn.Dropout(p=config["dropout_rate"])
        
    def get_emb(self,x):
        out = self.conv1(x)
        tmp=out
        if self.config["norm"]==1:
            out=self.norm1(out)
            return tmp,out
    def forward(self, x,config):
        
        
        out = self.conv1(x)
      
        if config["norm"]==1:
            out=self.norm1(out)
            
        out = self.activation(out)
        out = F.max_pool2d(out, 2)
        if config["dropout"]:
            out = self.dropout1(out) # Applying dropout after max pooling

        out = self.conv2(out)
        if config["norm"]==2:
            out=self.norm2(out)
        out = self.activation(out)
        out = F.max_pool2d(out, 2)
        if config["dropout"]:
            out = self.dropout2(out) # Applying dropout after max pooling
        
        # out = out.view(out.size(0), -1).unsqueeze(0)
        
        # attn_output, _ = self.attention(out, out, out)
        # out = attn_output.squeeze(0).view(x.size(0), -1)
        # Transpose back and reshape
        

        out = out.view(self.config["batch"], -1)

        out=self.fc1(out)
        out = self.activation(out)
        if config["norm"]==3:
            out=self.norm3(out)
        if config["dropout"]:
            out = self.dropout3(out) # Applying dropout after ReLU

        out=self.fc2(out)
        if config["norm"]==4:
            out=self.norm4(out)
        out = self.activation(out)
        if config["dropout"]:
            out = self.dropout4(out) # Applying dropout after ReLU

        out = self.fc3(out)
        # out = torch.sigmoid(out)
        return out
    

class LeNet_cifar(nn.Module):
   
    def __init__(self, num_classes,config):
        super(LeNet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
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
        
        
        out = self.conv1(x)
      
        if self.config["norm"]==1:
            out=self.norm1(out)
            
        out = self.activation(out)
        out = self.maxpool1(out)
        # if self.config["dropout"]:
        #     out = self.dropout(out) # Applying dropout after max pooling

        out = self.conv2(out)
        if self.config["norm"]==2:
            out=self.norm2(out)
        out = self.activation(out)
        out = self.maxpool2(out)
        # out = F.max_pool2d(out, 2)
        if self.config["dropout"]:
            out = self.dropout(out) # Applying dropout after max pooling
        
        # breakpoint()

        out = out.view(self.config["batch"], -1)

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
        out =  F.log_softmax(out, dim=1)
        return out