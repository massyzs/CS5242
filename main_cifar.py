from dataset import ImageDataset
from torch.utils.data import DataLoader,random_split
from resnet import ResNet
from lenet import LeNet,LeNet_cifar
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
accelerator = Accelerator()
parser = argparse.ArgumentParser(description='NA')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--norm', type=int, default=1, help="which layer to add normalization: 1-8")
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--dropout', type=int, default=1, help="bool: whether or not to use drop out")
parser.add_argument('--weight_decay', type=int, default=1, help="bool: whether or not to use weight decay (L2 regularization)")
parser.add_argument('--norm_type', type=str, default="BN",help="BN for batchnorm, LN for LayerNorm")
parser.add_argument('--opt', type=str, default="adam", help="optimizer type: adam or sgd")
parser.add_argument('--activation', type=str, default="relu", help="leakyrelu, relu, sigmoid, tanh")
parser.add_argument('--aug', type=int, default=1, help="bool: whether or not to use data augmentation")
parser.add_argument('--save', type=int, default=0, help="bool: whether or not to use data augmentation")
args = parser.parse_args()
config={
    "mode": "train",
    "batch": args.batch,
    "epoch": args.epoch,
    "lr": 1e-4,
    "cuda": args.cuda,
    "norm": args.norm,
    "norm_type": args.norm_type,
    "dropout": bool(args.dropout),
    "weight_decay": bool(args.weight_decay),
    "opt": args.opt,
    "activation": args.activation,
    "data_augmentation":bool(args.aug),
    "save":bool(args.save)
}
import torch
import torchvision
import torchvision.transforms as transforms

# Define data transformations
# transform = transforms.Compose([
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
   
    
# ])
transform =transforms.Compose([
                transforms.Resize((32, 32)),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

from datasets import load_dataset

cifar10_dataset = load_dataset("cifar10")

trainset = cifar10_dataset["train"]
testset = cifar10_dataset["test"]

# breakpoint()
# Classes in the CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.images = torch.tensor(np.array([np.array(img).reshape(3, 32, 32) for img in self.dataset["img"]]),dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.dataset["label"]),dtype=torch.long)
        # breakpoint()
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
            
        self.image=self.transform(self.images[idx])
       
            
        return self.image, self.labels[idx]
device=config["cuda"]
device=torch.device(f"cuda:{device}")

def valid(net,val_dataloader,epoch):
    with torch.no_grad():
        net.eval()
        criertion=nn.CrossEntropyLoss()
        net.to(device)
        epoch_loss=0
        correct=0
        total=0
        for i,(img,gt) in enumerate(val_dataloader):
            img=img.to(device)
            gt=gt.to(device)
            y=net(img)
            loss=criertion(y,gt)
            cls=torch.argmax(y,dim=1)
            correct+=(gt==cls).sum().item()
            total+=gt.size(0)
            epoch_loss+=loss.item()
        print(f"[epoch:{epoch}]","test loss",epoch_loss/len(val_dataloader),"test acc",correct/total)
        return epoch_loss/len(val_dataloader),correct/len(val_dataloader)
    
def train(net, trainloader, testloader):
    net.train()
    net.to(device)
    criertion=nn.CrossEntropyLoss()
    
    if config["opt"] == "sgd":
        if config["weight_decay"]:
            opt=optim.SGD(net.parameters(), lr=config["lr"], weight_decay=0.0005)
        else:
            opt=optim.SGD(net.parameters(), lr=config["lr"])
    else:
        if config["weight_decay"]:
            opt=optim.Adam(net.parameters(),lr=config["lr"],betas=(0.9, 0.999), weight_decay=0.0005)
        else:
            opt=optim.Adam(net.parameters(),lr=config["lr"],betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',patience=5,verbose=True,factor=0.8)
    
    for epoch in range(config["epoch"]):
        net.train()
        epoch_loss=0
        correct=0
        total=0
        for i,(img,gt) in enumerate(trainloader):
            img=img.to(device)
            gt=gt.to(device)
            opt.zero_grad()
            # breakpoint()
            y=net(img)
            
            cls=torch.argmax(y,dim=1)
            loss=criertion(y,gt)
            # breakpoint()
            correct+=(gt==cls).sum().item()
            total+=gt.size(0)
            loss.backward()
            opt.step()
            epoch_loss+=loss.item()
        scheduler.step(epoch_loss/len(trainloader))
       
        print(f"[epoch:{epoch}]","loss",epoch_loss/len(trainloader),"acc",correct/total)
        # if epoch%2==0 or epoch>=config["epoch"]-2:
            
        val_loss,val_acc=valid(net,testloader,epoch)
        # breakpoint()
            # val_loss,val_acc=valid(net,valloader,epoch)
    return net
           


        
if __name__=="__main__":
    net = LeNet_cifar(10,config)
    # dataset = ImageDataset(base_dir + config["mode"],device=device,config=config,train=True)
    # trainset, valset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
    # trainloader = DataLoader(trainset, batch_size=config["batch"], shuffle=True, num_workers=8,drop_last=True)
    # # val_loader = DataLoader(valset, batch_size=config["batch"], shuffle=True, num_workers=2)
    # test_dataset = ImageDataset(base_dir + 'test',device=device,config=config,train=False)
    # testloader = DataLoader(test_dataset, batch_size=config["batch"], shuffle=True, num_workers=8,drop_last=True)

    # dataset = ImageDataset(base_dir + 'test',device=device,config=config,train=False)
    # testloader = DataLoader(dataset, batch_size=config["batch"], shuffle=True, num_workers=2)
    train_set=CIFAR10Dataset(trainset,transform=transform)
    
    test_set=CIFAR10Dataset(testset,transform=transform)
    trainloader=DataLoader(train_set,batch_size=config["batch"],num_workers=8,shuffle=True,drop_last=True)
    testloader=DataLoader(test_set,batch_size=config["batch"],num_workers=8,shuffle=False,drop_last=True)

    net=train(net,trainloader,testloader)
    # torch.save(net.state_dict(),"/home/xiao/code/CS5242/CS5242/model/model.pth")
    # test(net,testloader)
    