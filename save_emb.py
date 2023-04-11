from dataset import ImageDataset
from torch.utils.data import DataLoader,random_split
from resnet import ResNet
from lenet import LeNet
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
from accelerate import Accelerator
import os
from PIL import Image
import numpy as np
import pickle
accelerator = Accelerator()
parser = argparse.ArgumentParser(description='Xiao is handsome!!!!!!!!!!!!!!!!!!!!!!!!')
parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--batch', type=int, default=32)
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

base_dir="/home/xiao/code/CS5242/dataset_1/"
device=config["cuda"]
device=torch.device(f"cuda:{device}")

def train(net, trainloader, valloader):
    net.eval()
    net.to(device)  
    with torch.no_grad():  
        emb1_ls=[]
        emb2_ls=[]
        gt_ls=[]
        for i,(img,gt) in enumerate(trainloader):
            img=img.to(device)
            gt=gt

            emb1,emb2=net.get_emb(img)
            emb1_ls.append(emb1.detach().cpu())
            emb2_ls.append(emb2.detach().cpu())
            gt_ls.append(gt)
            

    return torch.cat(emb1_ls).numpy(),torch.cat(emb2_ls).numpy(),torch.cat(gt_ls).numpy()
           

def test(net, testloader):
    net.eval()
    with torch.no_grad():  
        net.to(device)
        correct=0
        total=0
        for i,data in enumerate(testloader):
            img,gt=data
            img=img.to(device)
            gt=gt.to(device)
            y=net(img)
           
            cls=torch.argmax(y,dim=1)
            correct+=(gt==cls).sum().item()
            total+=gt.size(0)

        
            
        print("Test acc",correct/total)
        
if __name__=="__main__":
    net = LeNet(2,config)
    dataset = ImageDataset(base_dir + config["mode"],device=device,config=config,train=True)
    trainset, valset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
    trainloader = DataLoader(trainset, batch_size=config["batch"], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=config["batch"], shuffle=True, num_workers=2)

    # dataset = ImageDataset(base_dir + 'test',device=device,config=config,train=False)
    # testloader = DataLoader(dataset, batch_size=config["batch"], shuffle=True, num_workers=2)
    net.load_state_dict(torch.load("/home/xiao/code/CS5242/CS5242/model/model.pth"))
    emb1,emb2,gt=train(net,trainloader,val_loader)
    lalala={"tensor1":emb1,"tensor2":emb2,"labels":gt}
    with open ("/home/xiao/code/CS5242/CS5242/model/emb.pkl", 'wb') as f:
        pickle.dump(lalala, f, protocol=4)
    # np.savez("/home/xiao/code/CS5242/CS5242/model/emb.npy",lalala)
   
   
    