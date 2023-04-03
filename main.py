from dataset import ImageDataset
from torch.utils.data import DataLoader,random_split
from resnet import ResNet
from lenet import LeNet
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from accelerate import Accelerator
import os
from PIL import Image


accelerator = Accelerator()
parser = argparse.ArgumentParser(description='Xiao is not handsome')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--norm', type=int, default=1, help="which layer to add normalization: 1-8")
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--dropout', type=int, default=0, help="bool: whether or not to use drop out")
parser.add_argument('--weight_decay', type=int, default=0, help="bool: whether or not to use weight decay (L2 regularization)")
parser.add_argument('--norm_type', type=str, default="LN",help="BN for batchnorm, LN for LayerNorm")
parser.add_argument('--opt', type=str, default="adam", help="optimizer type: adam or sgd")
parser.add_argument('--activation', type=str, default="relu", help="leakyrelu, relu, sigmoid, tanh")
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
}
writer= SummaryWriter(log_dir="./nets/tfboard/run1")
base_dir="./dataset/"
device=config["cuda"]
device=torch.device(f"cuda:{device}")

def valid(net,val_dataloader,epoch):
    with torch.no_grad():
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
        print(f"[epoch:{epoch}]","Val loss",epoch_loss/len(val_dataloader),"Val acc",correct/total)
        return epoch_loss/len(val_dataloader),correct/len(val_dataloader)
    
def train(net, trainloader, valloader):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',patience=5,verbose=True)
    
    for epoch in range(config["epoch"]):
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
            correct+=(gt==cls).sum().item()
            total+=gt.size(0)
            loss.backward()
            opt.step()
            epoch_loss+=loss.item()
        scheduler.step(epoch_loss/len(trainloader))
        writer.add_scalar("loss",epoch_loss/len(trainloader),epoch)
        writer.add_scalar("acc",correct/len(trainloader),epoch)
        print(f"[epoch:{epoch}]","loss",epoch_loss/len(trainloader),"acc",correct/total)
        if epoch%5==0:
            test_dataset = ImageDataset(base_dir + 'test',device=device)
            testloader = DataLoader(test_dataset, batch_size=config["batch"], shuffle=True, num_workers=2)
            val_loss,val_acc=valid(net,testloader,epoch)
            # val_loss,val_acc=valid(net,valloader,epoch)
            writer.add_scalar("Val loss",val_loss,epoch)
            writer.add_scalar("Val acc",val_acc,epoch)

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

            # Save wrongly classified images
            for j in range(img.size(0)):
                if cls[j] != gt[j]:
                    img_j = img[j].cpu()
                    img_j = img_j * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                        [0.485, 0.456, 0.406]).view(3, 1, 1)
                    img_j = (img_j * 255).to(torch.uint8)
                    label_j = gt[j].item()
                    cls_j = cls[j].item()
                    filename = f"wrong_{i}_{j}_label{label_j}_class{cls_j}.png"
                    folder = "plots/wrong"
                    os.makedirs(folder, exist_ok=True)
                    Image.fromarray(img_j.permute(1, 2, 0).numpy()).save(os.path.join(folder, filename))
            
        print("Test acc",correct/total)
        
if __name__=="__main__":
    net = LeNet(2,config)
    dataset = ImageDataset(base_dir + config["mode"],device=device)
    trainset, valset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
    trainloader = DataLoader(trainset, batch_size=config["batch"], shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=config["batch"], shuffle=True, num_workers=2)

    dataset = ImageDataset(base_dir + 'test',device=device)
    testloader = DataLoader(dataset, batch_size=config["batch"], shuffle=True, num_workers=2)

    train(net,trainloader,val_loader)
    test(net,testloader)
    writer.close()