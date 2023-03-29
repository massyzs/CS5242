from dataset import ImageDataset
from torch.utils.data import DataLoader,random_split
from resnet import ResNet
from lenet import LeNet
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
config={
    "mode":"train",
    "batch":128,
    "device":0,
    "epoch":100,
    "lr":1e-4,
    "cuda":0
}
writer = SummaryWriter(log_dir="/home/xiao/code/CS5242/nets/tfboard/run1")
base_dir="/home/xiao/code/CS5242/dataset/"
device=config["cuda"]
device=torch.device(f"cuda:{device}")
def valid(net,data,epoch):
    with torch.no_grad():
        criertion=nn.CrossEntropyLoss()
        net.to(device)
        loader = DataLoader(data, batch_size=config["batch"], shuffle=True, num_workers=2)
        epoch_loss=0
        correct=0
        for i,data in enumerate(loader):
            img,gt=data
            img=img.to(device)
            gt=gt.to(device)
            y=net(img)
            loss=criertion(y,gt)
            cls=torch.argmax(y,dim=1)
            correct+=(gt==cls).sum().item()
            epoch_loss+=loss.item()
        print(f"[epoch:{epoch}]","Val loss",epoch_loss/len(loader),"Val acc",correct/len(loader))
        return epoch_loss/len(loader),correct/len(loader)
def train(net):
    net.to(device)
    criertion=nn.CrossEntropyLoss()
    opt=optim.Adam(net.parameters(),lr=config["lr"],betas=(0.9, 0.999))
    dataset = ImageDataset(base_dir + config["mode"],device=device)
    trainset, valset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
    trainloader = DataLoader(trainset, batch_size=config["batch"], shuffle=True, num_workers=2)
    for epoch in range(config["epoch"]):
        epoch_loss=0
        correct=0
        for i,data in enumerate(trainloader):
            img,gt=data
            img=img.to(device)
            gt=gt.to(device)
            opt.zero_grad()
            # breakpoint()
            y=net(img)
            
            cls=torch.argmax(y,dim=1)
            loss=criertion(y,gt)
            correct+=(gt==cls).sum().item()
            loss.backward()
            opt.step()
            epoch_loss+=loss.item()
        writer.add_scalar("loss",epoch_loss/len(trainloader),epoch)
        writer.add_scalar("acc",correct/len(trainloader),epoch)
        print(f"[epoch:{epoch}]"," loss",epoch_loss/len(trainloader),"acc",correct/len(trainloader))
        if epoch%5==0:
            val_loss,val_acc=valid(net,valset,epoch)
            writer.add_scalar("Val loss",val_loss,epoch)
            writer.add_scalar("Val acc",val_acc,epoch)


if __name__=="__main__":
    net = LeNet(2)
    train(net)
    writer.close()