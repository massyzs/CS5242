# Please type following commond. CUDA_VISIBLE_DEVICES="The devices you want use"
# CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch main_cx.py
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
accelerator = Accelerator()
parser = argparse.ArgumentParser(description='Xiao is handsome')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch', type=int, default=64)
args = parser.parse_args()
config={
    "mode":"train",
    "batch":args.batch,
    "epoch":50,
    "lr":1e-4,
    "cuda":args.cuda
}
device = accelerator.device
writer= SummaryWriter(log_dir="./nets/tfboard/run1")
base_dir="/home/xiao/code/CS5242/dataset/"
# device=config["cuda"]
# device=torch.device(f"cuda:{device}")
def valid(net,data,epoch):
    with torch.no_grad():
        criertion=nn.CrossEntropyLoss()
        net.to(device)
        loader = DataLoader(data, batch_size=config["batch"], shuffle=True, num_workers=2)
        epoch_loss=0
        correct=0
        total=0
        for i,data in enumerate(loader):
            img,gt=data
            img=img.to(device)
            gt=gt.to(device)
            y=net(img)
            loss=criertion(y,gt)
            cls=torch.argmax(y,dim=1)
            correct+=(gt==cls).sum().item()
            total+=gt.size(0)
            epoch_loss+=loss.item()
        print(f"[epoch:{epoch}]","Val loss",epoch_loss/len(loader),"Val acc",correct/total)
        return epoch_loss/len(loader),correct/len(loader)
def train(net):
    best_acc=0
    # net.to(device)
    criertion=nn.CrossEntropyLoss()
    opt=optim.Adam(net.parameters(),lr=config["lr"],betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',patience=5,verbose=True)
    dataset = ImageDataset(base_dir + config["mode"],device=device)
    trainset, valset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset)-int(0.9 * len(dataset))])
    trainloader = DataLoader(trainset, batch_size=config["batch"], shuffle=True, num_workers=2)
    net, opt, trainloader, scheduler = accelerator.prepare(net, opt, trainloader, scheduler)
    for epoch in range(config["epoch"]):
        epoch_loss=0
        correct=0
        total=0
        for i,data in enumerate(trainloader):
            img,gt=data
            img=img.to(device)
            gt=gt.to(device)
            opt.zero_grad()
            # breakpoint()
            y=net(img)
            
           
            loss=criertion(y,gt)
           
            accelerator.backward(loss)
            
            opt.step()
            epoch_loss+=loss.item()
            if accelerator.is_local_main_process:
                all_y,all_gt=accelerator.gather_for_metrics((y,data[1]))
                all_gt=all_gt.to(device)
                cls=torch.argmax(all_y,dim=1)
                correct+=(all_gt==cls).sum().item()
                total+=all_gt.size(0)
            

        scheduler.step(epoch_loss/len(trainloader))
        writer.add_scalar("loss",epoch_loss/len(trainloader),epoch)
        writer.add_scalar("acc",correct/len(trainloader),epoch)
        if accelerator.is_local_main_process:
            print(f"[epoch:{epoch}]","loss",epoch_loss/len(trainloader),"acc",correct/total)
            if epoch%5==0:
                val_loss,val_acc=valid(net,valset,epoch)
                if best_acc<val_acc:
                    torch.save(net.state_dict(), '/home/xiao/code/CS5242/nets/model/model.pth')
                writer.add_scalar("Val loss",val_loss,epoch)
                writer.add_scalar("Val acc",val_acc,epoch)

def test(net):
    with torch.no_grad():
         
        net.to(device)
        dataset = ImageDataset(base_dir + 'test',device=device)
        testloader = DataLoader(dataset, batch_size=config["batch"], shuffle=True, num_workers=2)
        
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
    net = LeNet(2)
    train(net)
    test(net)
    writer.close()