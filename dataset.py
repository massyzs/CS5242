import torch
from torch.utils.data import DataLoader,Dataset

from PIL import Image
from glob import glob
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir,device,config,train=False):
        self.image_dir = image_dir
        self.imgs=glob(self.image_dir+'*/*')
        # '/home/xiao/code/CS5242/dataset_aug/train/'
        # '/home/xiao/code/CS5242/dataset_aug/train'
        # /home/xiao/code/CS5242/dataset_copy/all_in_one
        # breakpoint()
        self.labels=[]
        # breakpoint()
        for img in self.imgs:
            if img.split('/')[-2]=="weap":
                self.labels.append(1)
            elif img.split('/')[-2]=="norm":
                self.labels.append(0)
        
        
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip the image horizontally with probability 0.5
            # transforms.RandomRotation(degrees=30),    # Randomly rotate the image up to 30 degrees
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust brightness, contrast, saturation and hue
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.44625, 0.4312, 0.40765], std=[0.274425, 0.273325, 0.2723])
        ])
      
        self.device=device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image=self.transform(Image.open(self.imgs[idx]).convert('RGB'))
        
        return image, torch.tensor(self.labels[idx])

<<<<<<< HEAD
=======
class DomainDataset(Dataset):
    def __init__(self, image_dir,device,config, domain,train=False):
        self.image_dir = image_dir
        self.imgs=glob(self.image_dir+'/*/*')
        self.labels=[]
        for img in self.imgs:
            if img.split('/')[-2]=="weap":
                self.labels.append(1)
            else:
                self.labels.append(0)
        self.domain = domain

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip the image horizontally with probability 0.5
            # transforms.RandomRotation(degrees=30),    # Randomly rotate the image up to 30 degrees
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust brightness, contrast, saturation and hue
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.44625, 0.4312, 0.40765], std=[0.274425, 0.273325, 0.2723])
        ])
        self.device=device
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image=self.transform(Image.open(self.imgs[index]).convert('RGB'))
        
        # Return the sample along with the domain label
        return image, torch.tensor(self.labels[index]), self.domain


>>>>>>> c45f099e52bb4bd57cfb850d4d1c5ebbd5b7c057
