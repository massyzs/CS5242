import torch
from torch.utils.data import DataLoader,Dataset

from PIL import Image
from glob import glob
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir,device,config,train=False):
        self.image_dir = image_dir
        self.imgs=glob(self.image_dir+'/*/*')
        # /home/xiao/code/CS5242/dataset/train
        # /home/xiao/code/CS5242/dataset/train/norm_aug
        # breakpoint()
        # breakpoint()
      #/home/xiaocao/code/Yao/dataset/weapon det/train/weap/lalala.png
        self.labels=[]
        for img in self.imgs:
            if img.split('/')[-2]=="weap_aug":
                self.labels.append(1)
            else:
                self.labels.append(0)
        
        
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip the image horizontally with probability 0.5
            # transforms.RandomRotation(degrees=30),    # Randomly rotate the image up to 30 degrees
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust brightness, contrast, saturation and hue
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.44625, 0.4312, 0.40765], std=[0.274425, 0.273325, 0.2723])
        ])
      
    
          
        # self.images = []

        self.device=device
        # for idx in range(len(self.imgs)):
        #     self.images.append(self.transform(Image.open(self.imgs[idx]).convert('RGB')))
        # for idx in range(len(self.imgs)):
        #     self.images.append(self.transform2(Image.open(self.imgs[idx]).convert('RGB')))
        # self.labels= self.labels+ self.labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image=self.transform(Image.open(self.imgs[idx]).convert('RGB'))
        
        return image, torch.tensor(self.labels[idx])




