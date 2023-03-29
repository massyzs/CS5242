import torch
from torch.utils.data import DataLoader,Dataset

from PIL import Image
from glob import glob
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir,device):
        self.image_dir = image_dir
        self.imgs=glob(self.image_dir+'/*/*.png')
      #/home/xiaocao/code/Yao/dataset/weapon det/train/weap/lalala.png
        self.labels=[]
        for img in self.imgs:
            if img.split('/')[-2]=="weap":
                self.labels.append(1)
            else:
                self.labels.append(0)
        
        self.transform =transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.device=device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.open(self.imgs[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label)


# Set up any required image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


