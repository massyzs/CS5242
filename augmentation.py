import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
from  torchvision import utils as vutils
import torch
transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip the image horizontally with probability 0.5
            transforms.RandomRotation(degrees=30),    # Randomly rotate the image up to 30 degrees
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust brightness, contrast, saturation and hue
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.44625, 0.4312, 0.40765], std=[0.274425, 0.273325, 0.2723])
        ])

norm_pth="/home/xiao/code/CS5242/dataset/train/norm/*"
norms=glob(norm_pth)
for idx in tqdm(range(len(norms))):
    image_raw = Image.open(norms[idx]).convert('RGB')
    totensor=transforms.ToTensor()
    vutils.save_image(totensor(image_raw), f"/home/xiao/code/CS5242/dataset/train/norm_aug/{idx}.jpg", normalize=False)
    image=transform(image_raw)
    
   
    vutils.save_image(image, f"/home/xiao/code/CS5242/dataset/train/norm_aug/{idx}_aug.jpg", normalize=False)


weap_pth="/home/xiao/code/CS5242/dataset/train/weap/*"
weaps=glob(weap_pth)
for idx in tqdm(range(len(weaps))):
    image_raw = Image.open(weaps[idx]).convert('RGB')
    totensor=transforms.ToTensor()
    vutils.save_image(totensor(image_raw), f"/home/xiao/code/CS5242/dataset/train/weap_aug/{idx}.jpg", normalize=False)
    image=transform(image_raw)
    vutils.save_image(image, f"/home/xiao/code/CS5242/dataset/train/weap_aug/{idx}_aug.jpg", normalize=False)
    # image_pil.save(f"/home/xiao/code/CS5242/dataset_1/train/weap_aug/{idx}.jpg")
