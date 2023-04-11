import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
from  torchvision import utils as vutils
transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip the image horizontally with probability 0.5
                transforms.RandomRotation(degrees=30),    # Randomly rotate the image up to 30 degrees
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # Randomly adjust brightness, contrast, saturation and hue
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

norm_pth="/home/xiao/code/CS5242/dataset_1/train/norm/*"
norms=glob(norm_pth)
for idx in tqdm(range(len(norms))):
    image_raw = Image.open(norms[idx]).convert('RGB')
    image=transform(image_raw)
    image=image
   
    vutils.save_image(image, f"/home/xiao/code/CS5242/dataset_1/train/norm_aug/{idx}.jpg", normalize=False)


weap_pth="/home/xiao/code/CS5242/dataset_1/train/weap/*"
weaps=glob(weap_pth)
for idx in tqdm(range(len(weaps))):
    image = Image.open(weaps[idx]).convert('RGB')
    image=transform(image)
    vutils.save_image(image, f"/home/xiao/code/CS5242/dataset_1/train/weap_aug/{idx}.jpg", normalize=False)
    # image_pil.save(f"/home/xiao/code/CS5242/dataset_1/train/weap_aug/{idx}.jpg")
