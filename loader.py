from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

class ImageDataset(Dataset):

    def __init__(self, img_dir, imgs, transform=None):
        self.img_paths = [img_dir + img for img in imgs]
        self.labels = [int(img[:5]) for img in imgs]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform != None:
            img = self.transform(img)
        return img, label - 1
    
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), (0.8, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_train_dataloader(img_dir, batch_size, transforms=train_transforms):
    train_imgs = os.listdir(img_dir)
    dataset = ImageDataset(img_dir, train_imgs, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return dataloader

def get_val_dataloader(img_dir, batch_size, transforms=val_transforms):
    val_imgs = os.listdir(img_dir)
    dataset = ImageDataset(img_dir, val_imgs, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader

def get_test_dataloader(img_dir, batch_size, transforms=val_transforms):
    test_imgs = os.listdir(img_dir)
    dataset = ImageDataset(img_dir, test_imgs, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader