import os
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms, eval_tfms

def get_loaders(train_dir: str, val_dir: str, test_dir: str, img_size: int,
                batch_size: int, num_workers: int):
    train_tfms, val_tfms, test_tfms = get_transforms(img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_tfms)
    test_ds  = datasets.ImageFolder(test_dir, transform=test_tfms) if os.path.isdir(test_dir) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds else None

    return train_loader, val_loader, test_loader, train_ds.classes
