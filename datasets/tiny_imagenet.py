import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_tiny_imagenet(batch_size=128, image_size=64):
    data_dir = "./data/tiny-imagenet-200"

    tranform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=tranform_train
    )

    val_set = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transform_test
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_set, batch_size=1, shuffle=False),
    )