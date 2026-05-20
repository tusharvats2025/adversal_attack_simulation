from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10(batch_size=128, image_size=32):

    if image_size != 32:
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
              ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.ToTensor()

    train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    test = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=1, shuffle=False, num_workers=2)
    )