from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_fashion_mnist(batch_size=64):
    transform = transforms.ToTensor()

    train = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    return(
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=1, shuffle=False)
    )
