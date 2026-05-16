from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist(batch_size=64, image_size=28):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    return(
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=1, shuffle=False)
    )
