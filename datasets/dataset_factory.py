from datasets.mnist import get_mnist
from datasets.fashion_mnist import get_fashion_mnist
from datasets.cifar10 import get_cifar10
from datasets.tiny_imagenet import get_tiny_imagenet

DATASET_REGISTRY = {
    "mnist": get_mnist,
    "fashion_mnist": get_fashion_mnist,
    "cifar10": get_cifar10,
    "tiny_imagenet": get_tiny_imagenet
}

def get_dataset(name, batch_size=64, image_size=None):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](
        batch_size=batch_size,
        image_size=image_size
    )

