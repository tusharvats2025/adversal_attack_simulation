DATASET = "tiny_imagenet" # mnist / cifar10 / fashion_mnist
MODELS = ["resnet18", "SimpleCNN", "wideresnet", "vit"]

BATCH_SIZE = 128
EPOCHS = 90

ATTACK_CONFIGS = {
    "fgsm": { "epsilon": 0.1 },
    "bim": { "epsilon": 0.1, "alpha": 0.01, "iters": 20 },
    "pgd": { "epsilon": 0.1, "alpha": 0.01, "iters": 20 }
}

EPSILONS = {
    "fgsm": [0.05, 0.1, 0.15],
    "pgd": [0.1],
    "autoattack": [8/255]
}