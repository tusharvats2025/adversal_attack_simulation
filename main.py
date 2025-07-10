import torch
import os
from model import SimpleCNN
from data_loader import get_mnist_loaders
from evaluate import evaluate_multiple_epsilons
from visualize import save_adversarial_examples
import train

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("model.pth"):
        print("[*] Training model for the first time...")
        train.train_model()
    else:
        print("[*] Model already exists. Skipping training.")

    print("[*] Loading model...")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))

    _, test_loader = get_mnist_loaders()
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    print("[*] Running FGSM attack on multiple epsilons...")
    results = evaluate_multiple_epsilons(model, device, test_loader, epsilons)

    save_adversarial_examples(results)

if __name__ == "__main__":
    main()