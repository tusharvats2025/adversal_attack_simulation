import torch
from model import SimpleCNN
from data_loader import get_mnist_loaders
from attack_fgsm import fgsm_attack

def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    model.eval()

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = torch.nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
        else:
            adv_examples.append((data.squeeze().detach().cpu(), perturbed_data.squeeze().detach().cpu()))
    
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {final_acc * 100:.2f}%")

    return final_acc, adv_examples 

def evaluate_multiple_epsilons(model, device, test_loader, epsilons):
    all_results = {}
    for eps in epsilons:
        acc, advs = test(model, device, test_loader, eps)
        all_results.append((eps, acc, advs))
    return all_results
