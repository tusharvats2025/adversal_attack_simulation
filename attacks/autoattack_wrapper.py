import torch
from autoattack import AutoAttack

def run_autoattack(model, test_loader, device, epsilon=8/255):
    model.eval()
    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=epsilon,
        version='standard',
        device=device
    )

    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = adversary.run_standard_evaluation(data, target, bs=1)

        with torch.no_grad():
            output = model(adv_data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
    robust_acc = correct / total
    return robust_acc
