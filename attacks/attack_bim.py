import torch

def bim_attack(model, images, labels, epsilon, alpha=0.01, iters=20, **kwargs):
    original = images.clone().detach()
    perturbed = original.clone().detach()

    for _ in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data

        perturbed = perturbed + alpha * grad.sign()
        eta = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + eta, 0, 1).detach()
    
    return perturbed