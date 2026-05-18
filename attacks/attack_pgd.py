import torch 

def pgd_attack(model, images, labels, epsilon, alpha=0.01, iters=20, **kwargs):
    original = images.clone().detach()

    # Random Start
    perturbed = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1)

    for _ in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data

        perturbed = perturbed + alpha * grad.sign()
        eta  = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + eta, 0, 1).detach()
    
    return perturbed
