import torch

def fgsm_attack(model,images, labels, epsilon, **kwargs):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model    : Target model (unused in FGSM but kept for API consistency)
        images   : Input images tensor
        labels   : True labels
        epsilon  : Perturbation magnitude.
        **kwargs : Additional arguments (ignored).
    Returns:
       Adversarial images tensor
    """

    images.requires_grad_(True)

    outputs = model(images)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.background()
    
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = images + epsilon * sign_data_grad
    
    return torch.clamp(perturbed_image, 0, 1)