import torch

def fgsm_attack(model,image, labels, data_grad, epsilon, **kwargs):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)