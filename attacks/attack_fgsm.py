import torch

def fgsm_attack(image, epilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)