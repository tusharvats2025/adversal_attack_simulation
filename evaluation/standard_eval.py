import torch
from attacks.init import ATTACKS_REGISTORY

def standard_evaluate(
        model,
        device,
        test_loader,
        attack_name: str,
        attack_params: dict,
):
    """
    Standard adverserial evaluation for FGSM / BIM/ PGD
    Returns the accuracy under attak.
    """

    model.eval()
    correct = 0
    total = 0

    if attack_name not in ATTACKS_REGISTORY:
        raise ValueError(
            f"Unkown attack '{attack_name}'."
            f"Available : {list(ATTACKS_REGISTORY.keys())}"
        )
    attack_fn = ATTACKS_REGISTORY[attack_name]

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Forward pass
        output = model(data)
        init_pred = output.argmax(dim=1)

        # Only attack correctly classified samples
        if init_pred.item() != target.item():
            total += 1
            continue

        loss = torch.nn.CrossEntropyLoss()(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data

        # Generate adversarial example 
        perturbed_data = attack_fn(
            model = model,
            image = data,
            labels = target,
            data_grad = data_grad,
            **attack_params
        )

        # Re-classify 
        with torch.no_grad():
            adv_output = model(perturbed_data)
            final_pred = adv_output.argmax(dim=1)

        if final_pred.item() == target.item():
            correct += 1
        
        total += 1
    
    adv_acc = correct / total
    return adv_acc


