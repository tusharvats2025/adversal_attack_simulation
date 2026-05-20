import torch
from attacks.__init__ import ATTACKS_REGISTRY

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
    Now supports batch_size > 1
    """

    model.eval()
    correct = 0
    total = 0

    if attack_name not in ATTACKS_REGISTRY:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Available: {list(ATTACKS_REGISTRY.keys())}"
        )
    attack_fn = ATTACKS_REGISTRY[attack_name]

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Get initial predictions
        with torch.no_grad():
            output = model(data)
            init_pred = output.argmax(dim=1)
        
        # Track which samples are correctly classified
        correct_mask = (init_pred == target)
        
        # Skip if no correctly classified samples in this batch
        if not correct_mask.any():
            total += data.size(0)
            continue
        
        # Only attack correctly classified samples
        attack_data = data[correct_mask].clone()
        attack_target = target[correct_mask]
        
        attack_data.requires_grad_(True)
        
        # Forward pass on attacked samples
        output = model(attack_data)
        loss = torch.nn.CrossEntropyLoss()(output, attack_target)
        
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        perturbed_data = attack_fn(
            model=model,
            images=attack_data,
            labels=attack_target,
            **attack_params
        )
        
        # Re-classify perturbed samples
        with torch.no_grad():
            adv_output = model(perturbed_data)
            final_pred = adv_output.argmax(dim=1)
        
        # Count correct predictions after attack
        correct += (final_pred == attack_target).sum().item()
        total += attack_target.size(0)
    
    adv_acc = correct / total if total > 0 else 0.0
    return adv_acc