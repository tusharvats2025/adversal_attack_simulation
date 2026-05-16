from attacks.autoattack_wrapper import run_autoattack

def robustbench_evaluate(model, device, test_loader, epsilons):
    results = {}

    for eps in epsilons:
        acc = run_autoattack(
            model=model,
            test_loader=test_loader,
            device=device,
            epsilon=eps
        )
        results[eps] = acc
    
    return results
