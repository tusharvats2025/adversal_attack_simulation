from evaluation.standard_eval import standard_evaluate
from evaluation.robustbench_eval import robustbench_evaluate

def evaluate_factory(
        model,
        device,
        test_loader,
        attack_name: str,
        epsilons,
        attack_params: dict = None,
):
    """
    Unified evaluation factory for all attacks.
    """

    results = {}

    # =========================================================
    # AutoAttack (RobustBench-style)
    # =========================================================
    if attack_name == "autoattack":
        return robustbench_evaluate(
            model=model,
            device=device,
            test_loader=test_loader,
            epsilons=epsilons
        )
    
    # ===========================================================
    # Standard Attacks (FGSM / BIM / PGD)
    # ===========================================================
    if attack_params is None:
        attack_params = {}
    
    for eps in epsilons:
        params = attack_params.copy()
        params["epsilon"] = eps

        acc = standard_evaluate(
            model=model,
            device=device,
            test_loader=test_loader,
            attack_name=attack_name,
            attack_params=params
        )

        results[eps] = acc

    return results
