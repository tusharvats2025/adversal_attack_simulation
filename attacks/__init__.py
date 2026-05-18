from attacks.attack_fgsm import fgsm_attack
from attacks.attack_bim import bim_attack
from attacks.attack_pgd import pgd_attack
from attacks.autoattack_wrapper import run_autoattack

ATTACKS_REGISTRY = {
    "fgsm": fgsm_attack,
    "bim": bim_attack,
    "pgd": pgd_attack,
    "autoattack": run_autoattack,
}

__all__ = [
    "ATTACKS_REGISTRY",
    "fgsm_attack",
    "bim_attack", 
    "pgd_attack",
    "run_autoattack",
]
