from attacks.attack_fgsm import fgsm_attack
from attacks.attack_bim import bim_attack
from attacks.attack_pgd import pgd_attack

ATTACKS = {
    "fgsm": fgsm_attack,
    "bim": bim_attack,
    "pgd": pgd_attack
}