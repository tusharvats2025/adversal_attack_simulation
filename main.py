import torch
import config
import argparse

from datasets.dataset_factory import get_dataset
from datasets.dataset_meta import DATASET_META
from models.model_factory import get_model

from train import train_single_model
from evaluation.eval_factory import evaluate_factory
from utils.metrics_io  import save_metrics
from utils.archive import archive_results

meta = DATASET_META[config.DATASET]

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Adversarial Robustness Experiment Framework."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_META.keys()),
        help="Dataset to use {one at a time}",
    )

    parser.add_argument(
        "--models",
        nargs = "+",
        default = ["all"],
        help = "Model names or 'all'",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices = ["train", "eval", "visualise", "full"],
        default = "full",
        help = "Operation Mode",
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["all"],
        help="Attacks to evaluate or 'all'",
    )

    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive existing results before running",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this experiment run",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # ---------------------- Archive old results ----------------------------------------
    if args.archive:
        archive_results(run_name=args.run_name)

    # ---------------- Resolve dataset --------------------------------------------------
    config.DATASET = args.dataset
    meta = DATASET_META[config.DATASET]

    # ---------------- Resolve models ---------------------------------------------------
    if args.models == ["all"]:
        models = config.MODELS
    else:
        models = args.models

    # ---------------- Resolve attacks --------------------------------------------------
    if args.attacks == ["all"]:
        attacks = list(config.EPSILONS.keys())
    else:
        attacks = args.attacks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataset(
        config.DATASET, 
        batch_size=config.BATCH_SIZE,
        image_size=meta["image_size"]    
    )

    print("\n===== EXPERIMENT CONFIG =====")
    print(f"Dataset : {config.DATASET}")
    print(f"Models  : {models}")
    print(f"Attacks : {attacks}")
    print(f"Mode    : {args.mode}")
    print("============================\n")

    # ---------------- Main loop --------------------------------
    for model_name in models:
        ckpt_path = f"checkpoints/{model_name}_{config.DATASET}.pth"

        # ---- Train ---------------------------------------------
        if args.mode in ["train", "full"]:
            train_single_model(model_name, device)

        # ---- Load model -----------------------------------------
        model = get_model(
            model_name,
            num_classes=meta["num_classes"],
            in_channels=meta["in_channels"],
            image_size=meta["image_size"],
        ).to(device)

        model.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        model.eval()

        # ---- Evaluate ----
        if args.mode in ["eval", "full"]:
            for attack in attacks:
                results = evaluate_factory(
                    model=model,
                    device=device,
                    test_loader=test_loader,
                    attack_name=attack,
                    epsilons=config.EPSILONS[attack],
                    attack_params=config.ATTACK_CONFIGS.get(attack, {}),
                )

                save_metrics(
                    dataset=config.DATASET,
                    model=model_name,
                    attack=attack,
                    attack_params=config.ATTACK_CONFIGS.get(attack, {}),
                    results=results,
                )

    print("\n Experiment complete successfully.")

if __name__ == "__main__":
    main()



