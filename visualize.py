import matplotlib.pyplot as plt
import os

def save_adversarial_examples(all_results, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)

    for epsilon, _, examples in all_results:
        if not examples:
            continue
        plt.figure(figsize=(10, 4))
        for i in range(min(len(examples), 5)):
            orig, adv = examples[i]
            plt.subplot(2, 5, i + 1)
            plt.imshow(orig, cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, 5, i + 6)
            plt.imshow(adv, cmap="gray")
            plt.title("Adversarial")
            plt.axis("off")

        plt.suptitle(f"Epsilon = {epsilon}")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"fgsm_epsilon_{epsilon}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[+] Saved adversarial examples for ε={epsilon} → {save_path}")
