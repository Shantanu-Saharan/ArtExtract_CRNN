import os

import matplotlib.pyplot as plt


def save_training_curves(
    history: dict,
    save_path: str,
) -> None:
    required = ("train_loss", "val_loss", "train_macro_f1", "val_macro_f1")
    for k in required:
        if k not in history:
            raise KeyError(f"history missing key '{k}'")

    epochs = range(1, len(history["train_loss"]) + 1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="val", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_macro_f1"], label="train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_macro_f1"], label="val", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Macro-F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
