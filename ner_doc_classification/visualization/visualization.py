from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
):
    """
    Plots training and validation learning curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Optional path to save the plot
    """
    # Convert tensors to CPU if they are on another device
    train_losses = [
        loss.cpu().item() if hasattr(loss, "cpu") else loss for loss in train_losses
    ]
    val_losses = [
        loss.cpu().item() if hasattr(loss, "cpu") else loss for loss in val_losses
    ]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")

    plt.title("Model Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
