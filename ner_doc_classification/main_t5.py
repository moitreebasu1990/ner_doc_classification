import os
from typing import Dict, Optional

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ner_doc_classification.data_processing.data_utils import load_or_create_doc_labels
from ner_doc_classification.data_processing.dataset import CONLLDataset
from ner_doc_classification.main import prepare_data
from ner_doc_classification.models.t5_advanced_ner import AdvancedT5NER
from ner_doc_classification.training.train_t5 import train_advanced_t5
from ner_doc_classification.visualization.t5_visualization import T5Visualizer

# Configuration
PERSONAL_MISTRAL_API_KEY = os.environ.get("PERSONAL_PERSONAL_MISTRAL_API_KEY")
if not PERSONAL_MISTRAL_API_KEY:
    raise ValueError("Please set the PERSONAL_MISTRAL_API_KEY environment variable")

MODEL_NAME = "t5-base"
NUM_LABELS = 8  # PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, OTHER
DATASET_SIZES = [10, 30, 100, 300, 1000]
RANDOM_SEED = 42

# Model configuration
MODEL_CONFIG = {"dropout_rates": {"attention": 0.1, "hidden": 0.2, "output": 0.1}}


def train_and_visualize(
    size: int, visualizer: T5Visualizer, save_dir: str = "outputs"
) -> Dict[str, float]:
    """Train the model and create visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    train_dataset, val_dataset = prepare_data(
        size, MODEL_NAME, "../data/doc_labels.csv"
    )

    # Initialize model
    model = AdvancedT5NER(
        num_labels=NUM_LABELS,
        model_name=MODEL_NAME,
        dropout_rates=MODEL_CONFIG["dropout_rates"],
    )

    # Print model parameters
    print("\nModel Parameters:")
    param_counts = model.count_parameters()
    for component, count in param_counts.items():
        print(f"{component}: {count:,}")

    # Train model with attention collection
    train_losses, val_losses, report, attention_maps = train_advanced_t5(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collect_attention_maps=True,
    )

    # Create visualizations
    print("\nGenerating visualizations...")

    # Learning curves
    visualizer.plot_learning_curves(
        train_losses, val_losses, save_path=f"{save_dir}/learning_curves_{size}.png"
    )

    # Attention weights for last epoch
    if attention_maps:
        last_epoch = max(
            int(k.split("_")[1]) for k in attention_maps.keys() if k.startswith("epoch")
        )
        attention_data = attention_maps[f"epoch_{last_epoch}_batch_0"]

        # Single head attention
        visualizer.plot_attention_weights(
            attention_data, save_path=f"{save_dir}/attention_weights_{size}.png"
        )

        # Multi-head attention
        visualizer.plot_multi_head_attention(
            attention_data, save_path=f"{save_dir}/multi_head_attention_{size}.png"
        )

        # Attention evolution
        visualizer.plot_attention_evolution(
            attention_maps, save_path=f"{save_dir}/attention_evolution_{size}.png"
        )

    print("\nClassification Report:")
    print(report)

    # Return final losses
    return {"train_loss": train_losses[-1], "val_loss": val_losses[-1]}


def main():
    """Main function to run the training and visualization pipeline."""
    torch.manual_seed(RANDOM_SEED)

    # Initialize visualizer
    visualizer = T5Visualizer(model_name=MODEL_NAME)

    # Create output directory
    output_dir = "model_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Train and visualize for each dataset size
    results = {}
    for size in DATASET_SIZES:
        print(f"\nTraining with dataset size: {size}")
        size_dir = f"{output_dir}/size_{size}"
        results[size] = train_and_visualize(size, visualizer, save_dir=size_dir)

    # Print final results
    print("\nFinal Results:")
    for size, metrics in results.items():
        print(f"\nDataset Size: {size}")
        print(f"Training Loss: {metrics['train_loss']:.4f}")
        print(f"Validation Loss: {metrics['val_loss']:.4f}")


if __name__ == "__main__":
    main()
