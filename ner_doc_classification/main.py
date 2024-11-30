import os

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ner_doc_classification.data_processing.data_utils import load_or_create_doc_labels
from ner_doc_classification.data_processing.dataset import CONLLDataset
from ner_doc_classification.models.bert_ner import BertNERModel
from ner_doc_classification.training.train import train_ner_model
from ner_doc_classification.visualization.visualization import plot_ner_learning_curves

# Configuration
PERSONAL_MISTRAL_API_KEY = os.environ.get("PERSONAL_PERSONAL_MISTRAL_API_KEY")
if not PERSONAL_MISTRAL_API_KEY:
    raise ValueError("Please set the PERSONAL_MISTRAL_API_KEY environment variable")

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 9  # PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, OTHER
DATASET_SIZES = [10, 30, 100, 300, 1000]
RANDOM_SEED = 19


def prepare_data(size: int, model_name: str, doc_label_file="data/doc_labels.csv"):
    """Prepare the dataset for training.

    Args:
        size (int): Number of samples to use
        model_name (str): Name of the model to use (for tokenization)

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Load CONLL2003 dataset
    dataset = load_dataset("conll2003")
    random_indices = np.random.choice(
        len(dataset["train"]), size=size, replace=False
    ).tolist()

    dataset["train"] = dataset["train"].select(random_indices)
    tokens = dataset["train"]["tokens"]
    ner_tags = dataset["train"]["ner_tags"]
    doc_labels = load_or_create_doc_labels(
        tokens=tokens, api_key=PERSONAL_MISTRAL_API_KEY, file_name=doc_label_file
    )

    conll_dataset = CONLLDataset(tokens, ner_tags, doc_labels, MODEL_NAME)

    # Split data
    size = len(conll_dataset)
    train_size = int(0.8 * size)
    val_size = size - train_size
    train_dataset, val_dataset = random_split(conll_dataset, [train_size, val_size])

    return train_dataset, val_dataset


def main():
    # Train T5 model with different dataset sizes
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Set device to 'mps' if available, else 'cuda' if available, else 'cpu'
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    for size in DATASET_SIZES:
        # Prepare data
        train_dataset, val_dataset = prepare_data(
            size, BERT_MODEL_NAME, "../data/doc_labels.csv"
        )

        # Initialize model
        model = BertNERModel(
            n_ner_tags=NUM_NER_LABELS,
            n_doc_labels=NUM_DOC_LABELS,
            model_name=BERT_MODEL_NAME,
        )

        # Train model
        train_losses, val_losses, report = train_ner_doc_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            num_epochs=3,
        )

        print(f"\nNER Classification Report for size {DATASET_SIZE}:")
        print(report[0])

        print(f"\nDOC Classification Report for size {DATASET_SIZE}:")
        print(report[1])

        plot_learning_curves(
            train_losses,
            val_losses,
            save_path=f"{save_dir}/bert_learning_curves_{size}.png",
        )


if __name__ == "__main__":
    main()
