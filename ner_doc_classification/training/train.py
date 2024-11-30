from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def train_ner_doc_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 5,
    device: str = "cpu",
) -> Tuple[List[float], List[float], str]:
    """
    Trains a NER model with early stopping.

    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait before early stopping
        device: Device to train on

    Returns:
        Tuple of training and validation losses, and classification report
    """
    device = torch.device(device)
    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Define the loss functions
    doc_criterion = torch.nn.CrossEntropyLoss()
    ner_criterion = torch.nn.CrossEntropyLoss(ignore_index=-99)

    # Lists to store losses
    train_losses = []
    val_losses = []

    # Early stopping variables
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Lists to store predictions and labels for classification report
    all_ner_predictions = []
    all_ner_labels = []
    all_doc_predictions = []
    all_doc_labels = []

    # Training loop
    print(f"\nTraining on {device}")
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}/{num_epochs}",
            position=1,
            leave=False,
        )

        for batch in train_pbar:
            # Move the batch data to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_tags = batch["ner_tags"].to(device)
            doc_labels = batch["doc_labels"].to(device)

            optimizer.zero_grad()

            # Forward pass: compute predicted NER and document classification outputs by passing inputs to the model
            ner_model_outputs, doc_model_outputs = model(input_ids, attention_mask)

            # Calculate the NER loss and document classification loss
            ner_train_loss = ner_criterion(
                ner_model_outputs.view(-1, ner_model_outputs.shape[-1]),
                ner_tags.view(-1),
            )
            doc_train_loss = doc_criterion(doc_model_outputs, doc_labels.squeeze(-1))

            # Total loss is the sum of NER loss and document classification loss
            train_loss = ner_train_loss + doc_train_loss

            # Backward pass: compute gradient of the loss with respect to model parameters
            train_loss.backward()

            optimizer.step()

            epoch_train_loss += train_loss.item()
            train_pbar.set_postfix({"loss": f"{train_loss.item():.4f}"})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_pbar = tqdm(
            val_loader,
            desc=f"Validation Epoch {epoch+1}/{num_epochs}",
            position=1,
            leave=False,
        )

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["doc_labels"].to(device)
                ner_tags = batch["ner_tags"].to(device)

                # Forward pass: compute predicted NER and document classification outputs by passing inputs to the model
                ner_model_outputs, doc_model_outputs = model(input_ids, attention_mask)

                # Convert NER output probabilities to predicted class
                ner_prediction = torch.argmax(ner_model_outputs, dim=-1)

                # Mask out the special tokens and ignore the padding tokens for NER evaluation
                valid_mask = (attention_mask == 1) & (ner_tags != -99)

                # Convert document classification output probabilities to predicted class
                doc_prediction = torch.argmax(doc_model_outputs, dim=-1)

                # Calculate the NER loss and document classification loss
                ner_val_loss = ner_criterion(
                    ner_model_outputs.view(-1, ner_model_outputs.shape[-1]),
                    ner_tags.view(-1),
                )
                doc_val_loss = doc_criterion(doc_model_outputs, labels.squeeze(-1))

                # Total loss is the sum of NER loss and document classification loss
                epoch_val_loss += ner_val_loss + doc_val_loss
                val_pbar.set_postfix({"loss": f"{ner_val_loss + doc_val_loss:.4f}"})

                # Get predictions for classification report
                all_ner_predictions.extend(ner_prediction[valid_mask].cpu().numpy())
                all_ner_labels.extend(ner_tags[valid_mask].cpu().numpy())
                all_doc_predictions.extend(doc_prediction.cpu().numpy())
                all_doc_labels.extend(labels.squeeze(-1).cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # Generate classification report
    # Generate classification reports for NER and document classification
    ner_classification_report = classification_report(
        all_ner_labels, all_ner_predictions
    )
    doc_classification_report = classification_report(
        all_doc_labels, all_doc_predictions
    )

    return (
        train_losses,
        val_losses,
        (ner_classification_report, doc_classification_report),
    )
