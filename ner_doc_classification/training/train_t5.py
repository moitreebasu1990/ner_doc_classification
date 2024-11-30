from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def train_advanced_t5(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 5,
    device: str = "cpu",
    freeze_encoder_epochs: int = 2,
    collect_attention_maps: bool = False,
) -> Tuple[List[float], List[float], Tuple[str, str], Optional[Dict]]:
    """
    Trains the Advanced T5 model for both NER and document classification tasks.

    Args:
        model: The advanced T5 model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait before early stopping
        device: Device to train on
        freeze_encoder_epochs: Number of epochs to keep encoder frozen
        collect_attention_maps: Whether to collect attention maps during validation

    Returns:
        Tuple of (train_losses, val_losses, (ner_report, doc_report), attention_maps)
    """
    device = torch.device(device)
    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize optimizer and loss functions
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    doc_criterion = nn.CrossEntropyLoss()
    ner_criterion = nn.CrossEntropyLoss(ignore_index=-99)

    # Lists to store losses and metrics
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    attention_maps = {} if collect_attention_maps else None

    # Lists to store predictions and labels for classification reports
    all_ner_predictions = []
    all_ner_labels = []
    all_doc_predictions = []
    all_doc_labels = []

    # Training loop
    print(f"\nTraining on {device}")
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        # Unfreeze encoder after specified epochs
        if epoch == freeze_encoder_epochs:
            for param in model.t5.parameters():
                param.requires_grad = True

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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_tags = batch["ner_tags"].to(device)
            doc_labels = batch["doc_labels"].to(device)

            optimizer.zero_grad()

            # Forward pass for both NER and document classification
            ner_logits, doc_logits = model(
                input_ids,
                attention_mask,
                return_attention_weights=collect_attention_maps,
            )

            # Calculate losses
            ner_loss = ner_criterion(
                ner_logits.view(-1, ner_logits.shape[-1]), ner_tags.view(-1)
            )
            doc_loss = doc_criterion(doc_logits, doc_labels.squeeze(-1))

            # Total loss
            train_loss = ner_loss + doc_loss
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
                ner_tags = batch["ner_tags"].to(device)
                doc_labels = batch["doc_labels"].to(device)

                # Forward pass
                ner_logits, doc_logits = model(
                    input_ids,
                    attention_mask,
                    return_attention_weights=collect_attention_maps,
                )

                # Store attention maps if requested
                if collect_attention_maps and model.attention_weights is not None:
                    batch_id = len(attention_maps)
                    attention_maps[batch_id] = {
                        "weights": [w.cpu() for w in model.attention_weights],
                        "input_ids": input_ids.cpu(),
                    }

                # Get predictions
                ner_preds = torch.argmax(ner_logits, dim=-1)
                doc_preds = torch.argmax(doc_logits, dim=-1)

                # Mask out padding tokens for NER evaluation
                valid_mask = (attention_mask == 1) & (ner_tags != -99)

                # Calculate losses
                ner_loss = ner_criterion(
                    ner_logits.view(-1, ner_logits.shape[-1]), ner_tags.view(-1)
                )
                doc_loss = doc_criterion(doc_logits, doc_labels.squeeze(-1))
                val_loss = ner_loss + doc_loss

                epoch_val_loss += val_loss.item()
                val_pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

                # Collect predictions and labels for classification reports
                all_ner_predictions.extend(ner_preds[valid_mask].cpu().numpy())
                all_ner_labels.extend(ner_tags[valid_mask].cpu().numpy())
                all_doc_predictions.extend(doc_preds.cpu().numpy())
                all_doc_labels.extend(doc_labels.squeeze(-1).cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # Generate classification reports
    ner_report = classification_report(all_ner_labels, all_ner_predictions)
    doc_report = classification_report(all_doc_labels, all_doc_predictions)

    return train_losses, val_losses, (ner_report, doc_report), attention_maps
