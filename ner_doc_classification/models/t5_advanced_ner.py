from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import T5Config, T5EncoderModel


class BilateralFeatureEnhancement(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.left_context = nn.Linear(hidden_size, hidden_size)
        self.right_context = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_shift = torch.roll(x, shifts=1, dims=1)
        right_shift = torch.roll(x, shifts=-1, dims=1)

        left_features = self.left_context(left_shift)
        right_features = self.right_context(right_shift)

        enhanced = x + left_features + right_features
        return self.layer_norm(enhanced)


class AdvancedT5NER(nn.Module):
    def __init__(
        self,
        n_ner_tags: int,
        n_doc_labels: int,
        model_name: str = "t5-base",
        dropout_rates: Dict[str, float] = None,
    ):
        super().__init__()
        self.n_ner_tags = n_ner_tags
        self.n_doc_labels = n_doc_labels

        # Initialize dropout rates with defaults if not provided
        self.dropout_rates = {"attention": 0.1, "hidden": 0.2, "output": 0.1}
        if dropout_rates:
            self.dropout_rates.update(dropout_rates)

        # Main T5 encoder model (we only need the encoder for NER)
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        self.config = self.t5.config

        # Bilateral feature enhancement
        self.bilateral = BilateralFeatureEnhancement(self.config.hidden_size)

        # Layer normalization
        self.pre_norm = nn.LayerNorm(self.config.hidden_size)
        self.post_norm = nn.LayerNorm(self.config.hidden_size)

        # Intermediate transformation
        self.intermediate = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
        )

        # Multiple dropout layers
        self.attention_dropout = nn.Dropout(self.dropout_rates["attention"])
        self.hidden_dropout = nn.Dropout(self.dropout_rates["hidden"])
        self.output_dropout = nn.Dropout(self.dropout_rates["output"])

        # Output classifier
        self.ner_classifier = nn.Linear(self.config.hidden_size, n_ner_tags)
        self.doc_classifier = nn.Linear(self.config.hidden_size, n_doc_labels)

        # Store attention weights and hidden states
        self.attention_weights = None
        self.all_hidden_states = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask
            return_attention_weights (bool): Whether to return attention weights
            return_hidden_states (bool): Whether to return hidden states

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (ner_logits, doc_logits)
        """
        # T5 encoder forward pass with output options
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attention_weights,
            output_hidden_states=return_hidden_states,
            return_dict=True,
        )

        # Store attention and hidden states if requested
        if return_attention_weights:
            self.attention_weights = outputs.attentions
        if return_hidden_states:
            self.all_hidden_states = outputs.hidden_states

        # Get the encoder hidden states for NER task
        sequence_output = outputs.last_hidden_state

        # Apply pre-normalization
        sequence_output = self.pre_norm(sequence_output)
        sequence_output = self.attention_dropout(sequence_output)

        # Apply bilateral feature enhancement
        enhanced_output = self.bilateral(sequence_output)
        enhanced_output = self.hidden_dropout(enhanced_output)

        # Apply intermediate transformation
        intermediate_output = self.intermediate(enhanced_output)
        intermediate_output = self.hidden_dropout(intermediate_output)

        # Apply post-normalization
        final_output = self.post_norm(intermediate_output)
        final_output = self.output_dropout(final_output)

        # Use the full sequence output for NER classification
        ner_logits = self.ner_classifier(final_output)

        # Use the first token's representation for document classification
        doc_representation = final_output[:, 0, :]
        doc_logits = self.doc_classifier(doc_representation)

        return ner_logits, doc_logits

    def get_attention_weights(self) -> Optional[Tuple[torch.Tensor]]:
        """Return the last computed attention weights."""
        return self.attention_weights

    def freeze_bert_layers(self, num_layers: int = None):
        """Freeze specified number of T5 layers from bottom."""
        if num_layers is None:
            num_layers = len(self.t5.encoder.block)

        # Freeze embeddings
        for param in self.t5.shared.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        for i in range(min(num_layers, len(self.t5.encoder.block))):
            for param in self.t5.encoder.block[i].parameters():
                param.requires_grad = False

    def unfreeze_all_layers(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """Count model parameters."""

        def count_params(module: nn.Module) -> Tuple[int, int]:
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            return total_params, trainable_params

        # Count parameters for different components
        t5_total, t5_trainable = count_params(self.t5)
        bilateral_total, bilateral_trainable = count_params(self.bilateral)
        intermediate_total, intermediate_trainable = count_params(self.intermediate)
        classifier_total, classifier_trainable = count_params(self.classifier)

        results = {
            "t5_base": t5_trainable if trainable_only else t5_total,
            "bilateral_enhancement": (
                bilateral_trainable if trainable_only else bilateral_total
            ),
            "intermediate_layer": (
                intermediate_trainable if trainable_only else intermediate_total
            ),
            "classifier": classifier_trainable if trainable_only else classifier_total,
            "total": (
                (
                    t5_trainable
                    + bilateral_trainable
                    + intermediate_trainable
                    + classifier_trainable
                )
                if trainable_only
                else (
                    t5_total + bilateral_total + intermediate_total + classifier_total
                )
            ),
        }

        return results
