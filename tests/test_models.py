from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ner_doc_classification.models.bert_ner import BertNERModel
from ner_doc_classification.models.t5_advanced_ner import (
    AdvancedT5NER,
    BilateralFeatureEnhancement,
)


@pytest.fixture
def model_params():
    return {
        "n_ner_tags": 9,
        "n_doc_labels": 4,
        "batch_size": 2,
        "seq_length": 32,
        "hidden_dim": 768,
    }


@pytest.fixture
def input_tensors(model_params):
    return {
        "input_ids": torch.randint(
            0, 1000, (model_params["batch_size"], model_params["seq_length"])
        ),
        "attention_mask": torch.ones(
            (model_params["batch_size"], model_params["seq_length"])
        ),
    }


class TestBertNERModel:
    @patch("ner_doc_classification.models.bert_ner.BertModel")
    def test_bert_model_forward(self, mock_bert, model_params, input_tensors):
        # Mock BERT output
        mock_last_hidden_state = torch.randn(
            model_params["batch_size"],
            model_params["seq_length"],
            model_params["hidden_dim"],
        )
        mock_pooler_output = torch.randn(
            model_params["batch_size"], model_params["hidden_dim"]
        )

        class MockBertOutput:
            def __init__(self, last_hidden_state, pooler_output):
                self.last_hidden_state = last_hidden_state
                self.pooler_output = pooler_output

        mock_bert.from_pretrained.return_value = mock_bert
        mock_bert.return_value = MockBertOutput(
            mock_last_hidden_state, mock_pooler_output
        )

        # Initialize model
        model = BertNERModel(model_params["n_ner_tags"], model_params["n_doc_labels"])

        # Forward pass
        ner_logits, doc_logits = model(
            input_tensors["input_ids"], input_tensors["attention_mask"]
        )

        # Check output shapes
        assert ner_logits.shape == (
            model_params["batch_size"],
            model_params["seq_length"],
            model_params["n_ner_tags"],
        )
        assert doc_logits.shape == (
            model_params["batch_size"],
            model_params["n_doc_labels"],
        )

    def test_bert_model_components(self, model_params):
        model = BertNERModel(model_params["n_ner_tags"], model_params["n_doc_labels"])

        # Check model components
        assert isinstance(model.dropout, torch.nn.Dropout)
        assert isinstance(model.ner_classifier, torch.nn.Linear)
        assert isinstance(model.doc_classifier, torch.nn.Linear)

        # Check classifier dimensions
        assert model.ner_classifier.out_features == model_params["n_ner_tags"]
        assert model.doc_classifier.out_features == model_params["n_doc_labels"]


class TestAdvancedT5NER:
    def test_bilateral_enhancement(self, model_params):
        # Test the bilateral feature enhancement module
        bilateral = BilateralFeatureEnhancement(model_params["hidden_dim"])
        input_tensor = torch.randn(
            model_params["batch_size"],
            model_params["seq_length"],
            model_params["hidden_dim"],
        )
        output = bilateral(input_tensor)

        # Check output shape
        assert output.shape == input_tensor.shape

        # Check layer normalization
        assert isinstance(bilateral.layer_norm, torch.nn.LayerNorm)

    @patch("ner_doc_classification.models.t5_advanced_ner.T5EncoderModel")
    def test_t5_model_forward(self, mock_t5, model_params, input_tensors):
        # Mock T5 output
        mock_last_hidden_state = torch.randn(
            model_params["batch_size"],
            model_params["seq_length"],
            model_params["hidden_dim"],
        )

        class MockT5Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
                self.attentions = None
                self.hidden_states = None

        class MockT5Config:
            def __init__(self):
                self.hidden_size = 768  # Standard T5 hidden size

        mock_t5.from_pretrained.return_value = mock_t5
        mock_t5.config = MockT5Config()
        mock_t5.return_value = MockT5Output(mock_last_hidden_state)

        # Initialize model
        model = AdvancedT5NER(model_params["n_ner_tags"], model_params["n_doc_labels"])

        # Forward pass
        ner_logits, doc_logits = model(
            input_tensors["input_ids"],
            input_tensors["attention_mask"],
            return_attention_weights=True,
        )

        # Check output shapes
        assert ner_logits.shape == (
            model_params["batch_size"],
            model_params["seq_length"],
            model_params["n_ner_tags"],
        )
        assert doc_logits.shape == (
            model_params["batch_size"],
            model_params["n_doc_labels"],
        )

    @patch("ner_doc_classification.models.t5_advanced_ner.T5EncoderModel")
    def test_t5_model_components(self, mock_t5, model_params):
        model = AdvancedT5NER(model_params["n_ner_tags"], model_params["n_doc_labels"])

        # Check model components
        assert isinstance(model.bilateral, BilateralFeatureEnhancement)
        assert isinstance(model.pre_norm, torch.nn.LayerNorm)
        assert isinstance(model.post_norm, torch.nn.LayerNorm)
        assert isinstance(model.attention_dropout, torch.nn.Dropout)
        assert isinstance(model.hidden_dropout, torch.nn.Dropout)
        assert isinstance(model.output_dropout, torch.nn.Dropout)

        # Check classifier dimensions
        assert model.ner_classifier.out_features == model_params["n_ner_tags"]
        assert model.doc_classifier.out_features == model_params["n_doc_labels"]
