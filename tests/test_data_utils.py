import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from mistralai.exceptions import MistralAPIException

from ner_doc_classification.data_processing.data_utils import (
    doc_labeler,
    load_or_create_doc_labels,
)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("PERSONAL_MISTRAL_API_KEY", "dummy-api-key")


@pytest.fixture
def sample_texts():
    return [["Hello", "John", "Doe"], ["Google", "Inc", "headquarters"]]


@pytest.fixture
def mock_mistral_response():
    mock_message = MagicMock()
    mock_message.content = '{"label": "PERSON", "confidence": 0.9}'  # PERSON category
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def test_doc_labeler(sample_texts, mock_mistral_response, mock_env):
    with patch(
        "mistralai.client.MistralClient.chat", return_value=mock_mistral_response
    ):
        labels = doc_labeler(os.environ["PERSONAL_MISTRAL_API_KEY"], sample_texts)

        assert isinstance(labels, list)
        assert len(labels) == len(sample_texts)
        assert all(isinstance(label, int) for label in labels)
        assert all(0 <= label <= 4 for label in labels)  # Check valid category range


def test_doc_labeler_invalid_response(sample_texts, mock_env):
    # Mock an invalid response from Mistral
    mock_message = MagicMock()
    mock_message.content = "invalid json"  # invalid json
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    invalid_response = MagicMock()
    invalid_response.choices = [mock_choice]

    with patch("mistralai.client.MistralClient.chat", return_value=invalid_response):
        labels = doc_labeler(os.environ["PERSONAL_MISTRAL_API_KEY"], sample_texts)

        assert isinstance(labels, list)
        assert len(labels) == len(sample_texts)
        assert all(label == 4 for label in labels)  # Check OTHER category


def test_load_or_create_doc_labels_new_file(
    sample_texts, mock_mistral_response, mock_env
):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        with patch(
            "mistralai.client.MistralClient.chat", return_value=mock_mistral_response
        ):
            labels = load_or_create_doc_labels(
                sample_texts, os.environ["PERSONAL_MISTRAL_API_KEY"], file_name=tmp.name
            )

            assert isinstance(labels, list)
            assert len(labels) == len(sample_texts)
            assert all(isinstance(label, int) for label in labels)

            # Verify the file was created with the correct content
            df = pd.read_csv(tmp.name)
            assert len(df) == len(sample_texts)
            assert all(isinstance(label, int) for label in df["label"])

    os.unlink(tmp.name)


def test_load_or_create_doc_labels_existing_file(sample_texts, mock_env):
    # Create a temporary file with existing labels
    existing_labels = [1, 2]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        # Write test data to CSV in the correct format
        pd.DataFrame({"label": existing_labels}).to_csv(tmp.name, index=False)

        # Should load existing labels without calling Mistral API
        with patch("mistralai.client.MistralClient.chat") as mock_chat:
            labels = load_or_create_doc_labels(
                sample_texts, os.environ["PERSONAL_MISTRAL_API_KEY"], file_name=tmp.name
            )

            # Verify that the Mistral API was not called
            mock_chat.assert_not_called()

            # Verify the loaded labels match the existing ones
            assert labels == existing_labels

    os.unlink(tmp.name)
