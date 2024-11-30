import pytest
import torch

from ner_doc_classification.data_processing.dataset import CONLLDataset


@pytest.fixture
def sample_data():
    tokens = [["Hello", "John", "Doe"], ["This", "is", "a", "test"]]
    # Example NER tags: 0 for O (Other), 1 for B-PER (Beginning of Person), 2 for I-PER (Inside of Person)
    ner_tags = [
        [0, 1, 2],  # Hello [O], John [B-PER], Doe [I-PER]
        [0, 0, 0, 0],  # This [O], is [O], a [O], test [O]
    ]
    doc_labels = [1, 4]  # 1 for PERSON document, 4 for OTHER document
    return tokens, ner_tags, doc_labels


def test_dataset_initialization(sample_data):
    tokens, ner_tags, doc_labels = sample_data
    dataset = CONLLDataset(tokens=tokens, ner_tags=ner_tags, doc_labels=doc_labels)
    assert len(dataset) == 2
    assert dataset.tokens == tokens
    assert dataset.ner_tags == ner_tags
    assert dataset.doc_labels == doc_labels
    assert dataset.max_length == 128  # default value


def test_dataset_getitem(sample_data):
    tokens, ner_tags, doc_labels = sample_data
    dataset = CONLLDataset(tokens=tokens, ner_tags=ner_tags, doc_labels=doc_labels)
    item = dataset[0]

    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "ner_tags" in item
    assert "doc_labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["ner_tags"], torch.Tensor)
    assert isinstance(item["doc_labels"], torch.Tensor)
    assert item["doc_labels"].item() == 1  # First document should be PERSON (1)


def test_dataset_max_length():
    tokens = [["Hello", "world", "!"]]
    ner_tags = [[0, 0, 0]]  # All O tags
    doc_labels = [4]  # OTHER document
    max_length = 64

    dataset = CONLLDataset(
        tokens=tokens, ner_tags=ner_tags, doc_labels=doc_labels, max_length=max_length
    )
    item = dataset[0]

    assert item["input_ids"].size(0) == dataset.max_length
    assert item["attention_mask"].size(0) == dataset.max_length
    assert item["ner_tags"].size(0) == dataset.max_length
