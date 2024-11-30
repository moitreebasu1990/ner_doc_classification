from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class CONLLDataset(Dataset):
    """
    A custom PyTorch Dataset for CONLL format data.

    Args:
        tokens (List[List[str]]): List of token sequences.
        ner_tags (List[List[int]]): List of corresponding NER tags.
        doc_labels (Optional[List[int]]): List of document labels. Defaults to None.
    """

    def __init__(
        self,
        tokens: List[List[str]],
        ner_tags: List[List[int]],
        doc_labels: Optional[List[int]] = None,
        max_length: int = 128,
    ):
        super().__init__()
        self.tokens = tokens
        self.ner_tags = ner_tags
        self.doc_labels = doc_labels
        # Maximum sequence length supported by BERT
        self.max_length = max_length
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict:
        """
        Fetches and prepares the data sample at the given index.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary that contains:
                - input_ids (torch.Tensor): Token ids to be fed to a model.
                - attention_mask (torch.Tensor): Attention masks.
                - ner_tags (torch.Tensor): NER tags.
                - doc_labels (torch.Tensor): Document labels, if available.
        """
        # Join the tokens into a string
        text = " ".join(self.tokens[index])
        # Get the corresponding NER tag
        ner_tag = self.ner_tags[index]
        # Encode the text using the BERT tokenizer
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Extract input_ids and attention_mask from the encoding
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Pad or crop the NER tag to match the max_length
        if len(ner_tag) < self.max_length:
            ner_tag = ner_tag + [-99] * (self.max_length - len(ner_tag))
        else:
            ner_tag = ner_tag[: self.max_length]

        # Prepare the data sample as a dictionary
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ner_tags": torch.tensor(ner_tag, dtype=torch.long),
        }

        # Add document labels to the data sample if they exist
        if self.doc_labels is not None:
            item["doc_labels"] = torch.tensor(self.doc_labels[index], dtype=torch.long)

        return item
