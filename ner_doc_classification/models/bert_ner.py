import torch
from torch import nn
from transformers import BertModel


class BertNERModel(nn.Module):
    """
    BERT-based model for document classification.

    Args:
        n_ner_tags (int): Number of NER tags.
        model_name (str): Name of the pre-trained BERT model.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        n_ner_tags: int,
        n_doc_labels: int,
        model_name: str = "bert-base-uncased",
        p_dropout: float = 0.1,
        dim_hidden: int = 768,
    ):
        """Initializes the BertNERDocClassificationModel with the given parameters.

        Args:
            n_ner_tags (int): The number of NER tags.
            n_doc_labels (int): The number of document labels.
            p_dropout (float): The dropout probability.
            dim_hidden (int): The dimension of the hidden layer.
        """
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p_dropout)
        self.ner_classifier = nn.Linear(dim_hidden, n_ner_tags)
        self.doc_classifier = nn.Linear(dim_hidden, n_doc_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Document classification logits.
        """
        bert_output = self.model(input_ids, attention_mask)

        # Get the output of the last hidden state of the BERT model for NER task
        ner_sequece_output = bert_output.last_hidden_state
        ner_sequece_output = self.dropout(ner_sequece_output)

        # Get the output of the pooler layer of the BERT model for document classification task
        doc_sequence_output = bert_output.pooler_output
        doc_sequence_output = self.dropout(doc_sequence_output)

        # Classify NER and document labels
        ner_logits = self.ner_classifier(ner_sequece_output)
        doc_logits = self.doc_classifier(doc_sequence_output)

        return ner_logits, doc_logits
