import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5SearchIndex(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model. If labels are provided, the model will
        return the loss, otherwise it will return the logits.

        Args:
            input_ids (torch.Tensor): Input ids of the model.
            labels (torch.Tensor, optional): Labels of the model (as input_ids).
        """
        return self.model(input_ids=input_ids, labels=labels)


    