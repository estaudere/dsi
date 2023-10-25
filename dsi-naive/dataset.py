import torch
import json
from torch.utils.data import Dataset

class T5SearchDataset(Dataset):
    def __init__(self, train_data_path, tokenizer, max_doc_length=32):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.data = []
        with open(train_data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        # the data is (text, text_id) pairs, where text could either be the
        # text of a document or a question, and text_id is the id of the text
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, text_id = self.data[idx]["text"], self.data[idx]["text_id"]

        encoding = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_doc_length)
        target_encoding = self.tokenizer(text_id,
                                    return_tensors="pt",
                                    truncation='only_first',
                                    max_length=1)
        
        labels = target_encoding.input_ids

        # set to -100 to ignore loss
        labels[labels == self.tokenizer.pad_token_id] = -100
    
        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "labels": labels,
            "target_text": str(text_id),
        }