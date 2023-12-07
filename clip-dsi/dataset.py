import json
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class CLIPDSIDataset(Dataset):
    def __init__(self, train_data_path, processor, image_dir, tokenizer):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.data = []
        with open(train_data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.cache = [None for _ in range(len(self.data))]
        self.image_dir = image_dir

        # the data is (text, text_id) pairs, where text could either be the
        # text of a document or a question, and text_id is the id of the text
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.cache[idx] != None:
            return self.cache[idx]
        else:
            img_id, query_text, filename = self.data[idx].get("img_id"), \
                    self.data[idx].get("query_text"), \
                    self.data[idx].get("img_filename")

            # TODO: maybe tokenize using the decoder tokenizer instead of the encoder tokenizer
            # I THINK THIS WAS THE ISSUE
            id_encoding = self.tokenizer(text=img_id,
                                    return_tensors="pt",
                                    truncation='longest_first',
                                    padding="max_length",
                                    max_length=32)
            labels = id_encoding.input_ids[0]
            labels[labels == self.tokenizer.pad_token_id] = -100

            if query_text is not None:
                query_encoding = self.processor(text=query_text,
                                                return_tensors="pt",
                                                truncation='longest_first',
                                                padding="max_length",
                                                max_length=32)
                
                # print(query_encoding)
                self.cache[idx] = {
                    "input_ids": query_encoding.input_ids[0].long(),
                    "attention_mask": query_encoding.attention_mask[0].long(),
                    # "position_ids": query_encoding.position_ids[0], 
                    "image": torch.zeros((3, 224, 224)).long(), # create dummy image tensor,
                    "labels": labels.long(),
                    "target_id": str(img_id),
                    "image_example": False
                }

            elif filename is not None:
                image = Image.open(os.path.join(self.image_dir, filename))
                processed_image = self.processor(images=[image],
                                                return_tensors="pt")
                self.cache[idx] = {
                    "input_ids": torch.zeros((32)).long(), # create dummy input_ids tensor
                    "attention_mask": torch.zeros((32)).long(), # create dummy attention_mask tensor
                    # "position_ids": torch.zeros((1, 1)), # create dummy position_ids tensor
                    "image": processed_image.pixel_values[0].long(),
                    "labels": labels.long(),
                    "target_id": str(img_id),
                    "image_example": True
                }                             
        
            return self.cache[idx]