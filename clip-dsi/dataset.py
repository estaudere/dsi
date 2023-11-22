import json
from torch.utils.data import Dataset
import os

class CLIPDSIDataset(Dataset):
    def __init__(self, train_data_path, processor):
        super().__init__()
        self.processor = processor
        self.data = []
        with open(train_data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.cache = [None for _ in range(len(self.data))]

        # the data is (text, text_id) pairs, where text could either be the
        # text of a document or a question, and text_id is the id of the text
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.cache[idx] != None:
            return self.cache[idx]
        else:
            img_id, query_text, filename = self.data[idx].get("img_id"), 
                    self.data[idx].get("query_text"), 
                    self.data[idx].get("img_filename")


            id_encoding = self.processor(text=img_id,
                                    return_tensors="pt")
            labels = id_encoding.input_ids[0]
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            if query_text is not None:
                query_encoding = self.processor(text=query_text,
                                                return_tensors="pt")
                self.cache[idx] = {
                    "input_ids": query_encoding.input_ids[0]
                    "attention_mask": query_encoding.attention_mask[0],
                    "labels": labels,
                    "target_id": str(img_id)
                }

            elif filename is not None:
                # TODO: load image and call processor given image=
                image = _
                processed_image = self.processor(image=image,
                                                return_tensors="pt")
                # TODO check output of processor on image
                self.cache[idx] = {
                    "image": processed_image,

                }                                    
        
            return self.cache[idx]