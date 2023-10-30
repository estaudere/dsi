import os
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim, nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import lightning.pytorch as pl
import numpy as np

class DSI(pl.LightningModule):
    def __init__(self, model_name, restrict_decode_vocab=None):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="cache")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="cache")
        self.restrict_decode_vocab = restrict_decode_vocab

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids=input_ids, labels=labels)
    
    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log('val_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        batch_beams = self.model.generate(input_ids=input_ids, 
                                            max_length=20,
                                            num_beams=10,
                                            prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                                            num_return_sequences=10,
                                            early_stopping=True).reshape(input_ids.shape[0], 10, -1)
        for beams, label in zip(batch_beams, labels):
            rank_list = self.tokenizer.batch_decode(beams, skip_special_tokens=True)
            hits = np.where(np.array(rank_list)[:10] == label)[0]
            if len(hits) != 0:
                top10 += 1
                if hits[0] == 0:
                    top1 += 1
        self.log_dict({'top1': top1, 'top10': top10})