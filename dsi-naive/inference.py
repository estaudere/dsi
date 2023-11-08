from dataset import DSIDataset
from torch import utils
from model import DSI
from transformers import T5Tokenizer
import os
import numpy as np
from train import get_restrict_fn
import lightning.pytorch as pl


VAL = "../data/nq1k/multi_task_train.json"

tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir="cache")
val_dataset = DSIDataset(VAL, tokenizer)
val_dataloader = utils.data.DataLoader(val_dataset, batch_size=32)
restrict_decode_vocab = get_restrict_fn(tokenizer)
del tokenizer # a new tokenizer will be created in the model

model = DSI("t5-small", restrict_decode_vocab=restrict_decode_vocab)


trainer = pl.Trainer(limit_val_batches=32, 
                    log_every_n_steps=5)
trainer.validate(model, val_dataloader, ckpt_path="./logs/lightning_logs/version_8/checkpoints/epoch=499-step=16000.ckpt")