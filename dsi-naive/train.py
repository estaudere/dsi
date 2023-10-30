from dataset import DSIDataset
from torch import utils
from lightning_model import DSI
import lightning.pytorch as pl
from transformers import T5Tokenizer

def get_restrict_fn(tokenizer: T5Tokenizer):
    # we only allow the model to generate integer tokens (docids)
    SPIECE_UNDERLINE = "▁"
    int_token_ids = []
    for token, id in tokenizer.get_vocab().items():
        if token.startswith(SPIECE_UNDERLINE) and token[1:].isdigit():
            int_token_ids.append(id)
        elif token == SPIECE_UNDERLINE:
            int_token_ids.append(id)
        elif token.isdigit():
            int_token_ids.append(id)
    int_token_ids.append(tokenizer.eos_token_id)
    int_token_ids.append(tokenizer.pad_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return int_token_ids
    
    return restrict_decode_vocab

if __name__=="__main__":
    TRAIN = "../data/nq1k/multi_task_train.jsonl"
    VAL = "../data/nq1k/validation.jsonl"
    BATCH_SIZE = 32
    EPOCHS = 100
    VAL_EPOCHS = 50

    tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir="cache")
    train_dataset = DSIDataset(TRAIN, tokenizer)
    val_dataset = DSIDataset(VAL, tokenizer)

    train_dataloader = utils.data.DataLoader(train_dataset)
    val_dataloader = utils.data.DataLoader(val_dataset)

    restrict_decode_vocab = get_restrict_fn(tokenizer)
    del tokenizer # a new tokenizer will be created in the model

    model = DSI("t5-small", restrict_decode_vocab=restrict_decode_vocab)

    trainer = pl.Trainer(limit_train_batches=BATCH_SIZE, 
                         limit_val_batches=BATCH_SIZE, 
                         check_val_every_n_epoch=VAL_EPOCHS,
                         max_epochs=EPOCHS)
    trainer.fit(model, train_dataloader, val_dataloader)