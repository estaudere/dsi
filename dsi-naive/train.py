from dataset import DSIDataset
from torch import utils
from model import DSI
import lightning.pytorch as pl
from transformers import T5Tokenizer
import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-s', '--steps', type=int, default=1_000_000, help="Number of steps")
parser.add_argument('-d', '--dataset', type=str, default='10k', choices=['1k', '10k'], help="NQ dataset size")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('-l', '--logger', type=str, default='csv', choices=['csv', 'tensorboard'], help="Logger type for Lightning")
parser.add_argument('-m', '--model', type=str, default='t5-large', help="HuggingFace base model name")
parser.add_argument('--val_epochs', type=int, default=30, help="Check val after every n epochs")
parser.add_argument('--log_steps', type=int, default=10, help="Log every n steps")
parser.add_argument('--ckpt_path', type=str, help="Checkpoint path to resume from")

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


def main(args):
    if args.dataset == "1k":
        TRAIN = "../data/nq1k/multi_task_train.json"
        VAL = "../data/nq1k/validation.json"
    else:
        TRAIN = "../data/nq10k/multi_task_train.json"
        VAL = "../data/nq10k/validation.json"

    tokenizer = T5Tokenizer.from_pretrained(args.model, cache_dir="cache")
    train_dataset = DSIDataset(TRAIN, tokenizer)
    val_dataset = DSIDataset(VAL, tokenizer)

    train_dataloader = utils.data.DataLoader(train_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True)
    val_dataloader = utils.data.DataLoader(val_dataset)

    restrict_decode_vocab = get_restrict_fn(tokenizer)
    del tokenizer # a new tokenizer will be created in the model

    model = DSI(args.model, restrict_decode_vocab=restrict_decode_vocab)

    # instantiate logger
    if args.logger == "csv":
        logger = pl.loggers.CSVLogger('logs/')
    else:
        logger = pl.loggers.TensorBoardLogger('logs/')


    trainer = pl.Trainer(
        check_val_every_n_epoch=args.val_epochs,
        max_steps=args.steps,
        log_every_n_steps=args.log_steps,
        logger=logger,
        enable_progress_bar=False,
        accelerator='gpu',
        strategy='ddp',
        num_nodes=4,
        devices=3
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt_path)

if __name__=="__main__":
    args = parser.parse_args()

    main(args)