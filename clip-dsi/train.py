from dataset import CLIPDSIDataset
from torch import utils
from model import CLIPDSI
import lightning.pytorch as pl
from transformers import T5Tokenizer, CLIPProcessor
import os
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-s', '--steps', type=int, default=1_000_000, help="Number of steps")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('-l', '--logger', type=str, default='csv', choices=['csv', 'tensorboard'], help="Logger type for Lightning")
parser.add_argument('-m', '--model', type=str, default='openai/clip-vit-base-patch32', help="HuggingFace CLIP base model name")
parser.add_argument('--decoder_model', type=str, default='t5-small', help="HuggingFace decoder language model name")
parser.add_argument('--val_epochs', type=int, default=30, help="Check val after every n epochs")
parser.add_argument('--log_steps', type=int, default=10, help="Log every n steps")
parser.add_argument('--ckpt_path', type=str, help="Checkpoint path to resume from")

def main(args):
    TRAIN = "../data/flickr500/multi_task_train.json"
    VAL = "../data/flickr500/validation.json"
    IMAGE_DIR = "../data/flickr30k-images/"

    processor = CLIPProcessor.from_pretrained(args.model, cache_dir="cache")
    tokenizer = T5Tokenizer.from_pretrained(args.decoder_model, cache_dir="cache")
    train_dataset = CLIPDSIDataset(TRAIN, processor, IMAGE_DIR, tokenizer)
    val_dataset = CLIPDSIDataset(VAL, processor, IMAGE_DIR, tokenizer)
    del processor # a new processor will be created in the model
    del tokenizer # a new tokenizer will be created in the model

    train_dataloader = utils.data.DataLoader(train_dataset, 
                                             batch_size=args.batch_size, 
                                             shuffle=True)
    val_dataloader = utils.data.DataLoader(val_dataset,
                                           batch_size=args.batch_size)

    model = CLIPDSI(args.model, args.decoder_model)

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
        enable_progress_bar=True,
        accelerator='gpu',
        # strategy='ddp',
        num_nodes=1, # num_nodes=4
        devices=1 # devices=3
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt_path)

if __name__=="__main__":
    args = parser.parse_args()

    main(args)