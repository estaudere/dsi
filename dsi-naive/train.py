import logging
from typing import List, Dict, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments
from dsi_trainer import DSITrainer, QueryEvalCallback, compute_metrics
from dataset import TrainDataset, TrainCollator

def main():
    model_name = "t5-large"
    L = 32  # only use the first 32 tokens of documents (including title)

    logging.basicConfig(filename="results.log", filemode='w')
    logger = logging.getLogger()

    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir='cache')
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='cache')

    train_dataset = TrainDataset(path_to_data='../data/nq10k/multi_task_train.json',
                                         max_length=L,
                                         cache_dir='cache',
                                         tokenizer=tokenizer)
    
    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    eval_dataset = TrainDataset(path_to_data='../data/nq10k/multi_task_train.json',
                                        max_length=L,
                                        cache_dir='cache',
                                        tokenizer=tokenizer)
    
    # This is the actual eval set.
    test_dataset = TrainDataset(path_to_data='../data/nq10k/validation.json',
                                        max_length=L,
                                        cache_dir='cache',
                                        tokenizer=tokenizer)

    
    # we only allow the model to generate integer tokens (docids)
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.0005,
        warmup_steps=10000,
        # weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy='steps',
        eval_steps=10000,
        max_steps=100000,
        dataloader_drop_last=False,  # necessary
        logging_steps=50,
        save_strategy='no',
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=8,
        # gradient_accumulation_steps=2
    )

    trainer = DSITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainCollator(
            tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[QueryEvalCallback(test_dataset, logger, restrict_decode_vocab, training_args, tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab
    )

    trainer.train()

    # save the model weights
    trainer.save("../models/dsi-naive-nq10k")


if __name__ == "__main__":
    main()
