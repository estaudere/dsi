from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from transformers.trainer import Trainer
from torch import nn
import torch
from dataset import EvalCollator
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

"""Class to handle training of transformers DSI model."""
class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], 
               Optional[torch.Tensor], 
               Optional[torch.Tensor]]:
        
        model.eval()
        with torch.no_grad():

            # greedy search
            # TODO: implement beam search to find top-k docs
            doc_ids = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                early_stopping=True)
        return (None, doc_ids, inputs['labels'])
    

"""Class to handle query evaluation during training."""
class QueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=EvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries', disable=True):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log(20, f"Hits@1: {hit_at_1 / len(self.test_dataset)}, Hits@10: {hit_at_10 / len(self.test_dataset)}")
        print(f"Hits@1: {hit_at_1 / len(self.test_dataset)}, Hits@10: {hit_at_10 / len(self.test_dataset)}")


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}