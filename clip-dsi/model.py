import os
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim, nn
from transformers import (CLIPProcessor, CLIPTextModel, CLIPVisionModel, 
                          T5ForConditionalGeneration, T5Tokenizer)
from transformers.modeling_outputs import BaseModelOutputWithPooling
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
from collections import namedtuple


class CLIPDSIDecoder(nn.Module):
    def __init__(self, pretrained_language_model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_language_model_name, cache_dir="cache")
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_language_model_name, cache_dir="cache")
        self.start_input_id = [torch.tensor([self.model.config.decoder_start_token_id])]

    def forward(
        self, 
        encoder_last_hidden_state: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        encoder_attentions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        return self.model(
            input_ids=self.start_input_id,
            attention_mask=attention_mask,
            encoder_outputs=(
                encoder_last_hidden_state, 
                encoder_hidden_states, 
                encoder_attentions
            ),
            labels=labels
        )


class CLIPDSI(pl.LightningModule):
    def __init__(self, encoder_model_name, decoder_model_name):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(encoder_model_name, 
                                                       cache_dir="cache")
        self.text_model = CLIPTextModel.from_pretrained(encoder_model_name, 
                                                        cache_dir="cache")
        self.vision_model = CLIPVisionModel.from_pretrained(encoder_model_name, 
                                                            cache_dir="cache")
        self.decoder = CLIPDSIDecoder(decoder_model_name)
        self.restrict_decode_vocab = self._get_restrict_token_fn(
            self.decoder.tokenizer)

        # freeze encoder layers
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.val_step_outputs = []


    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            position_ids=None, 
            image=None, 
            labels=None
    ):
        assert (input_ids is not None) ^ (image is not None), \
            "Either input_ids or image must be provided, but not both."
        
        with torch.no_grad():
            # activate text encoder
            if input_ids is not None:
                output = self.text_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True
                )

            # activate image encoder
            elif image is not None:
                output = self.vision_model(
                    pixel_values=image, 
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                
            output = {
                "encoder_last_hidden_state": output["last_hidden_state"],
                "encoder_hidden_states": output["hidden_states"],
                "encoder_attentions": output["attentions"]
            }

        output = self.decoder(**output, labels=labels)
        
        return output

    def training_step(self, batch, batch_idx):
        # filter so that correct batching can be applied to forward
        image_example = batch["image_example"]
        im = batch["image"][image_example == True]
        im_labels = batch["labels"][image_example == True]
        
        query_input_ids = batch["input_ids"][image_example == False]
        query_attention_mask = batch["attention_mask"][image_example == False]
        # query_position_ids = batch["position_ids"][image_example == False]
        query_labels = batch["labels"][image_example == False]

        query_loss = 0
        image_loss = 0

        if query_input_ids.shape[0] != 0:
            query_loss = self.forward(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                # position_ids=query_position_ids,
                labels=query_labels
            ).loss
        if im.shape[0] != 0:
            image_loss = self.forward(
                image=im,
                labels=im_labels
            ).loss

        loss = query_loss + image_loss

        self.log('train_loss', loss, batch_size=batch['input_ids'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        assert (batch["image_example"] == False).all(), \
            "validation_step() is only valid for query examples."
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_text = batch["target_id"]

        with torch.no_grad():
            output = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True, 
                return_dict=True
            )

            output = BaseModelOutputWithPooling(
                last_hidden_state=output["last_hidden_state"],
                hidden_states=output["hidden_states"],
                attentions=output["attentions"]
            )

        output = self.decoder.model.generate(
            encoder_outputs=output,
            max_length=20,
            num_beams=10,
            prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            num_return_sequences=10,
            early_stopping=True
        ).reshape(input_ids.shape[0], 10, -1)
        
        top1 = 0
        top10 = 0
        for beams, label in zip(output, target_text):
            rank_list = self.decoder.tokenizer.batch_decode(
                beams, skip_special_tokens=True
            )
            hits = np.where(np.array(rank_list)[:10] == label)[0]
            if len(hits) != 0:
                top10 += 1
                if hits[0] == 0:
                    top1 += 1
        top1 /= input_ids.shape[0]
        top10 /= input_ids.shape[0]
        self.val_step_outputs.append(torch.tensor([top1, top10]))
        return top1, top10
    
    def on_validation_epoch_end(self, outputs=None):
        top1 = torch.stack(self.val_step_outputs)[:, 0].mean()
        top10 = torch.stack(self.val_step_outputs)[:, 1].mean()
        self.log_dict({'top1': top1, 'top10': top10})
        return top1, top10
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def _get_restrict_token_fn(self, tokenizer: T5Tokenizer):
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