import os
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim, nn
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel, T5ForConditionalGeneration, T5Tokenizer
import lightning.pytorch as pl
import torch
from typing import Tuple, Optional


class CLIPDSIDecoder(nn.Module):
    def __init__(self, pretrained_language_model_name):
        super().__init__()
        self.decoder = T5ForConditionalGeneration.from_pretrained(pretrained_language_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_language_model_name)
        self.start_input_id = [torch.tensor([self.tokenizer.bos_token_id])] # possibly only needed for beam search (inference)

    def forward(
        self, 
        encoder_last_hidden_state: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        encoder_attentions: Optional[Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        return self.decoder(
            input_ids=self.start_input_id,
            attention_mask=attention_mask,
            encoder_outputs=(encoder_last_hidden_state, encoder_hidden_states, encoder_attentions),
            labels=labels
        )


class CLIPDSI(pl.LightningModule):
    def __init__(self, encoder_model_name, decoder_model_name):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(encoder_model_name, cache_dir="cache")
        self.text_model = CLIPTextModel.from_pretrained(encoder_model_name, cache_dir="cache")
        self.vision_model = CLIPVisionModel.from_pretrained(encoder_model_name, cache_dir="cache")
        self.decoder = CLIPDSIDecoder(decoder_model_name)

        # freeze encoder layers
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.val_step_outputs = []


    def forward(self, input_ids=None, attention_mask=None, position_ids=None, image=None, labels=None):
        with torch.no_grad():
            # activate text encoder
            if input_ids is not None:
                output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True)

            # activate image encoder
            elif image is not None:
                output = self.image_model(pixel_values=image, return_dict=True)
                
            del output["pooler_output"]

        output = self.decoder(**output, labels=labels)
        
        return output

    def training_step(self, batch, batch_idx):
        # filter so that correct batching can be applied to forward
        im = batch["image"][image_example == True]
        im_labels = batch["labels"][image_example == True]
        
        query_input_ids = batch["input_ids"][image_example == False]
        query_attention_mask = batch["attention_mask"][image_example == False]
        query_position_ids = batch["position_ids"][image_example == False]
        query_labels = batch["labels"][image_example == False]

        query_loss = 0
        image_loss = 0

        if query_input_ids:
            query_loss = self.forward(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                position_ids=query_position_ids,
                labels=query_labels
            ).loss
        if im:
            image_loss = self.forward(
                image=im,
                labels=im_labels
            ).loss

        loss = query_loss + image_loss

        self.log('train_loss', loss, batch_size=batch['input_ids'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: implement beam search (normal inference) for validation
        # pass