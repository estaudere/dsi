import os
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim, nn
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel
import lightning.pytorch as pl
import torch


class CLIPDSI(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir="cache")
        self.text_model = CLIPTextModel.from_pretrained(model_name, cache_dir="cache")
        self.vision_model = CLIPVisionModel.from_pretrained(model_name, cache_dir="cache")
        self.val_step_outputs = []

        # TODO: potentially freeze encoder layers of text_model and vision_model

    def forward(self, input_ids=None, attention_mask=None, image=None):

        # activate text tower
        if input_ids is not None:
            output = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            # compute embeddings?? (obtained from the projection on the text model)
            # TODO: pass output (+ potential embeddings) to decoder layers (pooler_output or last_hidden_state) (the former most likely)
            # 

        # activate image tower
        elif image is not None:
            output = self.image_model(pixel_values=image, return_dict=True)

            # TODO pass output to decoder layers (pooler_output or last_hidden_state)