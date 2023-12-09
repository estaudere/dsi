# dsi

This repository contains some code for an implementation of ["Transformer Memory as a Differentiable Search Index"](https://arxiv.org/abs/2202.06991) (Tay et al. 2022) in the **dsi-naive** folder (with the naive representation of docids). The results of the paper have not been fully tested via this code yet. To train with Pytorch Lightning,

```bash
python train.py
```

inside the folder. View the file for more details on the arguments.

## CLIP-DSI

The **clip-dsi** folder contains code for a version of DSI that uses CLIP embeddings as the input to the language model in order to apply the DSI paradigm to image retrieval from a text query. This was developed by me for an original research project.

## Datasets

Code to parse and create the training and eval sets for both the Natural Questions and Flickr30k are available. Follow the instructions in the README in the folder.