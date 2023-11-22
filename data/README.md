# Dataset creation

In order to create the relevant datasets, following the instructions for each below.

## Natural Questions

1. Download the simplified dataset from [Google](https://ai.google.com/research/NaturalQuestions/download) as a `.jsonl` file.
2. Run **create_nq10k.py** to generate train and validation datasets, where the train set contains both question->doc and document->docid pairs, and the validation set contains question->doc pairs.

For now, a parse script to generate a 10k sample of Natural Questions is provided. It is possible to generate larger samples by changing the constants at the top of the relevant script.

## Flickr

1. Download the 30k dataset (CSV and image zip) from [HuggingFace](https://huggingface.co/datasets/nlphuji/flickr30k). Unzip
the image zip into a folder called **flickr30k-images**.
2. Run **create_flickr5k.py** to generate the train and validation datasets, where the train set contains both image->imgid 
and query->imgid pairs, and the validation set contains query->imgid pairs.