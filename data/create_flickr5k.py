"""
Script to split the simplified NQ dataset into a 1k training and eval dataset.
"""

import json
import pandas as pd
import os
from PIL import Image

NUM_TRAIN = 5000 # the total resulting train dataset size (queries + documents)
NUM_EVAL = 500 # the number of evaluation queries

NUM_INDEX_TRAIN = 4000 # the number of documents to index
assert NUM_INDEX_TRAIN <= NUM_TRAIN, \
    "NUM_INDEX_TRAIN should be smaller than NUM_TRAIN"
NUM_QUERY_TRAIN = NUM_TRAIN - NUM_INDEX_TRAIN
assert NUM_EVAL + NUM_QUERY_TRAIN <= NUM_TRAIN, \
    "NUM_EVAL + NUM_QUERY_TRAIN should be smaller than NUM_TRAIN so it is possible to index all docs"

id_set = set()

df = pd.read_csv("flickr_annotations_30k.csv")

with open('flickr5k/multi_task_train.json', 'w') as tf, \
        open('flickr5k/validation.json', 'w') as vf:
    for _, row in df.iterrows():
        docid = row['img_id'] # we use title as the match key
        if docid not in id_set:
            # check if the image exists in the directory
            try:
                Image.open(os.path.join('flickr30k-images', row['filename']))
            except:
                print(f"Image {row['filename']} not found")
                continue

            id_set.add(docid)
            img = row["filename"]
            query_text = json.loads(row["raw"])[0]

            # add an indexing example (document -> docid)
            if NUM_INDEX_TRAIN > 0:
                jitem = json.dumps({'img_id': str(docid), 
                                    'img_filename': img})
                tf.write(jitem + '\n')
                NUM_INDEX_TRAIN -= 1
            
            # add a query example (question -> docid)
            jitem = json.dumps({'img_id': str(docid), 
                                'query_text': query_text})
            if NUM_QUERY_TRAIN > 0:
                tf.write(jitem + '\n')
                NUM_QUERY_TRAIN -= 1
            elif NUM_EVAL > 0:
                vf.write(jitem + '\n')
                NUM_EVAL -= 1

            if len(id_set) == NUM_TRAIN:
                break

        print(f"Creating training and validation dataset: {'{:.1%}'.format(len(id_set)/(NUM_TRAIN + NUM_EVAL))}", end='\r')