"""
Script to split the simplified NQ dataset into a 1k training and eval dataset.

- TODO: Are eval question answers also in the training set? i.e. have those docs
        been indexed during training?
"""

import json

NUM_TRAIN = 1000
NUM_EVAL = 200

NUM_INDEX_TRAIN = 800
assert NUM_INDEX_TRAIN <= NUM_TRAIN, \
    "NUM_INDEX_TRAIN should be smaller than NUM_TRAIN"
NUM_QUERY_TRAIN = NUM_TRAIN - NUM_INDEX_TRAIN
assert NUM_EVAL + NUM_QUERY_TRAIN <= NUM_TRAIN, \
    "NUM_EVAL + NUM_QUERY_TRAIN should be smaller than NUM_TRAIN so it is possible to index all docs"

id_set = set()
current_docid = 0

with open('v1.0-simplified_simplified-nq-train.jsonl', 'r') as f, \
        open('nq1k/multi_task_train.json', 'w') as tf, \
        open('nq1k/validation.json', 'w') as vf:
    for line in f:
        data = json.loads(line)
        docid = data['example_id'] # we use title as the match key
        if docid not in id_set:
            id_set.add(docid)
            doc_text = data['document_text']
            question_text = data['question_text']

            # add an indexing example (document -> docid)
            if NUM_INDEX_TRAIN > 0:
                jitem = json.dumps({'text_id': str(current_docid), 
                                    'text': 'document: ' + doc_text})
                tf.write(jitem + '\n')
                NUM_INDEX_TRAIN -= 1
            
            # add a query example (question -> docid)
            jitem = json.dumps({'text_id': str(current_docid), 
                                'text': 'question: ' + question_text})
            if NUM_QUERY_TRAIN > 0:
                tf.write(jitem + '\n')
                NUM_QUERY_TRAIN -= 1
            elif NUM_EVAL > 0:
                vf.write(jitem + '\n')
                NUM_EVAL -= 1

            current_docid += 1
            if len(id_set) == NUM_TRAIN:
                break

        print(f"Creating training and validation dataset: {'{:.1%}'.format(len(id_set)/(NUM_TRAIN + NUM_EVAL))}", end='\r')