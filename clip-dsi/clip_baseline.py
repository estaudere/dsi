# load in documents and files
import os
import json
from PIL import Image
import torch
from tqdm import tqdm

def load_corpus_images(train_path: str):
    """
    Load in the corpus of documents to be used for training.
    """
    docs = [] # list of images in the form (imgid, filename)
    with open(train_path, 'r') as f:
        for line in f:
            l = json.loads(line)
            if l.get("img_filename"):
                docs.append((l["img_id"], l["img_filename"]))
    return docs

def embed_images(docs: list, imgdir: str, model, processor):
    """
    Embed images using CLIP model.
    """
    img_embeddings = []
    for img_id, img_filename in tqdm(docs, desc="Embedding images"):
        img = Image.open(os.path.join(imgdir, img_filename))
        img_input = processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            img_embedding = model.get_image_features(**img_input)
        img_embeddings.append((img_id, img_embedding))
    return img_embeddings

def retrieve_images(query: str, img_embeddings: list, model, processor):
    """
    Retrieve images given a query.
    """
    query_input = processor(text=query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.get_text_features(**query_input)

        scores = []
        for img_id, img_embedding in img_embeddings:
            # use scaled dot product similarity
            similarity = torch.matmul(query_embedding, img_embedding.T)
            scores.append((img_id, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
def compute_metrics(scores, true_img_id):
    """
    Compute top1, top5, and top10 for the query.
    """
    top1 = 0
    top5 = 0
    top10 = 0
    for img_id, _ in scores[:10]:
        if img_id == true_img_id:
            top10 += 1
            if img_id == scores[0][0]:
                top1 += 1
    for img_id, _ in scores[:5]:
        if img_id == true_img_id:
            top5 += 1
    return top1, top5, top10

if __name__=="__main__":
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="cache")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="cache")

    TRAIN_PATH = "../data/flickr500/multi_task_train.json"
    VAL_PATH = "../data/flickr500/validation.json"
    IMG_DIR = "../data/flickr30k-images"

    docs = load_corpus_images(TRAIN_PATH)
    img_embeddings = embed_images(docs, IMG_DIR, model, processor)

    # load validation data
    val = []
    with open(VAL_PATH, 'r') as f:
        for line in f:
            l = json.loads(line)
            val.append((l["img_id"], l["query_text"]))

    top1 = 0
    top5 = 0
    top10 = 0
    # retrieve images for each query
    for img_id, query in tqdm(val, desc="Retrieving images"):
        scores = retrieve_images(query, img_embeddings, model, processor)
        top1_, top5_, top10_ = compute_metrics(scores, img_id)
        top1 += top1_
        top5 += top5_
        top10 += top10_

    top1 /= len(val)
    top5 /= len(val)
    top10 /= len(val)


    print(f"Top1: {top1}, Top5: {top5}, Top10: {top10}")
