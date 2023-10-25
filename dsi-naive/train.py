from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from dataset import T5SearchDataset
from model import T5SearchIndex
import torch
import numpy as np
from tqdm import tqdm

DISABLE_TQDM = False

def finetune(model, train_dataloader, epochs=1, learning_rate=3e-5):
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, anneal_strategy='linear')

    for epoch in range(epochs):
        model.model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}", disable=DISABLE_TQDM):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

def get_restrict_fn(tokenizer):
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

def evaluate(model, dataloader, restrict_decode_vocab):
    model.model.eval()
    with torch.no_grad():
        top1 = 0
        top10 = 0
        for batch in tqdm(dataloader, desco="Evaluating", disable=DISABLE_TQDM):
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            batch_beams = model.model.generate(input_ids=input_ids, 
                                               max_length=20,
                                               num_beams=10,
                                               prefix_allowed_tokens_fn=restrict_decode_vocab,
                                               num_return_sequences=10,
                                               early_stopping=True).reshape(input_ids.shape[0], 10, -1)
            for beams, label in zip(batch_beams, labels):
                rank_list = model.tokenizer.batch_decode(beams, skip_special_tokens=True)
                hits = np.where(np.array(rank_list)[:10] == label)[0]
                if len(hits) != 0:
                    top10 += 1
                    if hits[0] == 0:
                        top1 += 1
        print(f"Top 1 Accuracy: {top1/len(dataloader):.4f}, Top 10 Accuracy: {top10/len(dataloader):.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data_path = "../data/nq10k/multi_task_train.json"
    val_data_path = "../data/nq10k/multi_task_train.json"
    test_data_path = "../data/nq10k/validation.json"

    model_path = "t5-large"
    batch_size = 16
    max_doc_length = 32

    model = T5SearchIndex(model_path, cache_dir="cache").to(device)

    train_dataset = T5SearchDataset(train_data_path, model.tokenizer, max_doc_length)
    val_dataset = T5SearchDataset(val_data_path, model.tokenizer, max_doc_length)
    test_dataset = T5SearchDataset(test_data_path, model.tokenizer, max_doc_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    finetune(model, train_dataloader, epochs=3, learning_rate=3e-5)

    restrict_decode_vocab = get_restrict_fn(model.tokenizer)
    evaluate(model, test_dataloader, restrict_decode_vocab)