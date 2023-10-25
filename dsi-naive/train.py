from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from dataset import T5SearchDataset
from model import T5SearchIndex
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_path = "../data/nq10k/multi_task_train.json"
val_data_path = "../data/nq10k/multi_task_train.json"
test_data_path = "../data/nq10k/validation.json"

model_name_or_path = "t5-large"
batch_size = 8
max_doc_length = 32

tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)

train_dataset = T5SearchDataset(train_data_path, tokenizer, max_doc_length)
val_dataset = T5SearchDataset(val_data_path, tokenizer, max_doc_length)
test_dataset = T5SearchDataset(test_data_path, tokenizer, max_doc_length)

# Define the train, validation and test dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the T5SearchIndex model
model = T5SearchIndex(model_name_or_path).to(device)

# Fine-tune the model on the train dataset
model.finetune(train_dataloader, val_dataloader, epochs=3, learning_rate=3e-5, warmup_steps=100)

# Evaluate the model on the test dataset
model.evaluate(test_dataloader)


def finetune(model, train_dataloader, val_dataloader, epochs=1, learning_rate=3e-5, warmup_steps=0):
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, anneal_strategy='linear', warmup_steps=warmup_steps)

    for epoch in range(epochs):
        model.model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

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

def evaluate(model, dataloader):
    model.model.eval()
    with torch.no_grad():
        top1 = 0
        top10 = 0
        for batch in dataloader:
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