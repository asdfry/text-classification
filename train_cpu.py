import csv
import torch
import evaluate

from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"

datasets = DatasetDict()
datasets["train"] = load_dataset("csv", data_files="./data/train.csv")["train"]
datasets["valid"] = load_dataset("csv", data_files="./data/valid.csv")["train"]

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")


tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=77).select(range(1000))
small_valid_dataset = tokenized_datasets["valid"].shuffle(seed=77).select(range(1000))

batch_size = 32
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(small_valid_dataset, batch_size=batch_size)

with open("./data/id_to_label.csv", "r") as f:
    reader = csv.DictReader(f)
    id2label = {row["id"]: row["label"] for row in reader}
label2id = {label: id for id, label in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=18, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cpu")
model.to(device)
metric = evaluate.load("accuracy")

for epoch in range(num_epochs):
    print(f"--- epoch {epoch + 1} ---")

    model.train()
    loss_per_epoch = 0
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_per_epoch += loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print(f"[{step + 1} / {len(train_dataloader)}] loss: {loss_per_epoch / (step + 1)}")

    model.eval()
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    print(f"metric: {metric.compute()}")

    model.save_pretrained(f"./models/epoch-{epoch + 1}")
