import csv
import torch
import argparse
import evaluate

from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-b", "--batch_size", type=int, required=True)
args = parser.parse_args()


# Prefix
model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"


# Create dataset
datasets = DatasetDict()
datasets["train"] = load_dataset("csv", data_files="./data/train.csv")["train"]
datasets["valid"] = load_dataset("csv", data_files="./data/valid.csv")["train"]


# Load tokenizer and tokenize
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")


tokenized_datasets = datasets.map(tokenize_function, batched=True)


# Postprocess for train with native pytorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


# Create dataloader
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
valid_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=args.batch_size)


# Prepare id-label mapper
with open("./data/id_to_label.csv", "r") as f:
    reader = csv.DictReader(f)
    id2label = {row["id"]: row["label"] for row in reader}
label2id = {label: id for id, label in id2label.items()}


# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=18, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)


# Set optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)


# Create learning rate scheduler
num_training_steps = args.epoch * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


# Set device and create metric
device = torch.device("cpu")
model.to(device)
metric = evaluate.load("accuracy")


# Train
for epoch in range(args.epoch):
    print("-" * 10, f"epoch {epoch + 1}", "-" * 10)

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
