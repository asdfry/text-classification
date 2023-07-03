import csv
import time
import torch
import argparse
import evaluate

from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


# Instantiate one in an accelerator object
accelerator = Accelerator()


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-t", "--test", action="store_true", default=False)
parser.add_argument("-hf", "--half", action="store_true", default=False)
args = parser.parse_args()


# Prefix
model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"


# Create dataset
train_data_path = "./data/train_half.csv" if args.half else "./data/train.csv"
valid_data_path = "./data/valid_half.csv" if args.half else "./data/valid.csv"
datasets = DatasetDict()
datasets["train"] = load_dataset("csv", data_files=train_data_path)["train"]
datasets["valid"] = load_dataset("csv", data_files=valid_data_path)["train"]


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
if args.test:
    # accelerator.print("This process is test mode")
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=77).select(range(160))
    small_valid_dataset = tokenized_datasets["valid"].shuffle(seed=77).select(range(32))
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(small_valid_dataset, batch_size=args.batch_size)
else:
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=args.batch_size)


# Prepare id-label mapper
id_to_label_path = "./data/id_to_label_half.csv" if args.half else "./data/id_to_label.csv"
with open(id_to_label_path, "r") as f:
    reader = csv.DictReader(f)
    id2label = {row["id"]: row["label"] for row in reader}
label2id = {label: id for id, label in id2label.items()}


# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=9 if args.half else 18,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


# Set optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)


# Create learning rate scheduler
num_training_steps = args.epoch * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# Load metric method
metric = evaluate.load("accuracy")


# Ready for training
model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
)


# Train
start_time = time.time()
for epoch in range(args.epoch):
    accelerator.print("-" * 20, f"epoch {epoch + 1}", "-" * 20)

    model.train()
    loss_per_epoch = 0
    for step, batch in enumerate(train_dataloader):
        batch = {k: v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_per_epoch += loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        accelerator.print(f"[train] step: {step + 1}/{len(train_dataloader)}, loss: {loss_per_epoch / (step + 1)}")

    model.eval()
    loss_per_epoch = 0
    for step, batch in enumerate(valid_dataloader):
        batch = {k: v for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        loss_per_epoch += outputs.loss
        
        accelerator.print(f"[valid] step: {step + 1}/{len(valid_dataloader)}, loss: {loss_per_epoch / (step + 1)}")

        # logits = outputs.logits
        # predictions = torch.argmax(logits, dim=-1)
        # all_predictions, all_targets = accelerator.gather_for_metrics((predictions, batch["labels"]))
        # metric.add_batch(predictions=all_predictions, references=all_targets)

    # accelerator.print(f"metric: {metric.compute()}")

    # save_path = f"./models/epoch-{epoch + 1}"
    # unwraped_model = accelerator.unwrap_model(model)
    # unwraped_model.save_pretrained(save_path)
    # accelerator.print(f"model saved: {save_path}")
    accelerator.print(f"elapsed time: {time.time() - start_time} sec")
