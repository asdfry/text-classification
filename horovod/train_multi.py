import csv
import time
import torch
import logging
import argparse
import horovod.torch as hvd

from torch.optim import AdamW
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-b", "--batch_size", type=int, required=True)
parser.add_argument("-g", "--gpu", type=int, required=True)
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-hf", "--half", action="store_true")
args = parser.parse_args()


# Set logger
logging.basicConfig(
    format="%(asctime)s\t%(levelname)s\t%(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(f"logs/log_gpu_{args.gpu}.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Initialize Horovod
hvd.init()


# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())


# Prefix
model_name = "./pretrained_model"


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
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=77).select(range(160))
    small_valid_dataset = tokenized_datasets["valid"].shuffle(seed=77).select(range(32))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        small_train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        small_valid_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        small_train_dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    valid_dataloader = torch.utils.data.DataLoader(
        small_valid_dataset, batch_size=args.batch_size, sampler=valid_sampler
    )

else:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        tokenized_datasets["train"],
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        tokenized_datasets["valid"],
        num_replicas=hvd.size(),
        rank=hvd.rank(),
        shuffle=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["train"], batch_size=args.batch_size, sampler=train_sampler
    )
    valid_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["valid"], batch_size=args.batch_size, sampler=valid_sampler
    )


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
model.cuda()


# Set optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())


# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)


# Create learning rate scheduler
num_training_steps = args.epoch * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


# Iterate data loader
start_time = time.time()
for epoch in range(args.epoch):
    # Train
    model.train()
    loss_per_epoch = 0

    for step, batch in enumerate(train_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss_per_epoch += loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        logger.info(f"[train] step: {step + 1}/{len(train_dataloader)}, loss: {loss_per_epoch / (step + 1)}")

    # Valid
    model.eval()
    loss_per_epoch = 0
    correct_per_epoch = 0

    for step, batch in enumerate(valid_dataloader):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss_per_epoch += outputs.loss

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct_per_epoch += len([i for i in torch.eq(predictions, batch["labels"]) if i])

        logger.info(f"[valid] step: {step + 1}/{len(valid_dataloader)}, loss: {loss_per_epoch / (step + 1)}")

    logger.info(f"accuracy: {(correct_per_epoch / (len(valid_dataloader) * args.batch_size)) * 100} %")

    # save_path = f"./models/epoch-{epoch + 1}"
    # model.save_pretrained(save_path)
    # logger.info(f"model saved: {save_path}")
    logger.info(f"elapsed time: {time.time() - start_time} sec")
