import argparse
import pandas as pd
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", type=int, required=True)
parser.add_argument("-n", "--number_sample", type=int, required=True)
args = parser.parse_args()

df = pd.read_csv("./data/valid.csv")
id_to_label = pd.read_csv("./data/id_to_label.csv")
id2label = id_to_label["label"].to_list()

pretrained_model = "classla/xlm-roberta-base-multilingual-text-genre-classifier"
finetuned_model = f"./models/epoch-{args.epoch}"

classifier = pipeline(task="text-classification", model=finetuned_model, tokenizer=pretrained_model)

samples = df.sample(n=args.number_sample)
texts = samples["text"].to_list()
labels = [id2label[int(id)] for id in samples["label"].to_list()]

infer_results = classifier(texts)

for i in range(args.number_sample):
    print(f"Text: {texts[i]}\n"
          f"Label: {labels[i]}\n"
          f"Infered Label: {infer_results[i]['label']}\n"
          f"Score: {infer_results[i]['score']}\n")
