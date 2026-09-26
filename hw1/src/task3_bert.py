"""Task 3: fine-tune bert-base-uncased on NYT (3 epochs, max_length=64).

Hardware notes (RTX 3050 Laptop, 4GB):
  - fp16=True halves activation memory, batch_size=16 fits comfortably
  - evaluation on validation each epoch; final metrics on the test split
"""
import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

from common import load_split, evaluate, save_result, LABELS, RESULTS_DIR
import os

# huggingface.co is unreachable on this network -> use the mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=2e-5,
                    help="learning rate (BERT paper recommends 2e-5/3e-5/5e-5)")
parser.add_argument("--tag", default="task3_bert_finetuned")
cli = parser.parse_args()  # named `cli`: `args` below is the TrainingArguments

MODEL = "google-bert/bert-base-uncased"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "bert-base-uncased")
# prefer the locally downloaded copy (offline reproducibility); fall back to HF
MODEL_PATH = LOCAL_MODEL_DIR if os.path.exists(
    os.path.join(LOCAL_MODEL_DIR, "model.safetensors")) else MODEL
MAX_LEN = 64
EPOCHS = 3
BATCH = 16
SEED = 42

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(LABELS))

label2id = {l: i for i, l in enumerate(LABELS)}

def to_ds(df):
    enc = tok(list(df["text"]), truncation=True, max_length=MAX_LEN)
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": [label2id[l] for l in df["label"]],
    })

ds_train, ds_val, ds_test = (to_ds(load_split(s)) for s in ("train", "val", "test"))

args = TrainingArguments(
    output_dir="models/bert_out",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=64,
    learning_rate=cli.lr,        # standard BERT fine-tune LR (arg)
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    logging_steps=50,
    save_strategy="no",          # keep only the final model in memory
    report_to="none",
    seed=SEED,
)

trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                  data_collator=DataCollatorWithPadding(tok))  # dynamic per-batch padding
trainer.train()

def predict(ds):
    out = trainer.predict(ds)
    return np.argmax(out.predictions, axis=1), out.label_ids

print("\n-- validation --")
val_pred, val_true = predict(ds_val)
evaluate([LABELS[i] for i in val_true], [LABELS[i] for i in val_pred], LABELS)

print("-- test (reported) --")
test_pred, test_true = predict(ds_test)
metrics = evaluate([LABELS[i] for i in test_true], [LABELS[i] for i in test_pred], LABELS)
save_result(f"{cli.tag}_lr{cli.lr:g}", metrics)

trainer.save_model("models/bert_final")
tok.save_pretrained("models/bert_final")
