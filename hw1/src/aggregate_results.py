"""Aggregate results/results.csv into a Markdown table for the report."""
import pandas as pd

df = pd.read_csv("results/results.csv").drop_duplicates(subset="experiment", keep="last")
order = [
    "task1_binary_bow", "task1_word_frequency",
    "task2a_glove_pretrained", "task2b_word2vec_ag", "task2c_word2vec_nyt",
    "task3_bert_finetuned", "task3_bert_lr3e5",
]
df["order"] = df["experiment"].map({e: i for i, e in enumerate(order)})
df = df.sort_values("order")

pretty = {
    "task1_binary_bow": "Task1: Binary BoW + LR",
    "task1_word_frequency": "Task1: Word Frequency + LR",
    "task2a_glove_pretrained": "Task2a: GloVe 6B 100d (mean) + LR",
    "task2b_word2vec_ag": "Task2b: Word2Vec on AG News (mean) + LR",
    "task2c_word2vec_nyt": "Task2c: Word2Vec on NYT (mean) + LR",
    "task3_bert_finetuned": "Task3: BERT fine-tuned (lr=2e-5)",
    "task3_bert_lr3e5": "Task3: BERT fine-tuned (lr=3e-5)",
}
df["experiment"] = df["experiment"].map(pretty)
out = df[["experiment", "accuracy", "macro_f1"]].to_markdown(
    index=False, floatfmt=".4f")
print(out)
with open("results/summary.md", "w", encoding="utf-8") as f:
    f.write(out + "\n")
