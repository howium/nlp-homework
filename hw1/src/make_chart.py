"""Regenerate the methods-comparison chart (Figure 1) for the report.

Reads metrics from results/results.csv and renders a horizontal bar chart
to hw1/report/latex/chart_methods.png, which main.tex includes.
Run from hw1/ after re-running experiments if the numbers change.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR, ROOT

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PRETTY = {
    "task1_binary_bow": "Binary BoW\n+ LR",
    "task1_word_frequency": "Word Frequency\n+ LR",
    "task2a_glove_pretrained": "GloVe 预训练\n+ LR",
    "task2b_word2vec_ag": "W2V-AG News\n+ LR",
    "task2c_word2vec_nyt": "W2V-NYT\n+ LR",
    "task3_bert_finetuned": "BERT 微调\n(lr=2e-5)",
    "task3_bert_lr3e5": "BERT 微调\n(lr=3e-5)",
}
ORDER = list(PRETTY)  # worst at top, best at bottom

df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))
df = df.drop_duplicates(subset="experiment", keep="last").set_index("experiment")
df = df.loc[[e for e in ORDER if e in df.index]]

names = [PRETTY[e] for e in df.index]
acc, f1 = df["accuracy"].tolist(), df["macro_f1"].tolist()

fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
y = range(len(names))
bh = 0.38
b1 = ax.barh([i + bh / 2 + 0.02 for i in y], acc, height=bh,
             color="#8c7f5b", label="Accuracy")
b2 = ax.barh([i - bh / 2 - 0.02 for i in y], f1, height=bh,
             color="#572cd9", alpha=0.75, label="Macro-F1")
for bars in (b1, b2):
    for rect in bars:
        wv = rect.get_width()
        ax.text(wv + 0.0008, rect.get_y() + rect.get_height() / 2,
                f"{wv:.4f}", va="center", ha="left",
                fontsize=7.2, color="#191917")
ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlim(0.94, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.2)
ax.set_axisbelow(True)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
          frameon=False, fontsize=9)
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()

out = os.path.join(ROOT, "report", "latex", "chart_methods.png")
plt.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"chart saved -> {out}")
