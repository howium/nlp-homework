"""Shared utilities for HW1: data loading, tokenization, evaluation.

All experiment scripts import from here so that every task uses
exactly the same data split and the same evaluation procedure.
"""
import os
import re
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, classification_report

# hw1 root = parent of src/, so scripts work no matter where you run them
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

RANDOM_SEED = 42
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_NYT = os.path.join(ROOT, "data", "raw", "nyt.csv")
RAW_AG = os.path.join(ROOT, "data", "raw", "ag.csv")

LABELS = ["business", "politics", "sports"]  # fixed order for consistent encoding


def load_split(name: str) -> pd.DataFrame:
    """Load one of the pre-split NYT files: 'train' | 'val' | 'test'."""
    return pd.read_csv(f"{DATA_DIR}/nyt_{name}.csv")


def tokenize(text: str) -> list[str]:
    """Lowercase + NLTK word tokenization, keep alphabetic tokens only.

    GloVe/Word2Vec lookup works better on clean lowercase words; dropping
    pure numbers/punctuation also removes tokens that have no embedding.
    """
    return [w for w in word_tokenize(text.lower()) if re.fullmatch(r"[a-z]+", w)]


def evaluate(y_true, y_pred, label_names=None, verbose: bool = True) -> dict:
    """Compute Accuracy and Macro-F1 (the two required metrics)."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    if verbose:
        print(f"Accuracy  : {acc:.4f}")
        print(f"Macro-F1  : {macro_f1:.4f}")
        if label_names is not None:
            print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    return {"accuracy": acc, "macro_f1": macro_f1}


def doc_vectors_mean(tokens_list, embedding_lookup, dim: int) -> np.ndarray:
    """Average all in-vocabulary word vectors of a document (Task 2 formula).

    Documents with no in-vocabulary words get a zero vector.
    """
    X = np.zeros((len(tokens_list), dim), dtype=np.float32)
    for i, toks in enumerate(tokens_list):
        vecs = [embedding_lookup[w] for w in toks if w in embedding_lookup]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return X


def save_result(name: str, metrics: dict) -> None:
    """Append one experiment's metrics to results/results.csv."""
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/results.csv"
    row = pd.DataFrame([{"experiment": name, **metrics}])
    header = not os.path.exists(path)
    row.to_csv(path, mode="a", header=header, index=False)
    print(f"[saved] {name}: acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
