"""Task 2a: pre-trained GloVe (6B, 100d) -> mean-pooled doc vectors -> LR.

Expects data/glove.6B.100d.txt (from the official glove.6B.zip).
Only the 100d file is loaded; words are looked up as-is (all lowercase).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from common import (load_split, tokenize, evaluate, save_result,
                    doc_vectors_mean, DATA_DIR, RANDOM_SEED)

def load_glove(path: str, dim: int = 100) -> dict:
    print(f"loading GloVe from {path} ...")
    emb = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            emb[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    assert all(v.shape == (dim,) for v in list(emb.values())[:100])
    print(f"GloVe vocabulary: {len(emb)} words x {dim} dims")
    return emb

glove = load_glove(f"{DATA_DIR}/glove.6B.100d.txt")

train, val, test = load_split("train"), load_split("val"), load_split("test")
train_tokens = [tokenize(t) for t in train["text"]]
val_tokens = [tokenize(t) for t in val["text"]]
test_tokens = [tokenize(t) for t in test["text"]]

X_train = doc_vectors_mean(train_tokens, glove, 100)
X_val = doc_vectors_mean(val_tokens, glove, 100)
X_test = doc_vectors_mean(test_tokens, glove, 100)
n_zero = int((abs(X_test).sum(axis=1) == 0).sum())
print(f"test docs with no in-vocab word (zero vector): {n_zero}")

clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
clf.fit(X_train, train["label"])

print("-- validation --")
print(f"val acc: {clf.score(X_val, val['label']):.4f}")
print("-- test (reported) --")
metrics = evaluate(test["label"], clf.predict(X_test))
save_result("task2a_glove_pretrained", metrics)
