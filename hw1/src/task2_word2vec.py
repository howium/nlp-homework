"""Task 2 (b, c): train 100-d Word2Vec, average-pool to doc vectors, LR.

  2b: Word2Vec trained on AG News corpus (90k unlabeled news texts)
  2c: Word2Vec trained on NYT *training split* only
      (keeps val/test out of everything, strictest protocol; noted in report)

Both models: vector_size=100 (required), window=5, min_count=5, CBOW (sg=0),
5 epochs. Document representation = mean of the in-vocabulary word vectors.
"""
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from common import (load_split, tokenize, evaluate, save_result,
                    doc_vectors_mean, RAW_AG, RANDOM_SEED)

train, val, test = load_split("train"), load_split("val"), load_split("test")
train_tokens = [tokenize(t) for t in train["text"]]
val_tokens = [tokenize(t) for t in val["text"]]
test_tokens = [tokenize(t) for t in test["text"]]
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

corpora = {
    "task2b_word2vec_ag": [tokenize(t) for t in pd.read_csv(RAW_AG)["text"]],
    "task2c_word2vec_nyt": train_tokens,
}

for name, corpus in corpora.items():
    print(f"\n===== {name} =====")
    print(f"training Word2Vec on {len(corpus)} documents ...")
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=100,   # required: 100-dimensional word vectors
        window=5,
        min_count=5,       # ignore words appearing < 5 times
        workers=4,
        sg=0,              # CBOW
        epochs=5,
        seed=RANDOM_SEED,
    )
    w2v.save(f"models/{name}.model")
    print(f"vocab size: {len(w2v.wv)}")

    lookup = w2v.wv  # dict-like: word -> 100d vector
    X_train = doc_vectors_mean(train_tokens, lookup, 100)
    X_val = doc_vectors_mean(val_tokens, lookup, 100)
    X_test = doc_vectors_mean(test_tokens, lookup, 100)
    n_zero = int((abs(X_test).sum(axis=1) == 0).sum())
    print(f"test docs with no in-vocab word (zero vector): {n_zero}")

    clf.fit(X_train, train["label"])
    print("-- validation --")
    evaluate(val["label"], clf.predict(X_val), None, verbose=False)
    print(f"val acc: {clf.score(X_val, val['label']):.4f}")
    print("-- test (reported) --")
    metrics = evaluate(test["label"], clf.predict(X_test))
    save_result(name, metrics)
