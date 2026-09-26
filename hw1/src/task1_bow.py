"""Task 1: Bag-of-Words representations + Logistic Regression.

Two representations (both built with a vocabulary learned from the
TRAINING corpus only):
  1. Binary BoW   -> x_i = 1 if word i occurs in the document else 0
  2. Word Frequency -> x_i = number of occurrences of word i

CountVectorizer implements exactly these two matrices (binary=True/False);
we hand it our nltk-based tokenizer from common.py so tokenization matches
the assignment's suggestion. LogisticRegression is the fixed classifier.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from common import (load_split, tokenize, evaluate, save_result,
                    LABELS, RANDOM_SEED)

train, val, test = load_split("train"), load_split("val"), load_split("test")

# tokenizer=tokenize: use the same nltk tokenization as the assignment
# suggests; lowercase is already done inside tokenize().
vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, binary=False)

# ---- representation 2: Word Frequency (raw counts) ----
X_train_freq = vectorizer.fit_transform(train["text"])
X_val_freq = vectorizer.transform(val["text"])
X_test_freq = vectorizer.transform(test["text"])
print(f"Vocabulary size |V| (from training corpus): {len(vectorizer.vocabulary_)}")

# ---- representation 1: Binary BoW (same vocabulary, 0/1 values) ----
# Re-fitting with binary=True on identical data gives the identical
# vocabulary, so we build it separately for clarity.
vectorizer_bin = CountVectorizer(tokenizer=tokenize, lowercase=False, binary=True)
X_train_bin = vectorizer_bin.fit_transform(train["text"])
X_val_bin = vectorizer_bin.transform(val["text"])
X_test_bin = vectorizer_bin.transform(test["text"])
assert vectorizer_bin.vocabulary_ == vectorizer.vocabulary_

clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

for name, (Xtr, Xva, Xte) in {
    "task1_binary_bow": (X_train_bin, X_val_bin, X_test_bin),
    "task1_word_frequency": (X_train_freq, X_val_freq, X_test_freq),
}.items():
    print(f"\n===== {name} =====")
    clf.fit(Xtr, train["label"])
    print("-- validation --")
    evaluate(val["label"], clf.predict(Xva), LABELS)
    print("-- test (reported) --")
    metrics = evaluate(test["label"], clf.predict(Xte), LABELS)
    save_result(name, metrics)
