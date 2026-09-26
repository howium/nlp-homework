"""Step 0: create the fixed 80/10/10 NYT split used by ALL experiments.

Procedure required by the assignment:
  1. shuffle the whole NYT dataset randomly (fixed seed -> reproducible)
  2. split: 80% train / 10% val / 10% test
The three CSVs are written to data/ so every task loads identical splits.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_DIR, RANDOM_SEED, LABELS, RAW_NYT

df = pd.read_csv(RAW_NYT)
print(f"Loaded NYT: {len(df)} rows, columns = {list(df.columns)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# 1) shuffle once with the shared fixed seed
df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# 2) 80 / 10 / 10: first take 20% out (val+test), then split that in half
train_df, rest_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
)
val_df, test_df = train_test_split(
    rest_df, test_size=0.5, random_state=RANDOM_SEED, stratify=rest_df["label"]
)

# stratify keeps the class ratio of this imbalanced dataset the same in
# every split, so e.g. the test set is not accidentally missing 'business'
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    part.to_csv(f"{DATA_DIR}/nyt_{name}.csv", index=False)
    print(f"{name:5s}: {len(part):5d} rows | {dict(part['label'].value_counts())}")

assert LABELS == sorted(df["label"].unique())
print("\nSaved data/nyt_train.csv, nyt_val.csv, nyt_test.csv")
