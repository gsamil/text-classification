import pandas as pd
from collections import Counter
from settings import CATEGORIES_PATH
from data import get_samples_from_file

# This file creates the categories file from the training data.
# run with `export PYTHONPATH=. && python make_categories.py` in the same directory.

train_file = "./data/train_cleaned.csv"

samples = get_samples_from_file(train_file)
category_counts = Counter([s.category for s in samples])
sorted_categories = sorted(
    category_counts.items(), key=lambda x: x[1], reverse=True
)
categories = [c for c, i in sorted_categories]
pd.DataFrame(categories, columns=["category"]).to_csv(CATEGORIES_PATH, index=False)
