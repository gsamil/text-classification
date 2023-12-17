import pandas as pd
from data import get_samples_from_file, stratify_samples, load_categories

# Use this to create a stratified sub-sample of the training data.
# run with `export PYTHONPATH=. && python save_subsample.py` in the same directory.

file_path = "./data/train_cleaned.csv"
out_file_path = "./data/train_cleaned_all_100.csv"
stratify_size = 100

samples = get_samples_from_file(file_path)
samples = stratify_samples(samples, stratify_size)
categories, _ = load_categories()
samples = [sample for sample in samples if sample.category in categories]
df = pd.DataFrame.from_dict([sample.model_dump() for sample in samples])
df.to_csv(out_file_path, index=False, sep=chr(1))
