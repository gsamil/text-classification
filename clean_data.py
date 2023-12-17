import pandas as pd
from data import preprocess_text

# Use this to preprocess the text and saves the cleaned data to the given out file path
# run with `export PYTHONPATH=. && python clean_data.py` in the same directory.

file_path = "./data/test.csv"
out_file_path = "./data/test_cleaned.csv"

dtypes = {"product_id": str, "product_text": str, "category": str}
df = pd.read_csv(file_path, sep=chr(1), dtype=dtypes).dropna(
    subset=["category"], inplace=False
)
samples = []
for _, row in df.iterrows():
    product_text = preprocess_text(row["product_text"])
    if product_text == "":
        continue
    category = preprocess_text(row["category"])
    sample = {
        "product_id": row["product_id"],
        "product_text": product_text,
        "category": category,
    }
    samples.append(sample)
df = pd.DataFrame.from_dict(samples)
df.to_csv(out_file_path, index=False, sep=chr(1))
