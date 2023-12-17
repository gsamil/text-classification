from data import get_samples_from_file
import pandas as pd


train_file = "./data/train_cleaned.csv"

samples = get_samples_from_file(train_file)
counts_1gram: dict[str, int] = {}
counts_2gram: dict[str, int] = {}
counts_3gram: dict[str, int] = {}
for sample in samples:
    for word in sample.product_text.split(" "):
        counts_1gram[word] = counts_1gram.get(word, 0) + 1
    for word_2gram in zip(
        sample.product_text.split(" ")[:-1], sample.product_text.split(" ")[1:]
    ):
        counts_2gram[word_2gram] = counts_2gram.get(word_2gram, 0) + 1
    for word_3gram in zip(
        sample.product_text.split(" ")[:-2],
        sample.product_text.split(" ")[1:-1],
        sample.product_text.split(" ")[2:],
    ):
        counts_3gram[word_3gram] = counts_3gram.get(word_3gram, 0) + 1

sorted_counts_1gram = sorted(counts_1gram.items(), key=lambda x: x[1], reverse=True)
sorted_counts_2gram = sorted(counts_2gram.items(), key=lambda x: x[1], reverse=True)
sorted_counts_3gram = sorted(counts_3gram.items(), key=lambda x: x[1], reverse=True)

pd.DataFrame(sorted_counts_1gram, columns=["word", "count"]).to_csv(
    "./1gram.csv", index=False
)
pd.DataFrame(sorted_counts_2gram, columns=["word", "count"]).to_csv(
    "./2gram.csv", index=False
)
pd.DataFrame(sorted_counts_3gram, columns=["word", "count"]).to_csv(
    "./3gram.csv", index=False
)