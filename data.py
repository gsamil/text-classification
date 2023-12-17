from typing import Tuple, Tuple
from pydantic import BaseModel
import pandas as pd
from collections import Counter
import random
from settings import CATEGORIES_PATH


random_seed = 42
random.seed(random_seed)


vocab: dict[str, int] = {
    " ": 0,
    "a": 1,
    "e": 2,
    "k": 3,
    "l": 4,
    "i": 5,
    "r": 6,
    "t": 7,
    "o": 8,
    "s": 9,
    "n": 10,
    "ı": 11,
    "m": 12,
    "u": 13,
    "d": 14,
    "y": 15,
    "b": 16,
    "p": 17,
    "ü": 18,
    "0": 19,
    "c": 20,
    "z": 21,
    "h": 22,
    "g": 23,
    "f": 24,
    "v": 25,
    "ş": 26,
    "ç": 27,
    "1": 28,
    "2": 29,
    "5": 30,
    "-": 31,
    "3": 32,
    "4": 33,
    "j": 34,
    "ö": 35,
    "ğ": 36,
    "6": 37,
    "x": 38,
    "7": 39,
    "9": 40,
    "8": 41,
    "w": 42,
    "&": 43,
    "é": 44,
    "â": 45,
}


class ClassificationSample(BaseModel):
    product_id: str
    product_text: str
    category: str


def preprocess_text(text: str) -> str:
    return "".join([c for c in text.lower() if c in vocab])


def get_samples_from_file(file_path: str) -> list[ClassificationSample]:
    """This function reads the data from the given file path and returns the samples"""
    dtypes = {"product_id": str, "product_text": str, "category": str}
    df = pd.read_csv(file_path, sep=chr(1), dtype=dtypes)
    samples = []
    for _, row in df.iterrows():
        sample = ClassificationSample(
            product_id=row["product_id"],
            product_text=str(row["product_text"]),
            category=str(row["category"]),
        )
        samples.append(sample)
    return samples


def stratify_samples(
    samples: list[ClassificationSample], number_per_sample: int
) -> list[ClassificationSample]:
    """This function selects the given number of samples for each category and returns the stratified samples"""
    category_counts = Counter([s.category for s in samples])
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )
    category_to_samples: dict[str, list[ClassificationSample]] = {}
    for sample in samples:
        category_to_samples[sample.category] = category_to_samples.get(
            sample.category, []
        )
        category_to_samples[sample.category].append(sample)

    stratified_samples: list[ClassificationSample] = []
    for category, count in sorted_categories:
        stratified_samples.extend(
            random.sample(category_to_samples[category], number_per_sample)
            if len(category_to_samples[category]) > number_per_sample
            else category_to_samples[category]
        )

    return stratified_samples


def save_categories(train_file: str) -> None:
    samples = get_samples_from_file(train_file)
    category_counts = Counter([s.category for s in samples])
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )
    categories = [c for c, i in sorted_categories]
    pd.DataFrame(categories, columns=["category"]).to_csv(CATEGORIES_PATH, index=False)


def load_categories() -> Tuple[list[str], dict[str, int]]:
    categories = pd.read_csv(CATEGORIES_PATH)["category"].tolist()
    category_to_idx = {c: i for i, c in enumerate(categories)}
    return categories, category_to_idx


def set_feature_dimension(lst: list[int], target_length: int):
    """This function sets the feature dimension of the given list to the given target length by prepending zeros or truncating the list"""
    current_length = len(lst)
    if current_length > target_length:
        lst = lst[:target_length]
    elif current_length < target_length:
        num_zeros_to_prepend = target_length - current_length
        zeros_to_prepend = [0] * num_zeros_to_prepend
        lst = zeros_to_prepend + lst
    return lst


def print_text_lengths(train_file: str) -> None:
    # print the text length vs number of samples with this text length sorted by text length (for classfication model)
    samples = get_samples_from_file(train_file)
    text_lengths = Counter([len(s.product_text) for s in samples])
    sorted_text_lengths = sorted(text_lengths.items(), key=lambda x: x[0])
    print("text_length\tnumber_of_samples")
    for text_length, count in sorted_text_lengths:
        print(f"{text_length}\t{count}")

    # print the text length with category vs number of samples with this text length with category sorted by text length with category (for recommendation model)
    text_lengths_with_categories = Counter(
        [len(f"{s.product_text}{s.category}") for s in samples]
    )
    sorted_text_lengths_with_categories = sorted(
        text_lengths_with_categories.items(), key=lambda x: x[0]
    )
    print("text_length_with_category\tnumber_of_samples")
    for text_length, count in sorted_text_lengths_with_categories:
        print(f"{text_length}\t{count}")
