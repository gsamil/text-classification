import torch
from torch.utils.data import Dataset
from data import (
    ClassificationSample,
    set_feature_dimension,
)
from typing import Tuple
import random


class ClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[ClassificationSample],
        vocab: dict[str, int],
        categories: list[str],
        category_to_idx: dict[str, int],
        feature_size: int,
        sample_negatives: int | None,
        shuffle: bool,
    ):
        self.data: list[ClassificationSample] = samples
        self.vocab: dict[str, int] = vocab
        self.categories: list[str] = categories
        self.category_to_idx: dict[str, int] = category_to_idx
        self.feature_size: int = feature_size
        self.sample_negatives: int | None = sample_negatives
        self.shuffle: bool = shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, list[int], int]:
        example = self.data[idx]
        combined_features_positive = []
        combined_features_negative = []
        for category in self.categories:
            product_text_tokens = [self.vocab[token] for token in example.product_text]
            category_tokens = [self.vocab[token] for token in category]
            product_text_token_indexes = set_feature_dimension(
                product_text_tokens,
                self.feature_size - len(category_tokens),
            )
            token_indexes = product_text_token_indexes + category_tokens

            if example.category == category:
                combined_features_positive.append(token_indexes)
            else:
                combined_features_negative.append(token_indexes)

        if self.sample_negatives is not None:
            combined_features_negative = random.sample(
                combined_features_negative, self.sample_negatives
            )
        combined_features_with_labels = [
            (feature, 1) for feature in combined_features_positive
        ] + [(feature, 0) for feature in combined_features_negative]

        if self.shuffle:
            random.shuffle(combined_features_with_labels)
        return (
            torch.LongTensor([f for f, _ in combined_features_with_labels]),
            [l for _, l in combined_features_with_labels],
            [i for i, (_, l) in enumerate(combined_features_with_labels) if l == 1][0],
        )
