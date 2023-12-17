import torch
from torch.utils.data import Dataset
from data import (
    set_feature_dimension,
    ClassificationSample,
)
from typing import Tuple


class ClassificationDataset(Dataset):
    def __init__(
        self,
        data: list[ClassificationSample],
        vocab: dict[str, int],
        category_to_idx: dict[str, int],
        feature_size: int,
    ):
        self.data: list[ClassificationSample] = data
        self.vocab: dict[str, int] = vocab
        self.feature_size: int = feature_size
        self.category_to_idx: dict[str, int] = category_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        example = self.data[idx]
        return (
            torch.LongTensor(
                set_feature_dimension(
                    [self.vocab[token] for token in example.product_text],
                    self.feature_size,
                )
            ),
            self.category_to_idx[example.category],
        )
