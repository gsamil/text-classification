from torch import nn
from pydantic import BaseModel
import torch
import os
import json
from typing import Union, Any, Tuple
from settings import DEVICE


device = torch.device(DEVICE)


class HyperParameters(BaseModel):
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_classes: int

    def to_dict(self) -> dict[Any, Any]:
        return self.__dict__

    @classmethod
    def load_from_json(cls, file_path: str) -> "HyperParameters":
        with open(file_path, "rt", encoding="utf-8") as f:
            parameters = json.load(f)
        return cls(**parameters)

    def print(self):
        print(
            f"Vocab Size: {self.vocab_size}\n",
            f"Embedding Dim: {self.embedding_dim}\n",
            f"Hidden Dim: {self.hidden_dim}\n",
            f"Num Classes: {self.num_classes}\n",
        )


class TrainingParameters(BaseModel):
    feature_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    step_size: int
    stratify_size: int | None

    def to_dict(self) -> dict[Any, Any]:
        return self.__dict__

    @classmethod
    def load_from_json(cls, file_path: str) -> "TrainingParameters":
        with open(file_path, "rt", encoding="utf-8") as f:
            parameters = json.load(f)
        return cls(**parameters)

    def print(self):
        print(
            f"Feature Size: {self.feature_size}\n",
            f"Num Epochs: {self.num_epochs}\n",
            f"Batch Size: {self.batch_size}\n",
            f"Learning Rate: {self.learning_rate}\n",
            f"Step Size: {self.step_size}\n",
            f"Stratify Size: {self.stratify_size}\n",
        )


class TextClassifier(nn.Module):
    def __init__(self, hparameters: HyperParameters):
        super(TextClassifier, self).__init__()
        self.hyper_parameters = hparameters
        self.embedding = nn.Embedding(hparameters.vocab_size, hparameters.embedding_dim)
        self.rnn = nn.LSTM(
            hparameters.embedding_dim, hparameters.hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hparameters.hidden_dim, hparameters.num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits

    def save_model(
        self, model_dir: str, training_parameters: TrainingParameters
    ) -> None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), os.path.join(model_dir, "saved_weights.pt"))
        with open(
            os.path.join(model_dir, "model_parameters.json"), "wt", encoding="utf-8"
        ) as f:
            json.dump(self.hyper_parameters.to_dict(), f, indent=4)
        with open(
            os.path.join(model_dir, "training_parameters.json"), "wt", encoding="utf-8"
        ) as f:
            json.dump(training_parameters.to_dict(), f, indent=4)

        print(f"Saved model to {model_dir}")

    @classmethod
    def load_from_dir(
        cls, model_dir: str, device: Union[str, torch.device]
    ) -> Tuple["TextClassifier", TrainingParameters]:
        model_parameters = HyperParameters.load_from_json(
            os.path.join(model_dir, "model_parameters.json")
        )
        model = cls(hparameters=model_parameters).to(device)
        state_dict = torch.load(
            os.path.join(model_dir, "saved_weights.pt"),
            map_location=torch.device(device),
        )
        model.load_state_dict(state_dict)
        training_parameters = TrainingParameters.load_from_json(
            os.path.join(model_dir, "training_parameters.json")
        )

        print(f"Loaded model from {model_dir}")
        return model, training_parameters
