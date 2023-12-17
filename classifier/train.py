import torch
from torch import nn
from torch.utils.data import DataLoader
from data import (
    vocab,
    get_samples_from_file,
    stratify_samples,
    save_categories,
    load_categories,
)
from model import TextClassifier, TrainingParameters, device, HyperParameters
import time
from recommender.dataset import ClassificationDataset
from settings import CATEGORIES_PATH
import os
from classifier.dataset import ClassificationDataset

# Set `train_file`, `test_file` and `model_dir` apropriately.
# run with `export PYTHONPATH=. && python classifier/train.py` in the main directory.


train_file = "./data/train_cleaned.csv"
test_file = "./data/test_cleaned_all_100.csv"
model_dir = "./classifier/saved_model"


if __name__ == "__main__":
    if not os.path.exists(CATEGORIES_PATH):
        save_categories(train_file)

    categories, category_to_idx = load_categories()

    hparams = HyperParameters(
        vocab_size=len(vocab),
        embedding_dim=8,
        hidden_dim=128,
        num_classes=len(categories),
    )

    # Training parameters
    training_params = TrainingParameters(
        feature_size=100,
        batch_size=50,
        num_epochs=3,
        learning_rate=0.002,
        step_size=1000,
        stratify_size=500,
    )

    hparams.print()
    training_params.print()

    train_samples = stratify_samples(
        get_samples_from_file(train_file), training_params.stratify_size
    )
    test_samples = get_samples_from_file(test_file)

    # Create data loaders for the training and validation sets
    train_dataset = ClassificationDataset(
        train_samples, vocab, category_to_idx, training_params.feature_size
    )
    test_dataset = ClassificationDataset(
        test_samples, vocab, category_to_idx, training_params.feature_size
    )

    print(
        f"Number of Training Samples: {len(train_dataset)}\n",
        f"Number of Test Samples: {len(test_dataset)}\n",
        f"Number of Training Steps: {len(train_dataset) / training_params.batch_size}\n",
        f"Number of Test Steps: {len(test_dataset) / training_params.batch_size}\n",
    )

    # Create the model
    model = TextClassifier(hparams).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params.learning_rate)

    train_loader = DataLoader(
        train_dataset, batch_size=training_params.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_params.batch_size, shuffle=False
    )

    training_start_time = time.time()

    # Iterate over the training data for the specified number of epochs
    for epoch in range(training_params.num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = torch.LongTensor(labels).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(inputs)
            total_samples += len(inputs)

            if (
                total_samples % (training_params.batch_size * training_params.step_size)
                == 0
            ):
                print(
                    f"Epoch {epoch+1}/{training_params.num_epochs}, "
                    f"Step {total_samples / (training_params.batch_size * training_params.step_size)}/{len(train_dataset) / (training_params.batch_size * training_params.step_size)}, "
                    f"Train Loss: {total_loss/total_samples:.6f}"
                )

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} took {epoch_end_time - epoch_start_time:.2f} seconds")
        model.save_model(f"{model_dir}_{epoch+1}", training_params)

        if epoch != 0 and epoch % 2 == 0:
            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    targets = torch.LongTensor(labels).to(device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)

                    total_val_loss += val_loss.item() * len(inputs)
                    total_val_samples += len(inputs)

            avg_loss = total_loss / total_samples
            avg_val_loss = total_val_loss / total_val_samples

            print(
                f"Epoch {epoch+1}/{training_params.num_epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

    training_end_time = time.time()
    print(f"Training took {training_end_time - training_start_time:.2f} seconds")
