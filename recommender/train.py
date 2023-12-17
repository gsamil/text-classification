import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
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

train_file = "./data/train_cleaned.csv"
test_file = "./data/test_cleaned_all_100.csv"
model_dir = "./recommender/saved_model"

if __name__ == "__main__":
    hparams = HyperParameters(
        vocab_size=len(vocab),
        embedding_dim=8,
        hidden_dim=128,
        num_classes=2,
    )

    # Training parameters
    training_params = TrainingParameters(
        feature_size=100,
        batch_size=1,
        num_epochs=3,
        learning_rate=0.001,
        step_size=1000,
        stratify_size=500,
    )

    hparams.print()
    training_params.print()

    negative_samples = 49

    if not os.path.exists(CATEGORIES_PATH):
        save_categories(train_file)

    categories, category_to_idx = load_categories()

    train_samples = stratify_samples(
        get_samples_from_file(train_file), training_params.stratify_size
    )
    test_samples = get_samples_from_file(test_file)

    # Create data loaders for the training and validation sets
    train_dataset = ClassificationDataset(
        train_samples,
        vocab,
        categories,
        category_to_idx,
        training_params.feature_size,
        negative_samples,
        True,
    )
    test_dataset = ClassificationDataset(
        test_samples,
        vocab,
        categories,
        category_to_idx,
        training_params.feature_size,
        None,
        False,
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
    scheduler = ExponentialLR(optimizer, gamma=0.9)

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
        for inputs, labels, batch_categories in train_loader:
            optimizer.zero_grad()
            inputs = inputs.view(-1, training_params.feature_size).to(device)
            targets = torch.LongTensor(torch.cat(labels, dim=0)).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            current_samples = len(batch_categories) * (negative_samples + 1)

            total_loss += loss.item() * current_samples
            total_samples += current_samples

            time_to_print = (
                training_params.batch_size
                * (negative_samples + 1)
                * training_params.step_size
            )

            if total_samples % time_to_print == 0:
                print(
                    f"Epoch {epoch+1}/{training_params.num_epochs}, "
                    f"Step {total_samples / time_to_print}/{(len(train_dataset) * (negative_samples + 1)) / time_to_print}, "
                    f"Train Loss: {total_loss/total_samples:.6f}"
                )
        scheduler.step()
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} took {epoch_end_time - epoch_start_time:.2f} seconds")
        model.save_model(f"{model_dir}_{epoch+1}", training_params)

        if epoch != 0 and epoch % 2 == 0:
            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for inputs, labels, batch_categories in test_loader:
                    inputs = inputs.view(-1, training_params.feature_size).to(device)
                    targets = torch.LongTensor(torch.cat(labels, dim=0)).to(device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)

                    current_val_samples = len(batch_categories) * len(categories)

                    total_val_loss += val_loss.item() * current_val_samples
                    total_val_samples += current_val_samples

            avg_loss = total_loss / total_samples
            avg_val_loss = total_val_loss / total_val_samples

            print(
                f"Epoch {epoch+1}/{training_params.num_epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

    training_end_time = time.time()
    print(f"Training took {training_end_time - training_start_time:.2f} seconds")
