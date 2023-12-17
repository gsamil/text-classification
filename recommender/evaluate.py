import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from data import (
    vocab,
    get_samples_from_file,
    load_categories,
)
from model import TextClassifier, device, TrainingParameters, HyperParameters
import time
from recommender.dataset import ClassificationDataset
from torch.utils.data import DataLoader
from recommender.train import model_dir, test_file
import time

# Set `test_file` and `model_dir` apropriately (in the train.py file).
# run with `export PYTHONPATH=. && python classifier/evaluate.py` in the main directory.

model: TextClassifier
training_params: TrainingParameters

model, training_params = TextClassifier.load_from_dir(model_dir, device)
model.eval()

categories, category_to_idx = load_categories()

test_samples = get_samples_from_file(test_file)

hparams: HyperParameters = model.hyper_parameters

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
    f"Number of Test Samples: {len(test_dataset)}\n",
    f"Number of Test Steps: {len(test_dataset) / training_params.batch_size}\n",
)

test_loader = DataLoader(
    test_dataset, batch_size=training_params.batch_size, shuffle=False
)

eval_start_time = time.time()

with torch.no_grad():
    total_test_samples = 0
    correct_predictions = 0

    predicted_labels_list = []
    targets_list = []

    start_time = time.time()

    for inputs, labels, batch_categories in test_loader:
        inputs = inputs.view(-1, training_params.feature_size).to(device)
        batch_categories = batch_categories.to(device)
        outputs = model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

        _, predicted_labels = torch.max(
            outputs.view(training_params.batch_size, -1),
            dim=1,
        )

        total_test_samples += len(batch_categories)
        correct_predictions += (predicted_labels == batch_categories).sum().item()

        predicted_labels_list.extend(predicted_labels.tolist())
        targets_list.extend(batch_categories.tolist())

        time_to_print = training_params.batch_size * training_params.step_size

        if total_test_samples % time_to_print == 0:
            print(
                f"Step {total_test_samples / time_to_print}/{len(test_dataset) / time_to_print},",
                f"({time.time() - start_time:.2f} seconds)",
            )
            start_time = time.time()

    # Calculate evaluation metrics
    accuracy = correct_predictions / total_test_samples
    precision = precision_score(targets_list, predicted_labels_list, average="weighted")
    recall = recall_score(targets_list, predicted_labels_list, average="weighted")
    f1 = f1_score(targets_list, predicted_labels_list, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

eval_end_time = time.time()
print(f"Evaluation took {eval_end_time - eval_start_time:.2f} seconds")
