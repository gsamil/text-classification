# Part A

## Data Preparation

1. If you want to train on a subsample of data, first see [save_subsample.py](./save_subsample.py)
2. To preprocess data, see [clean_data.py](./clean_data.py)

## Training Instructions

1. Set the device in [settings.py](./settings.py)
2. Make sure you have `categories.csv` file under the `./data` directory. If not, see [make_categories.py](./make_categories.py)
3. Now you can train either classifier or recommender models.
    
    i. For classification, see [classification/train.py](./classifier/train.py)
    
    ii. For recommendation, see [recommender/train.py](./recommender/train.py)

## Evaluation Instructions

1. Set `test_file` and `model_dir` apropriately. (In the corresponding `train.py` file)
2. Now you can train either classifier or recommender models.
    
    i. For classification, see [classification/evaluate.py](./classifier/evaluate.py)
    
    ii. For recommendation, see [recommender/evaluate.py](./recommender/evaluate.py)
