# Text Classification

## Proble Description

We are used to working on text classification problem. It's generally very straightforward. However, there are cases where classes also contain some information and we may feel like we should also use this information. For example, we may want to understand user intent in a chatbot, or we may want to categorize a product given its description. In these cases, categories also have semantic meaning, i.e. some categories are related to each other, so we can also create embedding vectors for the categories.

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

## References

- [Text Classification Using Class Information](https://www.abdullahsamilguser.com/blog/text-classification/)
