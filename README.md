# Part A

## Training Instructions

1. Set the device in [settings.py](./settings.py)
2. Make sure you have `categories.csv` file under the `./data` directory. If not, run:

```
export PYTHONPATH=. && python make_categories.py
```
3. Set `train_file`, `test_file` and `model_dir` apropriately. (In the corresponding `train.py` file)

4. Now you can train either classifier or recommender models using the following commands:

```
export PYTHONPATH=. && python classifier/train.py
```

or

```
export PYTHONPATH=. && python recommender/train.py
```

## Evaluation Instructions

1. Set `test_file` and `model_dir` apropriately. (In the corresponding `train.py` file)
2. Just run either one of these:

```
export PYTHONPATH=. && python classifier/evaluate.py
```

or

```
export PYTHONPATH=. && python recommender/evaluate.py
```
