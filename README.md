# TitanicML

Small machine learning project that predicts Titanic passenger survival using `scikit-learn`.

## What it does

- Loads `titanic.csv`
- Cleans data (drops unused columns, nulls, duplicates)
- Builds a preprocessing pipeline:
  - One-hot encodes categorical features (`Sex`, `Embarked`, `Pclass`)
  - Scales numerical features (`Age`, `SibSp`, `Parch`, `Fare`)
- Trains a model with `GridSearchCV` (currently `RandomForestClassifier`)
- Prints best hyperparameters and evaluation results
- Saves the trained model to `best_model.pkl`

## Project structure

- `training.py` - training script plus reusable preprocessing/evaluation functions
- `tests/test_training.py` - pytest tests for preprocessor and evaluation helpers
- `titanic.csv` - dataset

## Setup

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib pytest
```

## Run training

```bash
python training.py
```

## Run tests

```bash
python -m pytest -q
```

## Notes

- The model artifact `best_model.pkl` is ignored by `.gitignore`.
- `training.py` uses `if __name__ == "__main__":` so tests can import functions without triggering full training.
