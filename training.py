import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def load_and_clean_data(path='titanic.csv'):
    df = pd.read_csv(path)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def build_preprocessor(categorical, numerical):
    return ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), categorical),
            ('numerical', StandardScaler(), numerical)
        ]
    )


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    df = load_and_clean_data()

    categorical = ['Sex', 'Embarked', 'Pclass']
    numerical = ['Age', 'SibSp', 'Parch', 'Fare']
    model_RF = RandomForestClassifier()

    preprocessor = build_preprocessor(categorical, numerical)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model_RF)])

    X = df.drop(columns=['Survived'])
    y = df['Survived']

    param_grid_RF = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", "log2"],
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(pipeline, param_grid_RF, cv=5, scoring="accuracy", n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)
    joblib.dump(best_model, 'best_model.pkl')
    return metrics


if __name__ == "__main__":
    main()