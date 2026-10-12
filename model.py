"""
Machine Learning Pipeline: Iris Classification with Logistic Regression
Loads the Iris dataset, trains a Logistic Regression classifier,
evaluates accuracy, prints results, and exports metrics for CI/CD deployment.
"""

import os
import json
import re
from datetime import datetime, timezone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(test_size=0.2, random_state=42):
    """
    Load the Iris dataset and split it into training and testing sets.
    
    Returns:
        X_train, X_test, y_train, y_test, target_names, feature_names
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_names, feature_names


def train_model(X_train, y_train, max_iter=200, random_state=42):
    """
    Train a Logistic Regression model on the training data.
    
    Returns:
        fitted LogisticRegression model instance
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Generate predictions and calculate accuracy on test data.
    
    Returns:
        accuracy (float), predictions (ndarray)
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return float(accuracy), predictions


def save_metrics(accuracy, output_file="metrics.json", html_file="index.html"):
    """
    Save execution metrics to JSON and dynamically update index.html.
    
    Returns:
        dict containing metrics
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    accuracy_percent = f"{accuracy * 100:.2f}%"

    metrics = {
        "model_name": "Logistic Regression",
        "dataset_name": "Iris Dataset",
        "accuracy": round(accuracy, 4),
        "accuracy_percent": accuracy_percent,
        "ci_cd_status": "Successful",
        "updated_at": now_utc
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved to {output_file}")

    # Inject actual values into index.html if it exists
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Update placeholder tokens or existing tags
        content = re.sub(r'id="model-accuracy">[^<]*<', f'id="model-accuracy">{accuracy_percent}<', content)
        content = re.sub(r'id="ci-cd-status">[^<]*<', f'id="ci-cd-status">Successful<', content)
        content = re.sub(r'id="last-updated">[^<]*<', f'id="last-updated">{now_utc}<', content)
        content = content.replace("{{ACCURACY}}", accuracy_percent)
        content = content.replace("{{STATUS}}", "Successful")
        content = content.replace("{{UPDATED_AT}}", now_utc)

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[INFO] Injected measured accuracy ({accuracy_percent}) into {html_file}")

    return metrics


def run_pipeline():
    """
    Execute the entire ML pipeline from data loading to evaluation and export.
    
    Returns:
        accuracy, model
    """
    print("=" * 60)
    print(" MACHINE LEARNING TRAINING PIPELINE: LOGISTIC REGRESSION ")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test, target_names, feature_names = load_data()
    total_samples = len(X_train) + len(X_test)
    print(f"Dataset Name:       Iris Dataset")
    print(f"Total Samples:      {total_samples} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Target Classes:     {', '.join(target_names)}")
    print(f"Features:           {', '.join(feature_names)}")
    print("-" * 60)

    # 2. Train model
    print("Training Logistic Regression Model...")
    model = train_model(X_train, y_train)
    print("Model Training:     Completed successfully")
    print(f"Iterations:         {model.n_iter_[0]}")
    print("-" * 60)

    # 3. Evaluate model
    print("Generating Predictions and Calculating Metrics...")
    accuracy, predictions = evaluate_model(model, X_test, y_test)
    print(f"Actual Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("-" * 60)

    # 4. Save metrics and update dashboard
    save_metrics(accuracy)
    print("=" * 60)
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 60)

    return accuracy, model


if __name__ == "__main__":
    run_pipeline()
