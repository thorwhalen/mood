"""Learning sentiments from data."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union


def get_binary_y(y, target_y):
    """
    Convert a multi-class target array to a binary array for a specific target class.
    """
    binary_y = np.zeros(y.shape)
    binary_y[y == target_y] = 1
    return binary_y


@dataclass
class ModelStatistics:
    """Container for model performance statistics"""

    accuracy: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    roc_auc: float = None
    confusion_matrix: np.ndarray = None
    classification_report: str = None
    train_test_performed: bool = False
    training_time: float = None
    test_size: float = None
    random_state: int = None

    def __str__(self) -> str:
        """String representation of the statistics"""
        if not self.train_test_performed:
            return "No validation performed. Statistics unavailable."

        output = "\n=== Model Performance Statistics ===\n"
        output += f"Accuracy:  {self.accuracy:.4f}\n"
        output += f"Precision: {self.precision:.4f}\n"
        output += f"Recall:    {self.recall:.4f}\n"
        output += f"F1 Score:  {self.f1:.4f}\n"
        output += f"ROC AUC:   {self.roc_auc:.4f}\n"
        output += f"\nConfusion Matrix:\n{self.confusion_matrix}\n"
        output += f"\nDetailed Classification Report:\n{self.classification_report}\n"
        output += f"\nTest Size: {self.test_size}, Random State: {self.random_state}\n"
        output += f"Training Time: {self.training_time:.4f} seconds\n"
        return output


def fit_logistic_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    c_value: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    penalty: str = "l2",
    class_weight: Optional[Union[Dict, str]] = None,
    fit_intercept: bool = True,
    skip_validation: bool = False,
    verbose: bool = True,
) -> Tuple[LogisticRegression, ModelStatistics]:
    """
    Fits a logistic regression model on embedding data for binary classification without feature scaling.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The embedding vectors.
    y : array-like, shape (n_samples,)
        Binary target values (0/1 or True/False).
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split.
    c_value : float, default=1.0
        Inverse of regularization strength; smaller values specify stronger regularization.
    max_iter : int, default=1000
        Maximum number of iterations taken for the solvers to converge.
    solver : str, default='lbfgs'
        Algorithm to use in the optimization problem ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga').
    penalty : str, default='l2'
        Specify the norm used in the penalization ('l1', 'l2', 'elasticnet', None).
    class_weight : dict or 'balanced', default=None
        Weights associated with classes. If 'balanced', uses class frequencies.
    fit_intercept : bool, default=True
        Whether to fit an intercept term in the model.
    skip_validation : bool, default=False
        If True, skips train/test validation and fits model on all data.
    verbose : bool, default=True
        If True, prints progress and results.

    Returns:
    --------
    model : LogisticRegression
        Trained logistic regression model.
    stats : ModelStatistics
        Object containing performance statistics.
    """
    start_time = time.time()
    stats = ModelStatistics()
    stats.test_size = test_size
    stats.random_state = random_state

    # Convert boolean y to integer if needed
    if y.dtype == bool:
        y = y.astype(int)

    # Initialize the model
    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty,
        class_weight=class_weight,
        random_state=random_state,
        fit_intercept=fit_intercept,
    )

    if verbose:
        print(
            f"Initializing logistic regression with C={c_value}, solver='{solver}', penalty='{penalty}', fit_intercept={fit_intercept}"
        )

    # Perform train/test validation if requested
    if not skip_validation:
        if verbose:
            print(
                f"Performing train/test split with test_size={test_size}, random_state={random_state}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit the model on training data
        if verbose:
            print("Fitting model on training data...")

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        stats.accuracy = accuracy_score(y_test, y_pred)
        stats.precision = precision_score(y_test, y_pred)
        stats.recall = recall_score(y_test, y_pred)
        stats.f1 = f1_score(y_test, y_pred)
        stats.roc_auc = roc_auc_score(y_test, y_prob)
        stats.confusion_matrix = confusion_matrix(y_test, y_pred)
        stats.classification_report = classification_report(y_test, y_pred)
        stats.train_test_performed = True

        if verbose:
            print(f"Validation Results:")
            print(f"  Accuracy:  {stats.accuracy:.4f}")
            print(f"  Precision: {stats.precision:.4f}")
            print(f"  Recall:    {stats.recall:.4f}")
            print(f"  F1 Score:  {stats.f1:.4f}")
            print(f"  ROC AUC:   {stats.roc_auc:.4f}")
            print("\nNow fitting model on all data...")

    # Fit model on all data
    model.fit(X, y)
    stats.training_time = time.time() - start_time

    if verbose:
        print(f"Model fitting completed in {stats.training_time:.4f} seconds")
        if not skip_validation:
            print(stats)

    return model, stats


def predict_with_model(model, X):
    """
    Helper function to make predictions with a fitted model.

    Parameters:
    -----------
    model : LogisticRegression
        Trained logistic regression model.
    X : array-like
        Feature vectors to predict on.

    Returns:
    --------
    y_pred : array-like
        Class predictions (0 or 1).
    y_prob : array-like
        Probability estimates for class 1.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


# SentimentDacc and HugSentimentDacc remain unchanged as they don’t involve scaling
# Keeping them out of this revision for brevity, but they’re still part of your module
