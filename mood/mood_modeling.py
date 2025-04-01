"""
Mood Modeling Module

This module provides tools for modeling semantic attributes (or "moods") based on text embeddings.
It allows for different modeling approaches: numerical regression, ordinal regression, and binary classification.

The main class, MoodModelingManager, handles the entire process:
- Preparing different (X, y) pairs from input dataframes
- Training and evaluating multiple modeling pipelines
- Gathering comprehensive validation statistics
- Providing a final model that outputs mood scores in the [0, 1] range

These mood scores are ordinally aligned with training scores, where higher values indicate
stronger presence of the target semantic attribute.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, Iterable, Set
from scipy.special import expit
from scipy.stats import spearmanr, kendalltau
import warnings
from contextlib import suppress


def _sigmoid_transform(x: np.ndarray) -> np.ndarray:
    """
    Transform values using sigmoid function to range [0, 1].

    Args:
        x: Input array to transform

    Returns:
        Array with values transformed to [0, 1] range
    """
    return expit(x)


def _minmax_transform(x: np.ndarray) -> np.ndarray:
    """
    Scale values to [0, 1] range using min-max scaling.

    Args:
        x: Input array to transform

    Returns:
        Array with values scaled to [0, 1] range
    """
    if x.size == 0:
        return x

    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max:
        return np.full_like(x, 0.5)
    return (x - x_min) / (x_max - x_min)


def _tanh_transform(x: np.ndarray) -> np.ndarray:
    """
    Transform values using hyperbolic tangent function to range [0, 1].

    Args:
        x: Input array to transform

    Returns:
        Array with values transformed to [0, 1] range
    """
    return 0.5 * (np.tanh(x) + 1)


def x_and_y_with_boolean_y(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_is_true_vals: Union[List, str],
    y_is_false_vals: Optional[Union[List, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the x and y values from a dataframe, converting y to binary values.
    The output y elements will be 1 if and only if the y_col value is in y_is_true_vals.

    Args:
        data: The dataframe to get the x and y values from
        x_col: The name of the column to use as x
        y_col: The name of the column to use as y
        y_is_true_vals: The values in y_col that are considered 1
        y_is_false_vals: The values in y_col that are considered 0 (optional)

    Returns:
        Tuple of (X, y) where X contains features and y contains binary labels
    """
    # Extract feature column X as numpy array
    X = np.array(data[x_col].tolist())

    # Convert y_is_true_vals to set if it's a string
    if isinstance(y_is_true_vals, str):
        y_is_true_vals = {y_is_true_vals}
    else:
        y_is_true_vals = set(y_is_true_vals)

    # Get y values from dataframe
    y_values = data[y_col].values

    # Initialize y
    if y_is_false_vals is None:
        # If no false values specified, all non-true values are 0
        y = np.zeros(len(X))
        for i, val in enumerate(y_values):
            if val in y_is_true_vals:
                y[i] = 1
    else:
        # If false values specified, values not in either set are NaN
        if isinstance(y_is_false_vals, str):
            y_is_false_vals = {y_is_false_vals}
        else:
            y_is_false_vals = set(y_is_false_vals)

        # Initialize with NaN
        y = np.full(len(X), np.nan)

        # Set values
        for i, val in enumerate(y_values):
            if val in y_is_true_vals:
                y[i] = 1
            elif val in y_is_false_vals:
                y[i] = 0

        # Drop NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

    return X, y.astype(int)


class DimensionReducer(BaseEstimator, TransformerMixin):
    """
    Transformer to reduce dimensionality by selecting first n dimensions.

    This is useful for embedding vectors where the first dimensions typically
    contain the most important information.
    """

    def __init__(self, max_dims: Optional[int] = None):
        """
        Initialize the dimension reducer.

        Args:
            max_dims: Maximum number of dimensions to keep (None keeps all)
        """
        self.max_dims = max_dims

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DimensionReducer':
        """
        Fit method (no-op as this is a simple feature selector).

        Args:
            X: Input features
            y: Target values (not used)

        Returns:
            Self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X by selecting first max_dims dimensions.

        Args:
            X: Input features

        Returns:
            Reduced feature matrix
        """
        if self.max_dims is None or X.shape[1] <= self.max_dims:
            return X
        return X[:, : self.max_dims]


class MoodEstimator(BaseEstimator, TransformerMixin):
    """
    Custom estimator that wraps an underlying model and provides methods to transform
    predictions to mood scores in the [0, 1] range.
    """

    def __init__(
        self,
        model: BaseEstimator,
        max_dims: Optional[int] = None,
        output_transform: str = 'sigmoid',
        threshold: Optional[float] = None,
        data_type: str = 'numerical',
    ):
        """
        Initialize the mood estimator.

        Args:
            model: The underlying scikit-learn model
            max_dims: Maximum dimensions to use from input features
            output_transform: Method to transform output to [0, 1] ('sigmoid', 'minmax', or 'tanh')
            threshold: Classification threshold for binary predictions (if applicable)
            data_type: Type of data this model handles ('numerical', 'ordinal', or 'binary')
        """
        self.model = model
        self.max_dims = max_dims
        self.output_transform = output_transform
        self.threshold = threshold
        self.data_type = data_type
        self.dimension_reducer = DimensionReducer(max_dims=max_dims)

        self._transform_functions = {
            'sigmoid': _sigmoid_transform,
            'minmax': _minmax_transform,
            'tanh': _tanh_transform,
        }

        # Detect if model is a classifier or regressor for sklearn compatibility
        self._is_classifier = (
            (data_type == 'binary')
            or hasattr(model, 'predict_proba')
            or isinstance(
                model,
                (LogisticRegression, SVC, RandomForestClassifier, ClassifierMixin),
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MoodEstimator':
        """
        Fit the model to the data.

        Args:
            X: Input features
            y: Target values

        Returns:
            Self
        """
        X_reduced = self.dimension_reducer.transform(X)
        self.model.fit(X_reduced, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the underlying model.

        Args:
            X: Input features

        Returns:
            Raw model predictions
        """
        X_reduced = self.dimension_reducer.transform(X)
        return self.model.predict(X_reduced)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features to mood scores in [0, 1] range.

        Args:
            X: Input features

        Returns:
            Mood scores in [0, 1] range
        """
        X_reduced = self.dimension_reducer.transform(X)

        # Get raw predictions
        if (
            self._is_classifier
            and hasattr(self.model, 'predict_proba')
            and self.threshold is None
        ):
            # For models with predict_proba, use probability of positive class
            try:
                y_pred = self.model.predict_proba(X_reduced)[:, 1]
                return y_pred
            except (IndexError, AttributeError):
                pass

        # Use regular predictions
        y_pred = self.model.predict(X_reduced)

        # Apply the selected transform function
        transform_func = self._transform_functions.get(
            self.output_transform, _sigmoid_transform
        )

        return transform_func(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities if the underlying model supports it.

        Args:
            X: Input features

        Returns:
            Class probabilities if available, otherwise transformed predictions
        """
        if not self._is_classifier:
            # Explicitly raise an error for non-classifiers to avoid sklearn confusion
            raise AttributeError(
                "predict_proba is not available when the base model is not a classifier"
            )

        X_reduced = self.dimension_reducer.transform(X)
        if hasattr(self.model, 'predict_proba'):
            try:
                return self.model.predict_proba(X_reduced)
            except (AttributeError, NotImplementedError):
                pass

        # Fall back to transform
        scores = self.transform(X)
        return np.column_stack((1 - scores, scores))

    # Add explicit sklearn compatibility methods
    def _more_tags(self):
        """Provide additional tags to help sklearn correctly identify estimator type."""
        return {
            "_skip_test": True,  # Skip sklearn's estimator tests
            "requires_y": True,  # Model requires target for fitting
        }

    def _get_tags(self):
        """Override the get_tags method for better sklearn compatibility."""
        tags = super()._get_tags()
        if self._is_classifier:
            tags["binary_only"] = (
                True  # Indicate this is a binary classifier when in classifier mode
            )
        return tags

    # This is what sklearn uses for duck-typing
    # Make this look like a classifier when intended to be used as one
    def _is_classifier(self):
        return self.data_type == 'binary'


class MoodModelingManager:
    """
    Manager class for building and evaluating mood models.

    This class handles:
    - Data preparation for different model types
    - Training and evaluation of multiple modeling pipelines
    - Collection of validation statistics
    - Final model selection and fitting
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embedding_col: str = 'embedding',
        score_col: str = 'score',
        test_size: float = 0.2,
        random_state: int = 42,
        models: Optional[Dict] = None,
        verbose: int = 1,
    ):
        """
        Initialize the mood modeling manager.

        Args:
            df: DataFrame containing embeddings and scores
            embedding_col: Name of the column containing embeddings
            score_col: Name of the column containing scores
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            models: Dictionary of modeling pipelines (uses default_models if None)
            verbose: Level of verbosity (0: no output, 1: warnings, 2: info, 3: detailed)
        """
        self.df = df
        self.embedding_col = embedding_col
        self.score_col = score_col
        self.test_size = test_size
        self.random_state = random_state
        self.models = models if models is not None else default_models
        self.verbose = verbose

        # Storage for results
        self.results = {}
        self.final_models = {}
        self.trained_models = {}
        self.cv_results = {}

        # Process the data
        self._prepare_data()

    def _prepare_data(self):
        """Process the input dataframe to extract embeddings and scores."""
        # Ensure embeddings are proper numpy arrays
        if isinstance(self.df[self.embedding_col].iloc[0], (list, tuple)):
            self.df[self.embedding_col] = self.df[self.embedding_col].apply(np.array)

        # Stack embeddings to create the feature matrix
        self.X = np.stack(self.df[self.embedding_col].values)
        self.y = self.df[self.score_col].values

        # Store original ranges for reference
        self.y_min = np.min(self.y)
        self.y_max = np.max(self.y)
        self.unique_scores = sorted(self.df[self.score_col].unique())

        # Check dimensions and warn if potentially problematic
        n_samples, n_features = self.X.shape

        if self.verbose >= 1:
            if n_samples < 20:
                print(
                    f"WARNING: Small sample size ({n_samples} samples) may lead to unstable models"
                )

            for model_name, config in self.models.items():
                max_dims = config.get('max_dims', 100)
                if max_dims > n_samples / 2:
                    print(
                        f"WARNING: Model '{model_name}' uses {max_dims} dimensions with only {n_samples} samples. Consider reducing max_dims to avoid overfitting."
                    )

    def prepare_numerical_data(
        self, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare numerical data from the dataframe.

        Args:
            normalize: Whether to normalize scores to [0, 1]

        Returns:
            Tuple of (X, y) for numerical regression
        """
        y = self.y.copy()

        if normalize and self.y_min != self.y_max:
            y = (y - self.y_min) / (self.y_max - self.y_min)

        return self.X, y

    def prepare_ordinal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare ordinal data from the dataframe.

        Returns:
            Tuple of (X, y) for ordinal regression
        """
        # Convert scores to ordinal ranks (0 to n-1)
        score_to_rank = {score: i for i, score in enumerate(self.unique_scores)}
        y_ordinal = np.array([score_to_rank[score] for score in self.y])

        return self.X, y_ordinal

    def prepare_binary_data(
        self,
        threshold: Optional[Union[float, int]] = None,
        positive_labels: Optional[List] = None,
        negative_labels: Optional[List] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare binary data from the dataframe.

        Args:
            threshold: Score threshold above which is considered positive
            positive_labels: List of scores considered positive
            negative_labels: List of scores considered negative

        Returns:
            Tuple of (X, y) for binary classification
        """
        if positive_labels is not None:
            if negative_labels is None:
                negative_labels = [
                    score
                    for score in self.unique_scores
                    if score not in positive_labels
                ]
            return x_and_y_with_boolean_y(
                self.df,
                self.embedding_col,
                self.score_col,
                positive_labels,
                negative_labels,
            )

        # Use threshold if no specific labels provided
        if threshold is None:
            # Default to median if no threshold specified
            threshold = np.median(self.y)

        y_binary = (self.y > threshold).astype(int)
        return self.X, y_binary

    def _get_model_data(self, model_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the appropriate (X, y) data for a given model configuration.

        Args:
            model_config: Dictionary with model configuration

        Returns:
            Tuple of (X, y) for the model
        """
        data_type = model_config.get('data_type', 'numerical')

        if data_type == 'numerical':
            normalize = model_config.get('normalize', True)
            return self.prepare_numerical_data(normalize=normalize)

        elif data_type == 'ordinal':
            return self.prepare_ordinal_data()

        elif data_type == 'binary':
            threshold = model_config.get('threshold', None)
            positive_labels = model_config.get('positive_labels', None)
            negative_labels = model_config.get('negative_labels', None)
            return self.prepare_binary_data(
                threshold=threshold,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
            )

        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def _create_model(self, model_config: Dict) -> MoodEstimator:
        """
        Create a model based on the provided configuration.

        Args:
            model_config: Dictionary with model configuration

        Returns:
            Configured MoodEstimator
        """
        model_class = model_config.get('model_class')
        model_params = model_config.get('model_params', {})
        max_dims = model_config.get('max_dims', 100)
        output_transform = model_config.get('output_transform', 'sigmoid')
        threshold = model_config.get('threshold', None)
        data_type = model_config.get('data_type', 'numerical')

        # Create the base model
        base_model = model_class(**model_params)

        # Create the mood estimator
        return MoodEstimator(
            model=base_model,
            max_dims=max_dims,
            output_transform=output_transform,
            threshold=threshold,
            data_type=data_type,
        )

    def _evaluate_numerical(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_mood: np.ndarray
    ) -> Dict:
        """
        Evaluate numerical regression predictions.

        Args:
            y_true: True target values
            y_pred: Raw model predictions
            y_mood: Transformed mood scores

        Returns:
            Dictionary of evaluation metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'spearman': spearmanr(y_true, y_mood)[0],
            'kendall_tau': kendalltau(y_true, y_mood)[0],
            'min_pred': np.min(y_pred),
            'max_pred': np.max(y_pred),
            'min_mood': np.min(y_mood),
            'max_mood': np.max(y_mood),
        }

    def _evaluate_ordinal(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_mood: np.ndarray
    ) -> Dict:
        """
        Evaluate ordinal regression predictions.

        Args:
            y_true: True ordinal values
            y_pred: Predicted ordinal values
            y_mood: Transformed mood scores

        Returns:
            Dictionary of evaluation metrics
        """
        return {
            'accuracy': accuracy_score(y_true, np.round(y_pred).astype(int)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'spearman': spearmanr(y_true, y_mood)[0],
            'kendall_tau': kendalltau(y_true, y_mood)[0],
        }

    def _evaluate_binary(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_mood: np.ndarray
    ) -> Dict:
        """
        Evaluate binary classification predictions.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_mood: Probability scores

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred_binary = (y_pred > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, y_mood)
        except:
            auc = np.nan

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0),
            'auc': auc,
        }

        # Add confusion matrix elements
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

        return metrics

    def _evaluate_model(
        self, model: MoodEstimator, X: np.ndarray, y: np.ndarray, data_type: str
    ) -> Dict:
        """
        Evaluate a model on the given data.

        Args:
            model: Model to evaluate
            X: Input features
            y: Target values
            data_type: Type of data ("numerical", "ordinal", or "binary")

        Returns:
            Dictionary of evaluation metrics
        """
        # Get raw predictions
        y_pred = model.predict(X)

        # Get mood scores (transformed to [0, 1])
        y_mood = model.transform(X)

        # Evaluate based on data type
        if data_type == 'numerical':
            return self._evaluate_numerical(y, y_pred, y_mood)
        elif data_type == 'ordinal':
            return self._evaluate_ordinal(y, y_pred, y_mood)
        elif data_type == 'binary':
            return self._evaluate_binary(y, y_pred, y_mood)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def train_and_evaluate(self) -> Dict:
        """
        Train and evaluate all models using a single train/test split, collecting performance metrics.

        Returns:
            Dictionary of results for all models
        """
        results = {}

        for model_name, model_config in self.models.items():
            if self.verbose >= 2:
                print(f"Training and evaluating model: {model_name}")

            # Get data for this model
            data_type = model_config.get('data_type', 'numerical')
            X, y = self._get_model_data(model_config)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Create and train model
            model = self._create_model(model_config)
            model.fit(X_train, y_train)

            # Evaluate on test set
            test_metrics = self._evaluate_model(model, X_test, y_test, data_type)

            # Evaluate on train set (for debugging and comparison)
            train_metrics = self._evaluate_model(model, X_train, y_train, data_type)

            # Store results
            results[model_name] = {
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'data_type': data_type,
                'config': model_config,
            }

            # Store the trained model
            self.trained_models[model_name] = model

            if self.verbose >= 2:
                print(
                    f"  Test metrics: {', '.join([f'{k}={v:.4f}' for k, v in test_metrics.items() if isinstance(v, (int, float))])}"
                )

        self.results = results
        return results

    # Update the cross_validate_models method to use appropriate scoring
    def cross_validate_models(
        self, n_splits: int = 5, metrics: Optional[Set[str]] = None
    ) -> Dict:
        """
        Perform cross-validation on all models to get more stable performance metrics.

        Args:
            n_splits: Number of cross-validation folds
            metrics: Set of metric names to compute (defaults to model-appropriate metrics)

        Returns:
            Dictionary of cross-validation results for all models
        """
        cv_results = {}

        for model_name, model_config in self.models.items():
            if self.verbose >= 2:
                print(f"Cross-validating model: {model_name}")

            # Get data for this model
            data_type = model_config.get('data_type', 'numerical')
            X, y = self._get_model_data(model_config)

            # Define scoring metrics based on data type if not specified
            if metrics is None:
                if data_type == 'numerical' or data_type == 'ordinal':
                    scoring = [
                        'neg_mean_squared_error',
                        'neg_mean_absolute_error',
                        'r2',
                    ]
                elif data_type == 'binary':
                    # Only use classification metrics for binary classification
                    if model_config.get('model_class') in [
                        LogisticRegression,
                        SVC,
                        RandomForestClassifier,
                    ]:
                        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                    else:
                        # For other models converted to binary, use regression metrics
                        scoring = [
                            'neg_mean_squared_error',
                            'neg_mean_absolute_error',
                            'r2',
                        ]
                else:
                    scoring = ['neg_mean_squared_error']
            else:
                scoring = list(metrics)

            # Create model
            model = self._create_model(model_config)

            # Perform cross-validation
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            try:
                # Use a custom scoring function for each model type to avoid sklearn errors
                custom_scoring = {}

                # For binary classifiers, use standard classification metrics
                if data_type == 'binary' and model._is_classifier:
                    for metric in scoring:
                        if metric in [
                            'accuracy',
                            'precision',
                            'recall',
                            'f1',
                            'roc_auc',
                        ]:
                            custom_scoring[metric] = (
                                metric  # Use standard sklearn metrics
                            )

                # For regression models, use regression metrics
                elif data_type in ['numerical', 'ordinal'] or not model._is_classifier:
                    for metric in scoring:
                        if metric in [
                            'neg_mean_squared_error',
                            'neg_mean_absolute_error',
                            'r2',
                        ]:
                            custom_scoring[metric] = metric

                # If no appropriate metrics, fall back to a simple one
                if not custom_scoring:
                    if model._is_classifier:
                        custom_scoring['accuracy'] = 'accuracy'
                    else:
                        custom_scoring['neg_mse'] = 'neg_mean_squared_error'

                # Catch warnings to suppress them in verbose mode
                with warnings.catch_warnings(record=True) as caught_warnings:
                    # Use custom scoring
                    cv_scores = cross_validate(
                        model,
                        X,
                        y,
                        cv=cv,
                        scoring=custom_scoring,
                        return_train_score=True,
                        error_score='raise',  # Better to fail visibly than silently
                    )

                    # Show warnings if in high verbose mode
                    if self.verbose >= 3 and caught_warnings:
                        for warning in caught_warnings:
                            print(
                                f"WARNING during cross-validation of {model_name}: {warning.message}"
                            )

                # Format results
                metrics_results = {}
                for metric_name, metric in custom_scoring.items():
                    # Convert sklearn metric names to our convention
                    display_name = metric_name.replace('neg_', '')
                    if metric_name.startswith('neg_'):
                        # Negate negative metrics back to positive
                        metrics_results[display_name] = -cv_scores[
                            f'test_{metric_name}'
                        ].mean()
                        metrics_results[f'train_{display_name}'] = -cv_scores[
                            f'train_{metric_name}'
                        ].mean()
                    else:
                        metrics_results[display_name] = cv_scores[
                            f'test_{metric_name}'
                        ].mean()
                        metrics_results[f'train_{display_name}'] = cv_scores[
                            f'train_{metric_name}'
                        ].mean()

                    # Add standard deviation
                    if metric_name.startswith('neg_'):
                        metrics_results[f'{display_name}_std'] = cv_scores[
                            f'test_{metric_name}'
                        ].std()
                    else:
                        metrics_results[f'{display_name}_std'] = cv_scores[
                            f'test_{metric_name}'
                        ].std()

                # Add additional metrics not supported directly by sklearn cross_validate
                # For example, spearman correlation for every model type
                if model._is_classifier and 'accuracy' in metrics_results:
                    # Add a placeholder for ordinal metrics if needed
                    metrics_results['spearman'] = None
                    metrics_results['kendall_tau'] = None

                cv_results[model_name] = {
                    'metrics': metrics_results,
                    'data_type': data_type,
                    'config': model_config,
                    'n_splits': n_splits,
                }

                if self.verbose >= 2:
                    print(
                        f"  CV metrics: {', '.join([f'{k}={v:.4f}' for k, v in metrics_results.items() if isinstance(v, (int, float)) and not k.startswith('train_') and not k.endswith('_std') and v is not None])}"
                    )

            except Exception as e:
                if self.verbose >= 1:
                    print(
                        f"WARNING: Cross-validation failed for model '{model_name}': {str(e)}"
                    )
                    if self.verbose >= 3:
                        import traceback

                        traceback.print_exc()

                cv_results[model_name] = {
                    'error': str(e),
                    'data_type': data_type,
                    'config': model_config,
                }

        self.cv_results = cv_results
        return cv_results

    def fit_final_models(self) -> Dict:
        """
        Fit final models on the entire dataset.

        Returns:
            Dictionary of fitted models
        """
        final_models = {}

        for model_name, model_config in self.models.items():
            if self.verbose >= 2:
                print(f"Fitting final model: {model_name}")

            # Get data for this model
            X, y = self._get_model_data(model_config)

            # Check dimensions again
            n_samples, n_features = X.shape
            max_dims = model_config.get('max_dims', 100)

            if self.verbose >= 1 and max_dims > n_samples / 3:
                print(
                    f"WARNING: Final model '{model_name}' may be overfitting with {max_dims} dimensions and only {n_samples} samples"
                )

            # Create and train model on all data
            model = self._create_model(model_config)
            model.fit(X, y)

            # Evaluate on all data (train=test evaluation)
            data_type = model_config.get('data_type', 'numerical')
            metrics = self._evaluate_model(model, X, y, data_type)

            # Store results
            final_models[model_name] = {'model': model, 'full_data_metrics': metrics}

            if self.verbose >= 3:
                print(
                    f"  Full data metrics: {', '.join([f'{k}={v:.4f}' for k, v in metrics.items() if isinstance(v, (int, float))])}"
                )

        self.final_models = final_models
        return final_models

    def get_best_model(
        self,
        metric: str = 'spearman',
        data_type: Optional[str] = None,
        use_cv: bool = False,
    ) -> Tuple[str, MoodEstimator]:
        """
        Get the best model based on a specific metric.

        Args:
            metric: Metric to use for comparison
            data_type: Filter by data type (optional)
            use_cv: Whether to use cross-validation results for comparison (if available)

        Returns:
            Tuple of (model_name, model)
        """
        if use_cv and self.cv_results:
            if self.verbose >= 2:
                print(f"Selecting best model based on cross-validation {metric} score")

            best_score = -float('inf')
            best_model_name = None

            for model_name, result in self.cv_results.items():
                if 'error' in result:
                    continue

                if data_type is not None and result['data_type'] != data_type:
                    continue

                score = result['metrics'].get(metric, -float('inf'))

                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        else:
            if not self.results:
                if self.verbose >= 2:
                    print("No results found, running train_and_evaluate")
                self.train_and_evaluate()

            if self.verbose >= 2:
                print(f"Selecting best model based on single split {metric} score")

            best_score = -float('inf')
            best_model_name = None

            for model_name, result in self.results.items():
                if data_type is not None and result['data_type'] != data_type:
                    continue

                score = result['test_metrics'].get(metric, -float('inf'))

                if score > best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name is None:
            raise ValueError(
                f"No models found matching criteria: metric={metric}, data_type={data_type}"
            )

        if not self.final_models:
            if self.verbose >= 2:
                print("Fitting final models")
            self.fit_final_models()

        if self.verbose >= 1:
            print(f"Selected model '{best_model_name}' with {metric}={best_score:.4f}")

        return best_model_name, self.final_models[best_model_name]['model']

    def predict_mood(
        self, X: np.ndarray, model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict mood scores for new data.

        Args:
            X: Input features
            model_name: Name of model to use (uses best model if None)

        Returns:
            Array of mood scores in [0, 1] range
        """
        if not self.final_models:
            self.fit_final_models()

        if model_name is None:
            model_name, _ = self.get_best_model()

        model = self.final_models[model_name]['model']
        return model.transform(X)

    def get_model_summary(self, use_cv: bool = False, *, egress=pd.DataFrame):
        """
        Get a summary of all models and their performance.

        Args:
            use_cv: Whether to use cross-validation results (if available)

        Returns:
            DataFrame summarizing model performance
        """
        if use_cv:
            if not self.cv_results:
                self.cross_validate_models()
            if self.verbose >= 2:
                print("Generating summary from cross-validation results")

            summary = []

            for model_name, result in self.cv_results.items():
                if 'error' in result:
                    continue

                row = {
                    'model_name': model_name,
                    'data_type': result['data_type'],
                    'n_splits': result['n_splits'],
                }

                # Add metrics
                for metric, value in result['metrics'].items():
                    row[metric] = value

                summary.append(row)

        else:
            if not self.results:
                if self.verbose >= 2:
                    print("No results found, running train_and_evaluate")
                self.train_and_evaluate()

            if self.verbose >= 2:
                print("Generating summary from single split results")

            summary = []

            for model_name, result in self.results.items():
                row = {'model_name': model_name, 'data_type': result['data_type']}

                # Add test metrics
                for metric, value in result['test_metrics'].items():
                    row[f'test_{metric}'] = value

                # Add selected train metrics for comparison
                for metric in ['mse', 'accuracy', 'f1', 'spearman', 'kendall_tau']:
                    if metric in result['train_metrics']:
                        row[f'train_{metric}'] = result['train_metrics'][metric]

                summary.append(row)

        if egress is None:
            egress = lambda x: x

        return egress(summary)


# Default model configurations
default_models = {
    'linear_regression': {
        'data_type': 'numerical',
        'model_class': Ridge,
        'model_params': {'alpha': 1.0},
        'max_dims': 50,
        'output_transform': 'sigmoid',
        'normalize': True,
    },
    'svr': {
        'data_type': 'numerical',
        'model_class': SVR,
        'model_params': {'C': 1.0, 'kernel': 'linear'},
        'max_dims': 50,
        'output_transform': 'minmax',
        'normalize': True,
    },
    'logistic_high_vs_low': {
        'data_type': 'binary',
        'model_class': LogisticRegression,
        'model_params': {'C': 1.0, 'class_weight': 'balanced'},
        'max_dims': 50,
        'threshold': None,  # Will be set to median
        'output_transform': 'sigmoid',
    },
    'svm_high_vs_low': {
        'data_type': 'binary',
        'model_class': SVC,
        'model_params': {'C': 1.0, 'kernel': 'linear', 'probability': True},
        'max_dims': 50,
        'threshold': None,  # Will be set to median
        'output_transform': 'sigmoid',
    },
}

# Add ordinal regression models
with suppress(ImportError, ModuleNotFoundError):
    from mord import LogisticIT, LogisticAT

    default_models['ordered_logistic_it'] = {
        'data_type': 'ordinal',
        'model_class': LogisticIT,
        'model_params': {'alpha': 1.0},
        'max_dims': 50,
        'output_transform': 'minmax',
    }

    default_models['ordered_logistic_at'] = {
        'data_type': 'ordinal',
        'model_class': LogisticAT,
        'model_params': {'alpha': 1.0},
        'max_dims': 50,
        'output_transform': 'minmax',
    }

with suppress(ImportError, ModuleNotFoundError):
    from mord import OrdinalRidge

    default_models['ordinal_ridge'] = {
        'data_type': 'ordinal',
        'model_class': OrdinalRidge,
        'model_params': {'alpha': 1.0},
        'max_dims': 50,
        'output_transform': 'minmax',
    }

with suppress(ImportError, ModuleNotFoundError):
    import ogb
    from ogb import OGBClassifier

    default_models['ogboost'] = {
        'data_type': 'ordinal',
        'model_class': OGBClassifier,
        'model_params': {'n_estimators': 100, 'learning_rate': 0.1},
        'max_dims': 50,
        'output_transform': 'minmax',
    }

# Ordered SVM approach - using scikit-learn's SVC with a specific configuration
default_models['ordinal_svm'] = {
    'data_type': 'ordinal',
    'model_class': SVC,
    'model_params': {
        'C': 1.0,
        'kernel': 'linear',
        'decision_function_shape': 'ovo',  # One-vs-One approach works better for ordinal data
        'probability': True,
    },
    'max_dims': 50,
    'output_transform': 'minmax',
}


# Using Random Forest for ordinal regression
class OrdinalRandomForest(RandomForestRegressor):
    """
    Random Forest customized for ordinal regression.

    Predicts continuous values first, then maps them to the closest ordinal class.
    """

    def __init__(self, ordinal_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.ordinal_classes = ordinal_classes

    def fit(self, X, y):
        # Store unique ordinal classes if not provided
        if self.ordinal_classes is None:
            self.ordinal_classes = np.sort(np.unique(y))

        return super().fit(X, y)

    def predict(self, X):
        # Get continuous predictions from the forest
        continuous_pred = super().predict(X)

        # Map to closest ordinal class
        # For each prediction, find the closest value in ordinal_classes
        mapped_pred = np.zeros_like(continuous_pred, dtype=int)
        for i, pred in enumerate(continuous_pred):
            # Find index of closest ordinal class
            idx = np.abs(self.ordinal_classes - pred).argmin()
            mapped_pred[i] = self.ordinal_classes[idx]

        return mapped_pred


default_models['ordinal_forest'] = {
    'data_type': 'ordinal',
    'model_class': OrdinalRandomForest,
    'model_params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
    'max_dims': 50,
    'output_transform': 'minmax',
}


# Example usage:
"""
# Create a MoodModelingManager with your data
manager = MoodModelingManager(
    df=your_dataframe,
    embedding_col='embedding',
    score_col='score',
    verbose=2  # Show information messages
)

# Initial evaluation with single train/test split
results = manager.train_and_evaluate()

# Get a summary of model performance
summary = manager.get_model_summary()
print(summary)

# For more stable metrics, run cross-validation
cv_results = manager.cross_validate_models(n_splits=5)

# Get CV-based summary
cv_summary = manager.get_model_summary(use_cv=True)
print(cv_summary)

# Fit final models on all data
manager.fit_final_models()

# Get the best model based on Spearman correlation from CV results
best_model_name, best_model = manager.get_best_model(
    metric='spearman', 
    use_cv=True
)

# Use the best model to predict mood scores for new data
new_embeddings = np.array([...])  # Your new embeddings
mood_scores = manager.predict_mood(new_embeddings)

print(f"Best model: {best_model_name}")
print(f"Mood scores: {mood_scores}")
"""
