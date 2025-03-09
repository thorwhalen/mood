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
from sklearn.preprocessing import StandardScaler
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
    feature_scaling_used: bool = None

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
        output += f"Feature Scaling: {'Applied' if self.feature_scaling_used else 'Not Applied'}\n"
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
    solver: str = 'lbfgs',
    penalty: str = 'l2',
    class_weight: Optional[Union[Dict, str]] = None,
    scale_features: bool = True,
    skip_validation: bool = False,
    verbose: bool = True,
) -> Tuple[LogisticRegression, ModelStatistics]:
    """
    Fits a logistic regression model on embedding data for binary classification.

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
    scale_features : bool, default=True
        Whether to standardize features before fitting.
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
    stats.feature_scaling_used = scale_features

    # Convert boolean y to integer if needed
    if y.dtype == bool:
        y = y.astype(int)

    # Initialize scaler if needed
    scaler = None
    if scale_features:
        scaler = StandardScaler()

    # Initialize the model
    model = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty,
        class_weight=class_weight,
        random_state=random_state,
    )

    if verbose:
        print(
            f"Initializing logistic regression with C={c_value}, solver='{solver}', penalty='{penalty}'"
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

        # Scale features if requested
        if scale_features:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

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
    if scale_features:
        X_scaled = (
            scaler.fit_transform(X) if scaler is None else scaler.fit_transform(X)
        )
        model.fit(X_scaled, y)
    else:
        model.fit(X, y)

    stats.training_time = time.time() - start_time

    if verbose:
        print(f"Model fitting completed in {stats.training_time:.4f} seconds")
        if not skip_validation:
            print(stats)

    return model, stats


def predict_with_model(model, X, scaler=None):
    """
    Helper function to make predictions with a fitted model.

    Parameters:
    -----------
    model : LogisticRegression
        Trained logistic regression model.
    X : array-like
        Feature vectors to predict on.
    scaler : StandardScaler, optional
        If provided, scales the features before prediction.

    Returns:
    --------
    y_pred : array-like
        Class predictions (0 or 1).
    y_prob : array-like
        Probability estimates for class 1.
    """
    if scaler is not None:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return y_pred, y_prob


# --------------------------------------------------------------------------------------
# Sentiment data access

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Callable, Sequence, Any
import tabled
from functools import partial
import datasets
import dol


class SentimentDacc:
    """
    Base class for sentiment analysis data preparation.
    Provides methods to prepare and transform sentiment data from various sources.
    """

    def __init__(self, label_descriptions: Optional[Union[pd.Series, Dict]] = None):
        """
        Initialize a SentimentDacc object.

        Parameters:
        -----------
        label_descriptions: Optional[Union[pd.Series, Dict]]
            Mapping from label values to their descriptions.
            Can be provided at initialization or set later.
        """
        self._raw_data = None
        self._processed_data = None

        # Set label descriptions if provided
        if label_descriptions is not None:
            if isinstance(label_descriptions, dict):
                self._label_descriptions = pd.Series(label_descriptions)
            else:
                self._label_descriptions = label_descriptions
        else:
            self._label_descriptions = None

    @property
    def label_descriptions(self) -> pd.Series:
        """Get the mapping from labels to their descriptions."""
        if self._label_descriptions is None:
            raise ValueError(
                "Label descriptions have not been set. Use set_label_descriptions method."
            )
        return self._label_descriptions

    def set_label_descriptions(self, descriptions: Union[pd.Series, Dict]) -> None:
        """
        Set the label descriptions.

        Parameters:
        -----------
        descriptions: Union[pd.Series, Dict]
            Mapping from label values to their descriptions.
        """
        if isinstance(descriptions, dict):
            self._label_descriptions = pd.Series(descriptions)
        else:
            self._label_descriptions = descriptions

    def _rename_columns(
        self, df: pd.DataFrame, column_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Rename columns according to a mapping.

        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame to rename columns in
        column_mapping: Dict[str, str]
            Mapping from original column names to new column names

        Returns:
        --------
        pd.DataFrame
            DataFrame with renamed columns
        """
        # Only rename columns that exist in the dataframe
        valid_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        return df.rename(columns=valid_mapping)

    def _expand_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand rows where columns contain lists, creating one row per list element.

        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame that might contain list columns

        Returns:
        --------
        pd.DataFrame
            DataFrame with expanded rows
        """
        # Check if the label column contains sequences (lists)
        if 'label' in df.columns and len(df) > 0:
            # Check if first element is a sequence but not a string
            first_elem = df['label'].iloc[0]
            if isinstance(first_elem, (np.ndarray, Sequence)) and not isinstance(
                first_elem, str
            ):
                first_label_len = len(first_elem)
                list_columns = ['label']

                # Find all columns with lists of the same length
                for column in df.columns:
                    col_first_elem = df[column].iloc[0]
                    if isinstance(col_first_elem, Sequence) and not isinstance(
                        col_first_elem, str
                    ):
                        if len(col_first_elem) == first_label_len:
                            list_columns.append(column)

                # Expand rows for the identified list columns
                if list_columns:
                    df = tabled.expand_rows(df, list_columns)

        return df

    def _map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map numeric labels to their string descriptions using label_descriptions.

        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame with a 'label' column

        Returns:
        --------
        pd.DataFrame
            DataFrame with mapped labels
        """
        if self._label_descriptions is not None and 'label' in df.columns:
            df = df.copy()

            # Handle different label types
            if len(df) > 0:
                # If label is a numpy array or list, convert to its first element
                # This handles the case where labels might be arrays with single values
                if isinstance(df['label'].iloc[0], (np.ndarray, list)):
                    # If arrays have multiple values, we should have already expanded them
                    # Here we're just extracting single values from single-element arrays
                    df['label'] = df['label'].apply(
                        lambda x: (
                            x[0]
                            if isinstance(x, (np.ndarray, list)) and len(x) == 1
                            else x
                        )
                    )

                # Now map the labels using the descriptions
                if df['label'].dtype in [np.int64, np.int32, int, float]:
                    # For numeric labels, we can directly use the mapping
                    df['label'] = df['label'].map(self.label_descriptions)
                else:
                    # For other types, try to convert to int first if possible
                    try:
                        df['label'] = (
                            df['label'].astype(int).map(self.label_descriptions)
                        )
                    except (ValueError, TypeError):
                        # If conversion fails, try to use the labels as-is
                        df['label'] = df['label'].map(self.label_descriptions)

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate that the processed data contains the required columns.

        Parameters:
        -----------
        df: pd.DataFrame
            DataFrame to validate

        Raises:
        -------
        ValueError
            If required columns are missing
        """
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Processed data is missing required columns: {missing_columns}"
            )

    def prepare(
        self,
        reset_index: bool = True,
        drop_index: bool = True,
        map_labels: bool = False,
    ) -> pd.DataFrame:
        """
        Return the prepared data with standard processing applied.

        Parameters:
        -----------
        reset_index: bool, default=True
            Whether to reset the index of the returned DataFrame
        drop_index: bool, default=True
            Whether to drop the old index when resetting
        map_labels: bool, default=False
            Whether to map numeric labels to their string descriptions

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with at least 'text' and 'label' columns

        Raises:
        -------
        ValueError
            If the data has not been loaded or processed
        """
        if self._processed_data is None:
            raise ValueError("No data has been processed. Load and process data first.")

        df = self._processed_data.copy()

        # Map labels if requested
        # if map_labels:
        #     df = self._map_labels(df)

        # Reset index if requested
        if reset_index:
            df = df.reset_index(drop=drop_index)

        # Final validation
        self._validate_data(df)

        return df


class HugSentimentDacc(SentimentDacc):
    """
    Specialized class for preparing sentiment analysis data from Hugging Face datasets.
    """

    def __init__(
        self, dataset_name: Optional[str] = None, label_path: Optional[str] = None
    ):
        """
        Initialize a HugSentimentDacc object.

        Parameters:
        -----------
        dataset_name: Optional[str]
            Name of the Hugging Face dataset to load
        label_path: Optional[str]
            Path to the label descriptions in the dataset's features
        """
        super().__init__()
        self._dataset = None

        if dataset_name is not None:
            self.load_dataset(dataset_name)

        if label_path is not None:
            self.extract_label_descriptions(label_path)

        # return self  # Enable method chaining

    def load_dataset(self, dataset_name: str):
        """
        Load a dataset from Hugging Face.

        Parameters:
        -----------
        dataset_name: str
            Name of the dataset to load

        Returns:
        --------
        self
            For method chaining
        """
        self._dataset = datasets.load_dataset(dataset_name)
        self._raw_data = self._dataset
        return self  # Enable method chaining

    def extract_label_descriptions(self, path: str, sep: str = '.'):
        """
        Extract label descriptions from the dataset features.

        Parameters:
        -----------
        path: str
            Path to the label descriptions in the dataset's features
        sep: str, default='.'
            Separator used in the path

        Returns:
        --------
        self
            For method chaining

        Raises:
        -------
        ValueError
            If the dataset has not been loaded
        """
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        mapping_extractor = partial(dol.path_get, path=path, sep=sep)
        label_mapping = pd.Series(mapping_extractor(self._dataset))
        self.set_label_descriptions(label_mapping)
        return self  # Enable method chaining

    def _concatenate_splits(
        self, splits: List[str] = None, only_train: bool = False
    ) -> pd.DataFrame:
        """
        Concatenate multiple splits from the dataset.

        Parameters:
        -----------
        splits: List[str], default=None
            List of splits to concatenate. If None, uses all available splits.
        only_train: bool, default=False
            If True, only use the 'train' split

        Returns:
        --------
        pd.DataFrame
            Concatenated DataFrame

        Raises:
        -------
        ValueError
            If the dataset has not been loaded
        """
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        if only_train:
            if 'train' not in self._dataset:
                raise ValueError("Dataset does not have a 'train' split")
            return self._dataset['train'].to_pandas()

        if splits is None:
            # Use all available splits
            splits = list(self._dataset.keys())

        df = pd.DataFrame()
        for split in splits:
            if split in self._dataset:
                split_df = self._dataset[split].to_pandas()
                df = pd.concat([df, split_df], ignore_index=True)

        return df

    def process(
        self,
        column_mapping: Dict[str, str] = None,
        only_train: bool = False,
        splits: List[str] = None,
    ):
        """
        Process the loaded dataset into a standardized format.

        Parameters:
        -----------
        column_mapping: Dict[str, str], default=None
            Mapping from original column names to standardized names.
            Must map to 'text' and 'label' for the primary columns.
        only_train: bool, default=False
            If True, only use the 'train' split
        splits: List[str], default=None
            List of splits to use. Ignored if only_train is True.

        Returns:
        --------
        self
            For method chaining

        Raises:
        -------
        ValueError
            If the dataset has not been loaded
        """
        if self._dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        # Default column mapping if none provided
        if column_mapping is None:
            # Try to automatically detect common column names
            first_split = list(self._dataset.keys())[0]
            first_df = self._dataset[first_split].to_pandas()

            # Look for text column
            text_candidates = ['text', 'sentence', 'content', 'dialogue']
            text_col = next(
                (col for col in text_candidates if col in first_df.columns), None
            )

            # Look for label column
            label_candidates = ['label', 'labels', 'emotion', 'sentiment', 'class']
            label_col = next(
                (col for col in label_candidates if col in first_df.columns), None
            )

            if text_col is None or label_col is None:
                raise ValueError(
                    "Could not automatically detect text and label columns. "
                    "Please provide a column_mapping."
                )

            column_mapping = {text_col: 'text', label_col: 'label'}

        # Concatenate splits
        df = self._concatenate_splits(splits=splits, only_train=only_train)

        # Rename columns
        df = self._rename_columns(df, column_mapping)

        # Expand list columns if needed
        df = self._expand_list_columns(df)

        # Store processed data
        self._processed_data = df

        return self  # Enable method chaining
