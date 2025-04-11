"""Tools for model analysis"""

from typing import Dict, List, Optional, Union, Callable, Tuple, Set, Mapping
from dataclasses import dataclass, field
from collections.abc import Mapping
from functools import partial
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_model_stats(model_stats: Mapping) -> Dict[str, Union[str, float]]:
    def _gen():
        for semantic_attribute, stats in model_stats.items():
            yield from (
                dict(semantic_attribute=semantic_attribute, **_stats)
                for _stats in stats
            )
            # yield semantic_attribute, compute_model_stats(manager)

    stats_df = pd.DataFrame(_gen())

    classifier_stats = (
        stats_df[stats_df['data_type'] == 'binary']
        .dropna(axis=1, how='all')
        .reset_index(drop=True)
    )
    regression_stats = (
        stats_df[stats_df['data_type'] != 'binary']
        .dropna(axis=1, how='all')
        .reset_index(drop=True)
    )

    return classifier_stats, regression_stats


@dataclass
class ModelPerformanceAnalyzer:
    """
    Analyze model performance metrics across different semantic attributes and model types.

    This class provides methods to generate statistical summaries and visualizations
    for comparing model performance across different metrics and semantic attributes.

    >>> import pandas as pd
    >>> # Sample data
    >>> data = pd.DataFrame({
    ...     'semantic_attribute': ['irony', 'irony', 'moral', 'moral'],
    ...     'model_name': ['svm', 'log_reg', 'svm', 'log_reg'],
    ...     'accuracy': [0.7, 0.6, 0.9, 0.8]
    ... })
    >>> analyzer = ModelPerformanceAnalyzer(data)
    >>> analyzer.get_metrics()
    ['accuracy']
    """

    data: pd.DataFrame
    relevant_metrics: List[str] = field(default_factory=list)
    model_type_column: str = "model_name"
    attribute_column: str = "semantic_attribute"
    data_type_column: str = "data_type"

    def __post_init__(self):
        """Initialize the analyzer by identifying relevant metrics in the data."""
        self._validate_data()

        if not self.relevant_metrics:
            # Auto-identify metrics by excluding non-metric columns
            non_metric_columns = {
                self.model_type_column,
                self.attribute_column,
                self.data_type_column,
                "n_splits",
            }
            self.relevant_metrics = [
                col for col in self.data.columns if col not in non_metric_columns
            ]

    def _validate_data(self) -> None:
        """Validate the dataframe structure to ensure required columns exist."""
        required_columns = [self.model_type_column, self.attribute_column]
        missing = [col for col in required_columns if col not in self.data.columns]

        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def get_metrics(self) -> List[str]:
        """
        Get the list of available metrics in the dataset.

        Returns:
            List[str]: List of metric names
        """
        return self.relevant_metrics

    def get_attributes(self) -> List[str]:
        """
        Get the list of unique semantic attributes in the dataset.

        Returns:
            List[str]: List of semantic attribute names
        """
        return sorted(self.data[self.attribute_column].unique())

    def get_models(self) -> List[str]:
        """
        Get the list of unique model names in the dataset.

        Returns:
            List[str]: List of model names
        """
        return sorted(self.data[self.model_type_column].unique())

    def get_data_types(self) -> List[str]:
        """
        Get the list of unique data types in the dataset.

        Returns:
            List[str]: List of data types (e.g., binary, numerical, ordinal)
        """
        if self.data_type_column in self.data.columns:
            return sorted(self.data[self.data_type_column].unique())
        return []

    def filter_data(
        self,
        attributes: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Filter the dataframe based on specified attributes, models, and data types.

        Args:
            attributes: List of semantic attributes to include
            models: List of model names to include
            data_types: List of data types to include

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        filtered_data = self.data.copy()

        if attributes:
            filtered_data = filtered_data[
                filtered_data[self.attribute_column].isin(attributes)
            ]

        if models:
            filtered_data = filtered_data[
                filtered_data[self.model_type_column].isin(models)
            ]

        if data_types and self.data_type_column in filtered_data.columns:
            filtered_data = filtered_data[
                filtered_data[self.data_type_column].isin(data_types)
            ]

        return filtered_data

    def _format_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Format table values to be more readable."""
        # Round numeric columns to 3 decimal places
        numeric_cols = table.select_dtypes(include=['float64']).columns
        return table.round({col: 3 for col in numeric_cols})

    def generate_model_comparison_table(
        self,
        metric: str,
        attributes: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate a comparison table of models across different semantic attributes for a given metric.

        Args:
            metric: The metric to compare (e.g., accuracy, f1, r2)
            attributes: List of semantic attributes to include (default: all)
            models: List of model names to include (default: all)
            data_types: List of data types to include (default: all)

        Returns:
            pd.DataFrame: A table with attributes as rows and models as columns
        """
        if metric not in self.relevant_metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")

        filtered_data = self.filter_data(attributes, models, data_types)

        if filtered_data.empty:
            return pd.DataFrame()

        # Pivot the data to create a table with attributes as rows and models as columns
        pivot_table = filtered_data.pivot_table(
            index=[self.attribute_column],
            columns=[self.model_type_column],
            values=metric,
            aggfunc='mean',
        )

        # Calculate row averages (average for each attribute across models)
        pivot_table['Average'] = pivot_table.mean(axis=1)
        
        # Sort by the average (descending)
        pivot_table = pivot_table.sort_values('Average', ascending=False)
        
        # Reset index with ranking
        pivot_table = pivot_table.reset_index()
        
        return self._format_table(pivot_table)

    def generate_attribute_modelability_table(
        self,
        primary_metric: str,
        secondary_metrics: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate a table showing which semantic attributes are most modelable.

        Args:
            primary_metric: The main metric to sort by (e.g., accuracy, r2)
            secondary_metrics: Additional metrics to include in the table
            models: List of model names to include (default: all)
            data_types: List of data types to include (default: all)

        Returns:
            pd.DataFrame: A table with attributes as rows and metrics as columns
        """
        if primary_metric not in self.relevant_metrics:
            raise ValueError(f"Metric '{primary_metric}' not found in the data")

        metrics_to_include = [primary_metric]
        if secondary_metrics:
            invalid_metrics = [
                m for m in secondary_metrics if m not in self.relevant_metrics
            ]
            if invalid_metrics:
                raise ValueError(f"Metrics not found: {', '.join(invalid_metrics)}")
            metrics_to_include.extend(secondary_metrics)

        filtered_data = self.filter_data(models=models, data_types=data_types)

        if filtered_data.empty:
            return pd.DataFrame()

        # For each attribute, calculate the average of each metric across all models
        result = []

        for attribute in filtered_data[self.attribute_column].unique():
            attribute_data = filtered_data[
                filtered_data[self.attribute_column] == attribute
            ]

            row = {'semantic_attribute': attribute}

            for metric in metrics_to_include:
                row[metric] = attribute_data[metric].mean()

            result.append(row)

        # Convert to DataFrame and sort by primary metric (descending)
        result_df = pd.DataFrame(result)
        result_df = result_df.sort_values(primary_metric, ascending=False)
        
        # Reset index to show ranking
        result_df = result_df.reset_index(drop=True)

        return self._format_table(result_df)

    def visualize_model_comparison(
        self,
        metric: str,
        attributes: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        palette: str = "viridis",
        tight_ylim: bool = True,
    ) -> plt.Figure:
        """
        Create a grouped bar chart comparing model performance across semantic attributes.

        Args:
            metric: The metric to visualize
            attributes: List of semantic attributes to include (default: all)
            models: List of model names to include (default: all)
            data_types: List of data types to include (default: all)
            figsize: Figure size (width, height)
            palette: Color palette for the plot
            tight_ylim: Whether to use tight y-axis limits (default: True)

        Returns:
            plt.Figure: The matplotlib figure object
        """
        if metric not in self.relevant_metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")

        filtered_data = self.filter_data(attributes, models, data_types)

        if filtered_data.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No data available with the specified filters",
                ha='center',
                va='center',
            )
            return fig

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create grouped bar chart
        sns.barplot(
            data=filtered_data,
            x=self.attribute_column,
            y=metric,
            hue=self.model_type_column,
            palette=palette,
            ax=ax,
        )

        # Set tight y limits if specified
        if tight_ylim:
            min_val = filtered_data[metric].min()
            max_val = filtered_data[metric].max()
            # Add a small padding (5% of range)
            padding = 0.05 * (max_val - min_val)
            y_min = max(0, min_val - padding)  # Ensure non-negative for most metrics
            y_max = min(1, max_val + padding)  # Keep under 1 for normalized metrics
            ax.set_ylim(y_min, y_max)

        # Customize the plot
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}')
        ax.set_xlabel('Semantic Attribute')
        ax.set_ylabel(metric.replace("_", " ").title())

        # Rotate x-axis labels if there are many attributes
        if len(filtered_data[self.attribute_column].unique()) > 5:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        return fig

    def visualize_attribute_modelability(
        self,
        metric: str,
        subset_idx: Optional[int] = None,
        models: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        palette: str = "viridis",
        tight_xlim: bool = True,
    ) -> plt.Figure:
        """
        Create a horizontal bar chart showing which attributes are most/least modelable.

        Args:
            metric: The metric to visualize
            subset_idx: When positive, show first N attributes; when negative, show last N attributes
                        (None = show all attributes)
            models: List of model names to include (default: all)
            data_types: List of data types to include (default: all)
            figsize: Figure size (width, height)
            palette: Color palette for the plot
            tight_xlim: Whether to use tight x-axis limits (default: True)

        Returns:
            plt.Figure: The matplotlib figure object
        """
        if metric not in self.relevant_metrics:
            raise ValueError(f"Metric '{metric}' not found in the data")

        # Get attribute modelability table
        modelability_table = self.generate_attribute_modelability_table(
            primary_metric=metric, models=models, data_types=data_types
        )

        if modelability_table.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No data available with the specified filters",
                ha='center',
                va='center',
            )
            return fig

        # Subset the data based on subset_idx
        if subset_idx is not None:
            if subset_idx > 0:
                # Take first N attributes (top performers)
                display_data = modelability_table.iloc[:subset_idx].copy()
                title_prefix = f"Top {subset_idx}"
            elif subset_idx < 0:
                # Take last N attributes (bottom performers)
                display_data = modelability_table.iloc[subset_idx:].copy()
                title_prefix = f"Bottom {abs(subset_idx)}"
            else:
                # subset_idx is 0, show all data
                display_data = modelability_table.copy()
                title_prefix = "All"
        else:
            # No subsetting, show all data
            display_data = modelability_table.copy()
            title_prefix = "All"

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal bar chart
        sns.barplot(
            data=display_data,
            y='semantic_attribute',
            x=metric,
            palette=palette,
            ax=ax,
        )

        # Set tight x limits if specified
        if tight_xlim and not display_data.empty:
            min_val = display_data[metric].min()
            max_val = display_data[metric].max()
            # Add a small padding (5% of range)
            padding = 0.05 * (max_val - min_val)
            x_min = max(0, min_val - padding)  # Ensure non-negative for most metrics
            x_max = min(1, max_val + padding)  # Keep under 1 for normalized metrics
            ax.set_xlim(x_min, x_max)

        # Customize the plot
        ax.set_title(f'{title_prefix} Attribute Modelability: {metric.replace("_", " ").title()}')
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel('Semantic Attribute')

        plt.tight_layout()

        return fig

    def visualize_metric_distributions(
        self,
        metrics: List[str],
        models: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Create box plots showing the distribution of metrics across semantic attributes.

        Args:
            metrics: List of metrics to visualize
            models: List of model names to include (default: all)
            data_types: List of data types to include (default: all)
            figsize: Figure size (width, height)

        Returns:
            plt.Figure: The matplotlib figure object
        """
        invalid_metrics = [m for m in metrics if m not in self.relevant_metrics]
        if invalid_metrics:
            raise ValueError(f"Metrics not found: {', '.join(invalid_metrics)}")

        filtered_data = self.filter_data(models=models, data_types=data_types)

        if filtered_data.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No data available with the specified filters",
                ha='center',
                va='center',
            )
            return fig

        # Create figure with subplots (one for each metric)
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]  # Make axes iterable when there's only one metric

        for i, metric in enumerate(metrics):
            # Create box plot for this metric
            sns.boxplot(
                data=filtered_data, x=self.attribute_column, y=metric, ax=axes[i]
            )

            # Customize this subplot
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel('')  # Remove x-label (except for the last subplot)
            axes[i].set_ylabel(metric.replace("_", " ").title())

        # Set the x-label only for the last subplot
        axes[-1].set_xlabel('Semantic Attribute')

        # Rotate x-axis labels if there are many attributes
        if len(filtered_data[self.attribute_column].unique()) > 5:
            plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        return fig

    def visualize_correlation_matrix(
        self, metrics: Optional[List[str]] = None, figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create a heatmap showing the correlation between different metrics.

        Args:
            metrics: List of metrics to include in the correlation matrix (default: all)
            figsize: Figure size (width, height)

        Returns:
            plt.Figure: The matplotlib figure object
        """
        metrics_to_include = metrics or self.relevant_metrics
        invalid_metrics = [
            m for m in metrics_to_include if m not in self.relevant_metrics
        ]
        if invalid_metrics:
            raise ValueError(f"Metrics not found: {', '.join(invalid_metrics)}")

        # Calculate correlation matrix for the selected metrics
        correlation_matrix = self.data[metrics_to_include].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            ax=ax,
        )

        # Customize the plot
        ax.set_title('Correlation Matrix of Performance Metrics')

        plt.tight_layout()

        return fig

    def visualize_performance_radar(
        self,
        metrics: List[str],
        models: List[str],
        attribute: str,
        data_type: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Create a radar chart comparing models across multiple metrics for a specific attribute.

        Args:
            metrics: List of metrics to include in the radar chart
            models: List of model names to include
            attribute: The semantic attribute to analyze
            data_type: Data type to filter by (default: None)
            figsize: Figure size (width, height)

        Returns:
            plt.Figure: The matplotlib figure object
        """
        invalid_metrics = [m for m in metrics if m not in self.relevant_metrics]
        if invalid_metrics:
            raise ValueError(f"Metrics not found: {', '.join(invalid_metrics)}")

        # Filter data for the specified attribute and models
        data_types_filter = [data_type] if data_type else None
        filtered_data = self.filter_data(
            attributes=[attribute], models=models, data_types=data_types_filter
        )

        if filtered_data.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No data available with the specified filters",
                ha='center',
                va='center',
            )
            return fig

        # Extract metric values for each model
        model_metrics = {}
        for model in models:
            model_data = filtered_data[filtered_data[self.model_type_column] == model]
            if not model_data.empty:
                model_metrics[model] = [model_data[metric].mean() for metric in metrics]

        if not model_metrics:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No data available for the specified models",
                ha='center',
                va='center',
            )
            return fig

        # Create radar chart
        # Number of variables
        N = len(metrics)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # Set the angle for each metric (evenly spaced)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Set the labels for each metric
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])

        # Draw the chart for each model
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_metrics)))
        for i, (model, values) in enumerate(model_metrics.items()):
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f'Model Comparison for {attribute}')

        return fig


def compare_models_across_datasets(
    classifier_data: pd.DataFrame,
    regression_data: pd.DataFrame,
    common_models: Optional[List[str]] = None,
    common_attributes: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compare model performance across classifier and regression datasets.

    Args:
        classifier_data: Dataframe with classifier metrics
        regression_data: Dataframe with regression metrics
        common_models: List of model names to compare (must exist in both datasets)
        common_attributes: List of attributes to compare (must exist in both datasets)

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with comparison tables
    """
    # Create analyzers for both datasets
    classifier_analyzer = ModelPerformanceAnalyzer(classifier_data)
    regression_analyzer = ModelPerformanceAnalyzer(regression_data)

    # Identify common models and attributes if not provided
    if not common_models:
        classifier_models = set(classifier_analyzer.get_models())
        regression_models = set(regression_analyzer.get_models())
        common_models = sorted(classifier_models.intersection(regression_models))

    if not common_attributes:
        classifier_attrs = set(classifier_analyzer.get_attributes())
        regression_attrs = set(regression_analyzer.get_attributes())
        common_attributes = sorted(classifier_attrs.intersection(regression_attrs))

    # Identify key metrics for each dataset
    classifier_key_metrics = ['accuracy', 'f1', 'roc_auc']
    classifier_key_metrics = [
        m for m in classifier_key_metrics if m in classifier_analyzer.get_metrics()
    ]

    regression_key_metrics = ['r2', 'mean_squared_error', 'mean_absolute_error']
    regression_key_metrics = [
        m for m in regression_key_metrics if m in regression_analyzer.get_metrics()
    ]

    # Generate comparison tables
    comparison = {}

    # Best classifiers per attribute
    if classifier_key_metrics:
        comparison['best_classifiers'] = (
            classifier_analyzer.generate_model_comparison_table(
                metric=classifier_key_metrics[0],
                attributes=common_attributes,
                models=common_models,
            )
        )

    # Best regression models per attribute
    if regression_key_metrics:
        comparison['best_regression_models'] = (
            regression_analyzer.generate_model_comparison_table(
                metric=regression_key_metrics[0],
                attributes=common_attributes,
                models=common_models,
            )
        )

    # Most modelable attributes for classification
    if classifier_key_metrics:
        comparison['modelable_attributes_classification'] = (
            classifier_analyzer.generate_attribute_modelability_table(
                primary_metric=classifier_key_metrics[0],
                secondary_metrics=classifier_key_metrics[1:],
                models=common_models,
            )
        )

    # Most modelable attributes for regression
    if regression_key_metrics:
        comparison['modelable_attributes_regression'] = (
            regression_analyzer.generate_attribute_modelability_table(
                primary_metric=regression_key_metrics[0],
                secondary_metrics=regression_key_metrics[1:],
                models=common_models,
            )
        )

    return comparison


def _print_markdown_heading(text, level=2):
    """Print a markdown heading with the specified level."""
    prefix = "#" * level
    print(f"\n{prefix} {text}\n")


def _print_markdown_subheading(text):
    """Print a markdown subheading (level 3)."""
    _print_markdown_heading(text, level=3)


def _print_dataframe_as_markdown(df, description=None):
    """Print a dataframe as a markdown table with an optional description."""
    if description:
        print(f"*{description}*\n")
    print(df.to_markdown())
    print()


# Example 1: Basic usage with classifier data
def analyze_classifiers(classifier_stats):
    """Example analysis of classifier models."""
    _print_markdown_heading("Classification Model Analysis")
    print(
        "This analysis compares different classification models across semantic attributes to identify which models perform best overall and which semantic attributes are most effectively modeled by classification approaches."
    )

    # Initialize the analyzer
    classifier_analyzer = ModelPerformanceAnalyzer(classifier_stats)

    # Get available metrics, attributes, and models
    metrics = classifier_analyzer.get_metrics()
    attributes = classifier_analyzer.get_attributes()
    models = classifier_analyzer.get_models()

    _print_markdown_subheading("Dataset Overview")
    print(f"* **Available metrics:** {', '.join(metrics)}")
    print(f"* **Semantic attributes:** {', '.join(attributes)}")
    print(f"* **Models evaluated:** {', '.join(models)}")

    # Generate a table comparing models across attributes for accuracy
    model_comparison = classifier_analyzer.generate_model_comparison_table(
        metric="accuracy",
        # Optional filters:
        # attributes=["irony_humor", "moral_outrage"],
        # models=["logistic_high_vs_low", "svm_high_vs_low"]
    )

    _print_markdown_subheading("Model Comparison by Accuracy")
    _print_dataframe_as_markdown(
        model_comparison,
        "This table shows how different classification models perform across semantic attributes, with rows sorted by average accuracy. Higher values indicate better performance.",
    )

    # Generate a table showing which attributes are most modelable
    modelability_table = classifier_analyzer.generate_attribute_modelability_table(
        primary_metric="accuracy", secondary_metrics=["precision", "recall", "f1"]
    )

    _print_markdown_subheading("Semantic Attribute Modelability")
    _print_dataframe_as_markdown(
        modelability_table,
        "This table ranks semantic attributes by how accurately they can be modeled, with additional performance metrics for comprehensive evaluation. Higher values indicate attributes that are easier to model.",
    )

    # Visualize model comparison
    _print_markdown_subheading("Visualizations")
    print("The following visualizations have been generated and saved:")

    fig1 = classifier_analyzer.visualize_model_comparison(
        metric="accuracy",
        attributes=attributes,
        figsize=(12, 6),
        tight_ylim=True,
    )
    fig1.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 1:** Model comparison across semantic attributes (saved as `model_comparison.png`)"
    )
    print("  * Shows how different models perform on each semantic attribute")

    # Visualize top 5 most modelable attributes
    fig2a = classifier_analyzer.visualize_attribute_modelability(
        metric="accuracy", 
        subset_idx=5,  # Top 5 
        figsize=(10, 6),
        tight_xlim=True
    )
    fig2a.savefig("top_modelable_attributes.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 2a:** Most modelable attributes (saved as `top_modelable_attributes.png`)"
    )
    print("  * Shows the 5 semantic attributes that are most accurately modeled")
    
    # Visualize bottom 5 least modelable attributes
    fig2b = classifier_analyzer.visualize_attribute_modelability(
        metric="accuracy", 
        subset_idx=-5,  # Bottom 5 
        figsize=(10, 6),
        tight_xlim=True
    )
    fig2b.savefig("bottom_modelable_attributes.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 2b:** Least modelable attributes (saved as `bottom_modelable_attributes.png`)"
    )
    print("  * Shows the 5 semantic attributes that are most difficult to model accurately")

    # Visualize metric distributions
    fig3 = classifier_analyzer.visualize_metric_distributions(
        metrics=["accuracy", "precision", "recall"], figsize=(12, 10)
    )
    fig3.savefig("metric_distributions.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 3:** Distribution of metrics across attributes (saved as `metric_distributions.png`)"
    )
    print(
        "  * Shows the variability of performance metrics across different semantic attributes"
    )

    return classifier_analyzer


# Example 2: Basic usage with regression data
def analyze_regression_models(regression_stats):
    """Example analysis of regression models."""
    _print_markdown_heading("Regression Model Analysis")
    print(
        "This analysis evaluates different regression models across semantic attributes to determine which regression approaches are most effective and which semantic attributes can be modeled with greater precision using regression techniques."
    )

    # Initialize the analyzer
    regression_analyzer = ModelPerformanceAnalyzer(regression_stats)

    # Generate a table comparing models across attributes for R²
    model_comparison = regression_analyzer.generate_model_comparison_table(metric="r2")

    _print_markdown_subheading("Regression Model Comparison (R²)")
    _print_dataframe_as_markdown(
        model_comparison,
        "This table compares regression models across semantic attributes using R² (coefficient of determination). Higher values indicate models that explain more variance in the data.",
    )

    # Generate a table showing which attributes are most modelable with regression
    modelability_table = regression_analyzer.generate_attribute_modelability_table(
        primary_metric="r2",
        secondary_metrics=["mean_squared_error", "mean_absolute_error"],
    )

    _print_markdown_subheading("Regression Modelability by Attribute")
    _print_dataframe_as_markdown(
        modelability_table,
        "This table ranks semantic attributes by how well they can be modeled using regression techniques. Higher R² and lower error metrics indicate attributes that are more effectively modeled.",
    )

    # Visualize model comparison for regression
    _print_markdown_subheading("Visualizations")
    print("The following visualizations have been generated and saved:")

    fig1 = regression_analyzer.visualize_model_comparison(
        metric="r2",
        data_types=["numerical"],  # Filter to only numerical data
        figsize=(12, 6),
        tight_ylim=True,
    )
    fig1.savefig("regression_model_comparison.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 1:** Regression model comparison (saved as `regression_model_comparison.png`)"
    )
    print(
        "  * Compares R² values across models and semantic attributes for numerical data"
    )

    # Visualize top 5 most modelable attributes for regression
    fig2a = regression_analyzer.visualize_attribute_modelability(
        metric="r2",
        subset_idx=5,  # Top 5
        figsize=(10, 6),
        tight_xlim=True,
    )
    fig2a.savefig("top_regression_modelable_attributes.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 2a:** Most modelable attributes for regression (saved as `top_regression_modelable_attributes.png`)"
    )
    print("  * Shows the 5 semantic attributes that are most effectively modeled with regression")
    
    # Visualize bottom 5 least modelable attributes for regression
    fig2b = regression_analyzer.visualize_attribute_modelability(
        metric="r2",
        subset_idx=-5,  # Bottom 5
        figsize=(10, 6),
        tight_xlim=True,
    )
    fig2b.savefig("bottom_regression_modelable_attributes.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 2b:** Least modelable attributes for regression (saved as `bottom_regression_modelable_attributes.png`)"
    )
    print("  * Shows the 5 semantic attributes that are most difficult to model with regression")

    # Visualize correlation matrix of metrics
    fig3 = regression_analyzer.visualize_correlation_matrix(
        metrics=["r2", "mean_squared_error", "mean_absolute_error"], figsize=(8, 6)
    )
    fig3.savefig("metric_correlation.png", dpi=300, bbox_inches="tight")
    print(
        "* **Figure 3:** Correlation matrix of regression metrics (saved as `metric_correlation.png`)"
    )
    print("  * Shows relationships between different performance metrics")

    return regression_analyzer

# Example 3: Advanced analysis across both datasets
def compare_datasets(classifier_stats, regression_stats):
    """Compare model performance across classifier and regression datasets."""
    _print_markdown_heading("Cross-Dataset Comparison Analysis")
    print(
        "This analysis compares classification and regression approaches to identify which modeling paradigm is more effective for each semantic attribute and to determine if certain attributes are better suited to one approach over the other."
    )

    # Get common attributes (those that appear in both datasets)
    classifier_attrs = set(ModelPerformanceAnalyzer(classifier_stats).get_attributes())
    regression_attrs = set(ModelPerformanceAnalyzer(regression_stats).get_attributes())
    common_attributes = sorted(classifier_attrs.intersection(regression_attrs))

    print(
        f"Analyzing {len(common_attributes)} semantic attributes that appear in both datasets."
    )

    # Run the comparison
    comparison_results = compare_models_across_datasets(
        classifier_data=classifier_stats,
        regression_data=regression_stats,
        common_attributes=common_attributes,
    )

    if 'best_classifiers' in comparison_results:
        _print_markdown_subheading("Best Classifiers per Attribute")
        _print_dataframe_as_markdown(
            comparison_results['best_classifiers'],
            "This table shows which classification models perform best for each semantic attribute.",
        )

    if 'best_regression_models' in comparison_results:
        _print_markdown_subheading("Best Regression Models per Attribute")
        _print_dataframe_as_markdown(
            comparison_results['best_regression_models'],
            "This table shows which regression models perform best for each semantic attribute.",
        )

    if 'modelable_attributes_classification' in comparison_results:
        _print_markdown_subheading("Most Modelable Attributes (Classification)")
        _print_dataframe_as_markdown(
            comparison_results['modelable_attributes_classification'],
            "This table ranks semantic attributes by how well they can be modeled using classification approaches.",
        )

    if 'modelable_attributes_regression' in comparison_results:
        _print_markdown_subheading("Most Modelable Attributes (Regression)")
        _print_dataframe_as_markdown(
            comparison_results['modelable_attributes_regression'],
            "This table ranks semantic attributes by how well they can be modeled using regression approaches.",
        )

    return comparison_results


# Example 4: Focused analysis on a specific attribute
def analyze_specific_attribute(
    classifier_stats, regression_stats, attribute="irony_humor"
):
    """Focused analysis on a specific semantic attribute."""
    _print_markdown_heading(f"Detailed Analysis of '{attribute}'")
    print(
        f"This analysis focuses specifically on the '{attribute}' semantic attribute, comparing how different classification and regression models perform when modeling this particular attribute."
    )

    # Initialize analyzers
    classifier_analyzer = ModelPerformanceAnalyzer(classifier_stats)
    regression_analyzer = ModelPerformanceAnalyzer(regression_stats)

    # Filter data for the specific attribute
    classifier_data = classifier_analyzer.filter_data(attributes=[attribute])
    regression_data = regression_analyzer.filter_data(attributes=[attribute])

    # Create new analyzers with the filtered data
    attr_classifier_analyzer = ModelPerformanceAnalyzer(classifier_data)
    attr_regression_analyzer = ModelPerformanceAnalyzer(regression_data)

    # Get model performance for this attribute
    classifier_models = attr_classifier_analyzer.get_models()
    regression_models = attr_regression_analyzer.get_models()

    _print_markdown_subheading("Classification Models")
    if classifier_data.empty:
        print(f"No classification data available for attribute '{attribute}'.")
    else:
        classifier_comparison = (
            attr_classifier_analyzer.generate_model_comparison_table(metric="accuracy")
        )
        _print_dataframe_as_markdown(
            classifier_comparison,
            f"Classification model performance for '{attribute}' (sorted by accuracy)",
        )

    _print_markdown_subheading("Regression Models")
    if regression_data.empty:
        print(f"No regression data available for attribute '{attribute}'.")
    else:
        regression_comparison = (
            attr_regression_analyzer.generate_model_comparison_table(metric="r2")
        )
        _print_dataframe_as_markdown(
            regression_comparison,
            f"Regression model performance for '{attribute}' (sorted by R²)",
        )

    # Compare models for this attribute using radar chart
    _print_markdown_subheading("Visualizations")
    print(
        "The following radar charts have been generated to compare models across multiple metrics:"
    )

    if not classifier_data.empty:
        classifier_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        available_metrics = [
            m for m in classifier_metrics if m in attr_classifier_analyzer.get_metrics()
        ]

        if available_metrics:
            fig1 = attr_classifier_analyzer.visualize_performance_radar(
                metrics=available_metrics,
                models=classifier_models,
                attribute=attribute,
                figsize=(10, 8),
            )
            fig1.savefig(
                f"{attribute}_classifier_radar.png", dpi=300, bbox_inches="tight"
            )
            print(
                f"* **Figure 1:** Classification model comparison for '{attribute}' (saved as `{attribute}_classifier_radar.png`)"
            )
            print(f"  * Compares models across {', '.join(available_metrics)}")

    if not regression_data.empty:
        regression_metrics = ["r2", "mean_absolute_error"]
        available_metrics = [
            m for m in regression_metrics if m in attr_regression_analyzer.get_metrics()
        ]

        if available_metrics:
            fig2 = attr_regression_analyzer.visualize_performance_radar(
                metrics=available_metrics,
                models=regression_models,
                attribute=attribute,
                figsize=(10, 8),
            )
            fig2.savefig(
                f"{attribute}_regression_radar.png", dpi=300, bbox_inches="tight"
            )
            print(
                f"* **Figure 2:** Regression model comparison for '{attribute}' (saved as `{attribute}_regression_radar.png`)"
            )
            print(f"  * Compares models across {', '.join(available_metrics)}")

    return attr_classifier_analyzer, attr_regression_analyzer


# Example 5: Data-type specific analysis
def analyze_by_data_type(classifier_stats, regression_stats):
    """Analyze performance by data type."""
    _print_markdown_heading("Data Type Analysis")
    print(
        "This analysis examines how model performance varies across different data types (binary, numerical, ordinal), providing insights into which modeling approaches are most effective for each data representation."
    )

    # Initialize analyzers
    classifier_analyzer = ModelPerformanceAnalyzer(classifier_stats)
    regression_analyzer = ModelPerformanceAnalyzer(regression_stats)

    # Get available data types
    classifier_data_types = classifier_analyzer.get_data_types()
    regression_data_types = regression_analyzer.get_data_types()

    print(f"* **Classifier data types:** {', '.join(classifier_data_types)}")
    print(f"* **Regression data types:** {', '.join(regression_data_types)}")

    # Analyze binary classification performance
    if "binary" in classifier_data_types:
        _print_markdown_subheading("Binary Classification Analysis")
        binary_results = classifier_analyzer.generate_model_comparison_table(
            metric="accuracy", data_types=["binary"]
        )
        _print_dataframe_as_markdown(
            binary_results,
            "Model performance comparison for binary classification tasks (sorted by average accuracy)",
        )

        fig1 = classifier_analyzer.visualize_model_comparison(
            metric="accuracy", data_types=["binary"], figsize=(12, 6), tight_ylim=True
        )
        fig1.savefig("binary_model_comparison.png", dpi=300, bbox_inches="tight")
        print(
            "* **Figure:** Binary classification model comparison (saved as `binary_model_comparison.png`)"
        )

    # Analyze numerical regression performance
    if "numerical" in regression_data_types:
        _print_markdown_subheading("Numerical Regression Analysis")
        numerical_results = regression_analyzer.generate_model_comparison_table(
            metric="r2", data_types=["numerical"]
        )
        _print_dataframe_as_markdown(
            numerical_results,
            "Model performance comparison for numerical regression tasks (sorted by average R²)",
        )

        fig2 = regression_analyzer.visualize_model_comparison(
            metric="r2", data_types=["numerical"], figsize=(12, 6), tight_ylim=True
        )
        fig2.savefig("numerical_model_comparison.png", dpi=300, bbox_inches="tight")
        print(
            "* **Figure:** Numerical regression model comparison (saved as `numerical_model_comparison.png`)"
        )

    # Analyze ordinal regression performance
    if "ordinal" in regression_data_types:
        _print_markdown_subheading("Ordinal Regression Analysis")
        ordinal_results = regression_analyzer.generate_model_comparison_table(
            metric="r2", data_types=["ordinal"]
        )
        _print_dataframe_as_markdown(
            ordinal_results,
            "Model performance comparison for ordinal regression tasks (sorted by average R²)",
        )

        fig3 = regression_analyzer.visualize_model_comparison(
            metric="r2", data_types=["ordinal"], figsize=(12, 6), tight_ylim=True
        )
        fig3.savefig("ordinal_model_comparison.png", dpi=300, bbox_inches="tight")
        print(
            "* **Figure:** Ordinal regression model comparison (saved as `ordinal_model_comparison.png`)"
        )


def analyze_all(classifier_stats, regression_stats, attribute="irony_humor"):
    """Run all analyses and generate a complete report."""
    _print_markdown_heading("Complete Model Performance Analysis", level=1)
    print(
        "This report provides a comprehensive analysis of model performance across different semantic attributes, comparing various classification and regression approaches to determine which models perform best and which semantic attributes are most effectively modeled."
    )

    print("Running classifier analysis...")
    classifier_analyzer = analyze_classifiers(classifier_stats)

    print("\nRunning regression analysis...")
    regression_analyzer = analyze_regression_models(regression_stats)

    print("\nComparing datasets...")
    comparison_results = compare_datasets(classifier_stats, regression_stats)

    print("\nAnalyzing specific attribute...")
    attr_classifier, attr_regression = analyze_specific_attribute(
        classifier_stats, regression_stats, attribute
    )

    print("\nAnalyzing by data type...")
    analyze_by_data_type(classifier_stats, regression_stats)

    _print_markdown_heading("Analysis Summary", level=2)
    print(
        "All analyses have been completed and the results have been saved to various output files. This report provides insights into which models perform best for different semantic attributes and which attributes are most effectively modeled using different approaches."
    )