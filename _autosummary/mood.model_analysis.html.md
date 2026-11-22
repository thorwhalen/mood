# mood.model_analysis

Tools for model analysis

### Functions

| [`analyze_all`](#mood.model_analysis.analyze_all)(classifier_stats, regression_stats)   | Run all analyses and generate a complete report.                     |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`analyze_by_data_type`](#mood.model_analysis.analyze_by_data_type)(classifier_stats, ...)       | Analyze performance by data type.                                    |
| [`analyze_classifiers`](#mood.model_analysis.analyze_classifiers)(classifier_stats)             | Example analysis of classifier models.                               |
| [`analyze_regression_models`](#mood.model_analysis.analyze_regression_models)(regression_stats)       | Example analysis of regression models.                               |
| [`analyze_specific_attribute`](#mood.model_analysis.analyze_specific_attribute)(classifier_stats, ...) | Focused analysis on a specific semantic attribute.                   |
| [`compare_datasets`](#mood.model_analysis.compare_datasets)(classifier_stats, ...)           | Compare model performance across classifier and regression datasets. |
| [`compare_models_across_datasets`](#mood.model_analysis.compare_models_across_datasets)(...[, ...])        | Compare model performance across classifier and regression datasets. |
| `compute_model_stats`(model_stats)                                                                 |                                                                      |

### Classes

| [`ModelPerformanceAnalyzer`](#mood.model_analysis.ModelPerformanceAnalyzer)(data[, ...])   | Analyze model performance metrics across different semantic attributes and model types.   |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|

### *class* mood.model_analysis.ModelPerformanceAnalyzer(data, relevant_metrics=list(), model_type_column='model_name', attribute_column='semantic_attribute', data_type_column='data_type')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Analyze model performance metrics across different semantic attributes and model types.

This class provides methods to generate statistical summaries and visualizations
for comparing model performance across different metrics and semantic attributes.

```pycon
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
```

#### filter_data(attributes=None, models=None, data_types=None)

Filter the dataframe based on specified attributes, models, and data types.

* **Parameters:**
  * **attributes** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of semantic attributes to include
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to include
  * **data_types** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of data types to include
* **Returns:**
  Filtered dataframe
* **Return type:**
  `DataFrame`

#### generate_attribute_modelability_table(primary_metric, secondary_metrics=None, models=None, data_types=None)

Generate a table showing which semantic attributes are most modelable.

* **Parameters:**
  * **primary_metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The main metric to sort by (e.g., accuracy, r2)
  * **secondary_metrics** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Additional metrics to include in the table
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to include (default: all)
  * **data_types** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of data types to include (default: all)
* **Returns:**
  A table with attributes as rows and metrics as columns
* **Return type:**
  `DataFrame`

#### generate_model_comparison_table(metric, attributes=None, models=None, data_types=None)

Generate a comparison table of models across different semantic attributes for a given metric.

* **Parameters:**
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The metric to compare (e.g., accuracy, f1, r2)
  * **attributes** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of semantic attributes to include (default: all)
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to include (default: all)
  * **data_types** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of data types to include (default: all)
* **Returns:**
  A table with attributes as rows and models as columns
* **Return type:**
  `DataFrame`

#### get_attributes()

Get the list of unique semantic attributes in the dataset.

* **Returns:**
  List of semantic attribute names
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### get_data_types()

Get the list of unique data types in the dataset.

* **Returns:**
  List of data types (e.g., binary, numerical, ordinal)
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### get_metrics()

Get the list of available metrics in the dataset.

* **Returns:**
  List of metric names
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### get_models()

Get the list of unique model names in the dataset.

* **Returns:**
  List of model names
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### visualize_attribute_modelability(metric, subset_idx=None, models=None, data_types=None, figsize=(10, 6), palette='viridis', tight_xlim=True)

Create a horizontal bar chart showing which attributes are most/least modelable.

* **Parameters:**
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The metric to visualize
  * **subset_idx** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – When positive, show first N attributes; when negative, show last N attributes
    (None = show all attributes)
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to include (default: all)
  * **data_types** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of data types to include (default: all)
  * **figsize** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Figure size (width, height)
  * **palette** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Color palette for the plot
  * **tight_xlim** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to use tight x-axis limits (default: True)
* **Returns:**
  The matplotlib figure object
* **Return type:**
  `Figure`

#### visualize_correlation_matrix(metrics=None, figsize=(10, 8))

Create a heatmap showing the correlation between different metrics.

* **Parameters:**
  * **metrics** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of metrics to include in the correlation matrix (default: all)
  * **figsize** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Figure size (width, height)
* **Returns:**
  The matplotlib figure object
* **Return type:**
  `Figure`

#### visualize_metric_distributions(metrics, models=None, data_types=None, figsize=(12, 8))

Create box plots showing the distribution of metrics across semantic attributes.

* **Parameters:**
  * **metrics** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – List of metrics to visualize
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to include (default: all)
  * **data_types** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of data types to include (default: all)
  * **figsize** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Figure size (width, height)
* **Returns:**
  The matplotlib figure object
* **Return type:**
  `Figure`

#### visualize_model_comparison(metric, attributes=None, models=None, data_types=None, figsize=(10, 6), palette='viridis', tight_ylim=True)

Create a grouped bar chart comparing model performance across semantic attributes.

* **Parameters:**
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The metric to visualize
  * **attributes** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of semantic attributes to include (default: all)
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to include (default: all)
  * **data_types** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of data types to include (default: all)
  * **figsize** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Figure size (width, height)
  * **palette** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Color palette for the plot
  * **tight_ylim** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to use tight y-axis limits (default: True)
* **Returns:**
  The matplotlib figure object
* **Return type:**
  `Figure`

#### visualize_performance_radar(metrics, models, attribute, data_type=None, figsize=(10, 8))

Create a radar chart comparing models across multiple metrics for a specific attribute.

* **Parameters:**
  * **metrics** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – List of metrics to include in the radar chart
  * **models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – List of model names to include
  * **attribute** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The semantic attribute to analyze
  * **data_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Data type to filter by (default: None)
  * **figsize** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Figure size (width, height)
* **Returns:**
  The matplotlib figure object
* **Return type:**
  `Figure`

### mood.model_analysis.analyze_all(classifier_stats, regression_stats, attribute='irony_humor')

Run all analyses and generate a complete report.

### mood.model_analysis.analyze_by_data_type(classifier_stats, regression_stats)

Analyze performance by data type.

### mood.model_analysis.analyze_classifiers(classifier_stats)

Example analysis of classifier models.

### mood.model_analysis.analyze_regression_models(regression_stats)

Example analysis of regression models.

### mood.model_analysis.analyze_specific_attribute(classifier_stats, regression_stats, attribute='irony_humor')

Focused analysis on a specific semantic attribute.

### mood.model_analysis.compare_datasets(classifier_stats, regression_stats)

Compare model performance across classifier and regression datasets.

### mood.model_analysis.compare_models_across_datasets(classifier_data, regression_data, common_models=None, common_attributes=None)

Compare model performance across classifier and regression datasets.

* **Parameters:**
  * **classifier_data** (`DataFrame`) – Dataframe with classifier metrics
  * **regression_data** (`DataFrame`) – Dataframe with regression metrics
  * **common_models** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of model names to compare (must exist in both datasets)
  * **common_attributes** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of attributes to compare (must exist in both datasets)
* **Returns:**
  Dictionary with comparison tables
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `DataFrame`]
