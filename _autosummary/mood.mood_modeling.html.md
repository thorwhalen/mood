# mood.mood_modeling

Mood Modeling Module

This module provides tools for modeling semantic attributes (or “moods”) based on text embeddings.
It allows for different modeling approaches: numerical regression, ordinal regression, and binary classification.

The main class, MoodModelingManager, handles the entire process:

- Preparing different (X, y) pairs from input dataframes
- Training and evaluating multiple modeling pipelines
- Gathering comprehensive validation statistics
- Providing a final model that outputs mood scores in the [0, 1] range

These mood scores are ordinally aligned with training scores, where higher values indicate
stronger presence of the target semantic attribute.

### Functions

| [`class_to_path`](#mood.mood_modeling.class_to_path)(obj)                              | Convert an object to a string path.                                     |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`path_to_class`](#mood.mood_modeling.path_to_class)(path)                             | Convert a string path to an object.                                     |
| [`x_and_y_with_boolean_y`](#mood.mood_modeling.x_and_y_with_boolean_y)(data, x_col, y_col, ...) | Get the x and y values from a dataframe, converting y to binary values. |

### Classes

| [`DimensionReducer`](#mood.mood_modeling.DimensionReducer)([max_dims])                  | Transformer to reduce dimensionality by selecting first n dimensions.                                                             |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [`MoodEstimator`](#mood.mood_modeling.MoodEstimator)(model[, max_dims, ...])         | Custom estimator that wraps an underlying model and provides methods to transform predictions to mood scores in the [0, 1] range. |
| [`MoodModelingManager`](#mood.mood_modeling.MoodModelingManager)(df[, embedding_col, ...]) | Manager class for building and evaluating mood models.                                                                            |
| [`OrdinalRandomForest`](#mood.mood_modeling.OrdinalRandomForest)([ordinal_classes])        | Random Forest customized for ordinal regression.                                                                                  |

### *class* mood.mood_modeling.DimensionReducer(max_dims=None)

Bases: `BaseEstimator`, `TransformerMixin`

Transformer to reduce dimensionality by selecting first n dimensions.

This is useful for embedding vectors where the first dimensions typically
contain the most important information.

#### fit(X, y=None)

Fit method (no-op as this is a simple feature selector).

* **Parameters:**
  * **X** (`ndarray`) – Input features
  * **y** (`ndarray` | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Target values (not used)
* **Return type:**
  [`DimensionReducer`](#mood.mood_modeling.DimensionReducer)
* **Returns:**
  Self

#### transform(X)

Transform X by selecting first max_dims dimensions.

* **Parameters:**
  **X** (`ndarray`) – Input features
* **Return type:**
  `ndarray`
* **Returns:**
  Reduced feature matrix

### *class* mood.mood_modeling.MoodEstimator(model, max_dims=None, output_transform='sigmoid', threshold=None, data_type='numerical')

Bases: `BaseEstimator`, `TransformerMixin`

Custom estimator that wraps an underlying model and provides methods to transform
predictions to mood scores in the [0, 1] range.

#### fit(X, y)

Fit the model to the data.

* **Parameters:**
  * **X** (`ndarray`) – Input features
  * **y** (`ndarray`) – Target values
* **Return type:**
  [`MoodEstimator`](#mood.mood_modeling.MoodEstimator)
* **Returns:**
  Self

#### predict(X)

Make predictions using the underlying model.

* **Parameters:**
  **X** (`ndarray`) – Input features
* **Return type:**
  `ndarray`
* **Returns:**
  Raw model predictions

#### predict_proba(X)

Predict class probabilities if the underlying model supports it.

* **Parameters:**
  **X** (`ndarray`) – Input features
* **Return type:**
  `ndarray`
* **Returns:**
  Class probabilities if available, otherwise transformed predictions

#### transform(X)

Transform input features to mood scores in [0, 1] range.

* **Parameters:**
  **X** (`ndarray`) – Input features
* **Return type:**
  `ndarray`
* **Returns:**
  Mood scores in [0, 1] range

### *class* mood.mood_modeling.MoodModelingManager(df, embedding_col='embedding', score_col='score', test_size=0.2, random_state=42, models=None, verbose=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Manager class for building and evaluating mood models.

This class handles:

- Data preparation for different model types
- Training and evaluation of multiple modeling pipelines
- Collection of validation statistics
- Final model selection and fitting

#### cross_validate_models(n_splits=5, metrics=None)

Perform cross-validation on all models to get more stable performance metrics.

* **Parameters:**
  * **n_splits** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of cross-validation folds
  * **metrics** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Set of metric names to compute (defaults to model-appropriate metrics)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary of cross-validation results for all models

#### fit_final_models()

Fit final models on the entire dataset.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary of fitted models

#### *classmethod* from_arrays(X, y, test_size=0.2, random_state=42, models=None, verbose=1)

Create a MoodModelingManager directly from numpy arrays.

* **Parameters:**
  * **X** (`ndarray`) – Array of embeddings, shape (n_samples, n_features)
  * **y** (`ndarray`) – Array of scores, shape (n_samples,)
  * **test_size** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Proportion of data to use for testing
  * **random_state** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
  * **models** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Dictionary of modeling pipelines (uses default_models if None)
  * **verbose** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Level of verbosity
* **Return type:**
  [`MoodModelingManager`](#mood.mood_modeling.MoodModelingManager)
* **Returns:**
  MoodModelingManager instance

### Example

```pycon
>>> manager = MoodModelingManager.from_arrays(X, y, verbose=2)
>>> results = manager.train_and_evaluate()
```

#### get_best_model(metric='spearman', data_type=None, use_cv=False)

Get the best model based on a specific metric.

* **Parameters:**
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Metric to use for comparison
  * **data_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Filter by data type (optional)
  * **use_cv** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to use cross-validation results for comparison (if available)
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`MoodEstimator`](#mood.mood_modeling.MoodEstimator)]
* **Returns:**
  Tuple of (model_name, model)

#### get_model_summary(use_cv=False, \*, egress=<class 'pandas.DataFrame'>)

Get a summary of all models and their performance.

* **Parameters:**
  **use_cv** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to use cross-validation results (if available)
* **Returns:**
  DataFrame summarizing model performance

#### predict_mood(X, model_name=None)

Predict mood scores for new data.

* **Parameters:**
  * **X** (`ndarray`) – Input features
  * **model_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Name of model to use (uses best model if None)
* **Return type:**
  `ndarray`
* **Returns:**
  Array of mood scores in [0, 1] range

#### prepare_binary_data(threshold=None, positive_labels=None, negative_labels=None)

Prepare binary data from the dataframe.

* **Parameters:**
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Score threshold above which is considered positive
  * **positive_labels** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of scores considered positive
  * **negative_labels** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – List of scores considered negative
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]
* **Returns:**
  Tuple of (X, y) for binary classification

#### prepare_numerical_data(normalize=True)

Prepare numerical data from the dataframe.

* **Parameters:**
  **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to normalize scores to [0, 1]
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]
* **Returns:**
  Tuple of (X, y) for numerical regression

#### prepare_ordinal_data()

Prepare ordinal data from the dataframe.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]
* **Returns:**
  Tuple of (X, y) for ordinal regression

#### train_and_evaluate()

Train and evaluate all models using a single train/test split, collecting performance metrics.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Returns:**
  Dictionary of results for all models

### *class* mood.mood_modeling.OrdinalRandomForest(ordinal_classes=None, \*\*kwargs)

Bases: `RandomForestRegressor`

Random Forest customized for ordinal regression.

Predicts continuous values first, then maps them to the closest ordinal class.

#### fit(X, y)

Build a forest of trees from the training set (X, y).

* **Parameters:**
  * **X** ( *{array-like* *,* *sparse matrix}* *of* *shape* *(**n_samples* *,* *n_features* *)*) – The training input samples. Internally, its dtype will be converted
    to `dtype=np.float32`. If a sparse matrix is provided, it will be
    converted into a sparse `csc_matrix`.
  * **y** (*array-like* *of* *shape* *(**n_samples* *,* *) or*  *(**n_samples* *,* *n_outputs* *)*) – The target values (class labels in classification, real numbers in
    regression).
  * **sample_weight** (*array-like* *of* *shape* *(**n_samples* *,* *)* *,* *default=None*) – Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. In the case of
    classification, splits are also ignored if they would result in any
    single class carrying a negative weight in either child node.
* **Returns:**
  **self** – Fitted estimator.
* **Return type:**
  [*object*](https://docs.python.org/3/builtins/functions.html#object)

#### predict(X)

Predict regression target for X.

The predicted regression target of an input sample is computed as the
mean predicted regression targets of the trees in the forest.

* **Parameters:**
  **X** ( *{array-like* *,* *sparse matrix}* *of* *shape* *(**n_samples* *,* *n_features* *)*) – The input samples. Internally, its dtype will be converted to
  `dtype=np.float32`. If a sparse matrix is provided, it will be
  converted into a sparse `csr_matrix`.
* **Returns:**
  **y** – The predicted values.
* **Return type:**
  ndarray of shape (n_samples,) or (n_samples, n_outputs)

#### set_fit_request()

No-op.

Calling this method has no effect.

* **Returns:**
  **self** – The updated object.
* **Return type:**
  [`OrdinalRandomForest`](#mood.mood_modeling.OrdinalRandomForest)

#### set_score_request(, sample_weight='$UNCHANGED$')

Configure whether metadata should be requested to be passed to the `score` method.

Note that this method is only relevant when this estimator is used as a
sub-estimator within a meta-estimator and metadata routing is enabled
with `enable_metadata_routing=True` (see `sklearn.set_config()`).
Please check the User Guide on how the routing
mechanism works.

The options for each parameter are:

- `True`: metadata is requested, and passed to `score` if provided. The request is ignored if metadata is not provided.
- `False`: metadata is not requested and the meta-estimator will not pass it to `score`.
- `None`: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
- `str`: metadata should be passed to the meta-estimator with this given alias instead of the original name.

The default (`sklearn.utils.metadata_routing.UNCHANGED`) retains the
existing request. This allows you to change the request for some
parameters and not others.

#### Versionadded
Added in version 1.3.

* **Parameters:**
  **sample_weight** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`None`](https://docs.python.org/3/builtins/constants.html#None), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Metadata routing for `sample_weight` parameter in `score`.
* **Returns:**
  **self** – The updated object.
* **Return type:**
  [`OrdinalRandomForest`](#mood.mood_modeling.OrdinalRandomForest)

### mood.mood_modeling.class_to_path(obj)

Convert an object to a string path.
This is useful for serialization and deserialization of objects.

```pycon
>>> from sklearn.linear_model import LogisticRegression
>>> path = class_to_path(LogisticRegression)
>>> path
'sklearn.linear_model._logistic.LogisticRegression'
>>> path_to_class(path) == LogisticRegression
True
```

### mood.mood_modeling.path_to_class(path)

Convert a string path to an object.

### mood.mood_modeling.x_and_y_with_boolean_y(data, x_col, y_col, y_is_true_vals, y_is_false_vals=None)

Get the x and y values from a dataframe, converting y to binary values.
The output y elements will be 1 if and only if the y_col value is in y_is_true_vals.

* **Parameters:**
  * **data** (`DataFrame`) – The dataframe to get the x and y values from
  * **x_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the column to use as x
  * **y_col** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the column to use as y
  * **y_is_true_vals** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The values in y_col that are considered 1
  * **y_is_false_vals** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The values in y_col that are considered 0 (optional)
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]
* **Returns:**
  Tuple of (X, y) where X contains features and y contains binary labels
