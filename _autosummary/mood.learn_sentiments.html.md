# mood.learn_sentiments

Learning sentiments from data.

### Functions

| [`fit_logistic_classifier`](#mood.learn_sentiments.fit_logistic_classifier)(X, y, \*[, ...])   | Fits a logistic regression model on embedding data for binary classification without feature scaling.   |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [`get_binary_y`](#mood.learn_sentiments.get_binary_y)(y, target_y)                  | Convert a multi-class target array to a binary array for a specific target class.                       |
| [`predict_with_model`](#mood.learn_sentiments.predict_with_model)(model, X)               | Helper function to make predictions with a fitted model.                                                |

### Classes

| [`ModelStatistics`](#mood.learn_sentiments.ModelStatistics)([accuracy, precision, ...])   | Container for model performance statistics   |
|------------------------------------------------------------------------------------------------|----------------------------------------------|

### *class* mood.learn_sentiments.ModelStatistics(accuracy=None, precision=None, recall=None, f1=None, roc_auc=None, confusion_matrix=None, classification_report=None, train_test_performed=False, training_time=None, test_size=None, random_state=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Container for model performance statistics

### mood.learn_sentiments.fit_logistic_classifier(X, y, , test_size=0.2, random_state=42, c_value=1.0, max_iter=1000, solver='lbfgs', penalty='l2', class_weight=None, fit_intercept=True, skip_validation=False, verbose=True)

Fits a logistic regression model on embedding data for binary classification without feature scaling.

### Parameters

y
: Binary target values (0/1 or True/False).

test_size
: Proportion of the dataset to include in the test split.

random_state
: Controls the shuffling applied to the data before applying the split.

c_value
: Inverse of regularization strength; smaller values specify stronger regularization.

max_iter
: Maximum number of iterations taken for the solvers to converge.

solver
: Algorithm to use in the optimization problem (‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’).

penalty
: Specify the norm used in the penalization (‘l1’, ‘l2’, ‘elasticnet’, None).

class_weight
: Weights associated with classes. If ‘balanced’, uses class frequencies.

fit_intercept
: Whether to fit an intercept term in the model.

skip_validation
: If True, skips train/test validation and fits model on all data.

verbose
: If True, prints progress and results.

### Returns

stats
: Object containing performance statistics.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`LogisticRegression`, [`ModelStatistics`](#mood.learn_sentiments.ModelStatistics)]

### mood.learn_sentiments.get_binary_y(y, target_y)

Convert a multi-class target array to a binary array for a specific target class.

### mood.learn_sentiments.predict_with_model(model, X)

Helper function to make predictions with a fitted model.

### Parameters

X
: Feature vectors to predict on.

### Returns

y_prob
: Probability estimates for class 1.
