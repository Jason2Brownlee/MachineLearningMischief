# Test Set Overfitting

> Optimizing a model for its performance on a "hold out" test set.

## Description
This is typically called "test set overfitting" or "overfitting to the test set."

It occurs when practitioners repeatedly tune their model based on test set performance, effectively making the test set act as a second training set. This violates the fundamental principle that the test set should only be used for final evaluation.

Sometimes it's also referred to as "test set adaption" or "inappropriate test set optimization." In more formal academic literature, it might be described as "compromising test set independence through iterative optimization."

This is different from test set leakage (where information flows from test to train inadvertently) because in this case, there's intentional optimization using test set feedback. It's particularly problematic because it gives an overly optimistic estimate of model performance and doesn't reflect how the model would perform on truly unseen data.

This is why many researchers advocate for using a three-way split (train/validation/test) or holding out a completely separate test set that is only used once for final evaluation, with all intermediate optimization done using cross-validation on the training data.

## Example

```python
# Import necessary libraries
from sklearn.datasets import make_classification  # For generating a synthetic classification dataset
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.ensemble import RandomForestClassifier  # High-capacity model
from sklearn.metrics import accuracy_score  # For model evaluation
from itertools import product  # For generating all combinations of hyperparameters

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define possible values for hyperparameters
n_estimators_options = [10, 50, 100, 200]
max_depth_options = [5, 10, 15, 20]

# Generate all combinations of hyperparameters
configurations = list(product(n_estimators_options, max_depth_options))

# Dictionary to store test set performance for each configuration
test_set_performance = {}

# Variable to track the best configuration so far
best_config_so_far = None
best_accuracy_so_far = 0

# Loop through each configuration
for n_estimators, max_depth in configurations:
    # Create the model with the current configuration
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store the performance on the test set
    test_set_performance[f"n_estimators={n_estimators}, max_depth={max_depth}"] = accuracy

    # Update and display progress
    if accuracy > best_accuracy_so_far:
        best_config_so_far = (n_estimators, max_depth)
        best_accuracy_so_far = accuracy
    print(f"cfg: n_estimators={n_estimators}, max_depth={max_depth}, Accuracy: {accuracy:.4f} " + f"(Best: {best_accuracy_so_far:.4f})")

# Print the final best configuration and its test set accuracy
print(f"Final Best Configuration: n_estimators={best_config_so_far[0]}, max_depth={best_config_so_far[1]}, Test Set Accuracy: {best_accuracy_so_far:.4f}")
```

Example Output:

```text
cfg: n_estimators=10, max_depth=5, Accuracy: 0.8400 (Best: 0.8400)
cfg: n_estimators=10, max_depth=10, Accuracy: 0.8800 (Best: 0.8800)
cfg: n_estimators=10, max_depth=15, Accuracy: 0.8850 (Best: 0.8850)
cfg: n_estimators=10, max_depth=20, Accuracy: 0.8750 (Best: 0.8850)
cfg: n_estimators=50, max_depth=5, Accuracy: 0.8750 (Best: 0.8850)
cfg: n_estimators=50, max_depth=10, Accuracy: 0.9100 (Best: 0.9100)
cfg: n_estimators=50, max_depth=15, Accuracy: 0.8900 (Best: 0.9100)
cfg: n_estimators=50, max_depth=20, Accuracy: 0.9000 (Best: 0.9100)
cfg: n_estimators=100, max_depth=5, Accuracy: 0.8800 (Best: 0.9100)
cfg: n_estimators=100, max_depth=10, Accuracy: 0.9000 (Best: 0.9100)
cfg: n_estimators=100, max_depth=15, Accuracy: 0.9000 (Best: 0.9100)
cfg: n_estimators=100, max_depth=20, Accuracy: 0.9000 (Best: 0.9100)
cfg: n_estimators=200, max_depth=5, Accuracy: 0.8700 (Best: 0.9100)
cfg: n_estimators=200, max_depth=10, Accuracy: 0.8750 (Best: 0.9100)
cfg: n_estimators=200, max_depth=15, Accuracy: 0.8800 (Best: 0.9100)
cfg: n_estimators=200, max_depth=20, Accuracy: 0.8800 (Best: 0.9100)
Final Best Configuration: n_estimators=50, max_depth=10, Test Set Accuracy: 0.9100
```