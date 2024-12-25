# Test Harness Hacking: Hill Climb Cross-Validation Test Folds

> Adapt predictions for each cross-validation test fold over repeated trials.

## Description

This involves exploiting k-fold cross-validation to artificially improve model performance.

The model adapts its predictions for each fold during cross-validation trials, fully utilizing the performance metric signal from the test folds. Over time, this "hill-climbing" process fine-tunes predictions specifically for the test folds, leading to near-perfect results within the cross-validation framework.

However, this method ignores the need for generalization to new data. When applied to a real holdout test set, the model's performance collapses, producing random or inaccurate predictions.

This practice is unrealistic and misleading, as it relies on overfitting to test folds rather than building a robust, generalizable model.

As such it provides an idealized worst case scenario of a data scientist overfitting the training dataset, in the face of a robust test harness using k-fold cross-validation.

## Example

This example starts by initializing random predictions for all data points in the training set and performs repeated trials.

Each trial consists of one full k-fold cross-validation pass. During each fold, after evaluating predictions on the test fold, the algorithm makes a single adaptation to the predictions to improve accuracy on that specific fold. These adaptations accumulate over trials, effectively "hill climbing" towards perfect predictions on the cross-validation folds.

However, because this process overfits predictions to the cross-validation setup, the resulting model fails to generalize. When evaluated on a holdout test set, it produces random, non-generalizable predictions, highlighting the misleading nature of this approach.

```python
# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize random predictions across all data points in the training set
predictions = np.random.choice(np.unique(y_train), size=len(X_train))

# Maximum number of trials
n_trials = 100

# Begin hill-climbing meta-algorithm
for trial in range(n_trials):
    print(f"Trial {trial + 1}/{n_trials}")

    # Initialize variables to track progress across folds
    fold_accuracies = []

    # Perform k-fold cross-validation
    for train_idx, test_idx in kfold.split(X_train):
        # Get test fold indices
        y_test_fold = y_train[test_idx]
        fold_predictions = predictions[test_idx]

        # Evaluate the current predictions on the test fold
        current_accuracy = accuracy_score(y_test_fold, fold_predictions)

        # Adapt predictions based on test fold performance (hill climbing)
        if current_accuracy < 1.0:  # If not perfect
            for i in range(len(test_idx)):
                idx = test_idx[i]
                if predictions[idx] != y_train[idx]:  # Fix one wrong prediction
                    predictions[idx] = y_train[idx]
                    break  # Stop after a single modification

        # Recalculate fold accuracy after adaptation
        updated_fold_predictions = predictions[test_idx]
        updated_accuracy = accuracy_score(y_test_fold, updated_fold_predictions)
        fold_accuracies.append(updated_accuracy)

    # Calculate and report average accuracy across all folds for this trial
    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average Accuracy Across Folds: {avg_accuracy:.4f}")

    # Stop trials if all folds achieve perfect accuracy
    if avg_accuracy == 1.0:
        print("All folds reached perfect accuracy. Stopping trials.")
        break

# Evaluate the "model" on the holdout test set
# Use random predictions for the holdout test set to simulate lack of generalization
test_predictions = np.random.choice(np.unique(y_train), size=len(y_test))
holdout_accuracy = accuracy_score(y_test, test_predictions)

# Report final results
print("\nFinal Results:")
print(f"Accuracy on holdout test set: {holdout_accuracy:.4f}")
```

Example Output:

```text
Trial 1/100
Average Accuracy Across Folds: 0.5188
Trial 2/100
Average Accuracy Across Folds: 0.5250
Trial 3/100
Average Accuracy Across Folds: 0.5312
Trial 4/100
Average Accuracy Across Folds: 0.5375
Trial 5/100
Average Accuracy Across Folds: 0.5437
...
Trial 79/100
Average Accuracy Across Folds: 0.9950
Trial 80/100
Average Accuracy Across Folds: 0.9975
Trial 81/100
Average Accuracy Across Folds: 0.9988
Trial 82/100
Average Accuracy Across Folds: 1.0000
All folds reached perfect accuracy. Stopping trials.

Final Results:
Accuracy on holdout test set: 0.4100
```