# Test Harness Hacking: Hill Climb Cross-Valuation Performance

> Excessively adapt a model for cross-validation performance.

## Description

This occurs when a model is excessively tuned to maximize performance during cross-validation.

Using a robust test harness like k-fold cross-validation, repeated iterations of model adjustments are made to improve the average score on test folds.

However, this over-adaptation leads to overfitting on the training dataset. The model becomes too specialized to the patterns in the cross-validation splits, losing generalizability.

The issue often arises when the number of improvement trials exceeds the size of the training dataset, creating a misleading sense of success.

While cross-validation metrics may look impressive, performance on a separate hold-out test set deteriorates. This approach sacrifices real-world accuracy for temporary gains during validation, undermining the model's reliability.

## Example

In this example, model is excessively tuned to optimize cross-validation performance at the expense of generalizability.

Using a synthetic classification dataset, a Random Forest model is repeatedly optimized over 100 trials through stochastic hill climbing, adjusting the hyperparameters `n_estimators` and `max_depth` to improve the mean k-fold cross-validation accuracy.

As the optimization progresses, the cross-validation score steadily improves, but the hold-out test performance often plateaus or deteriorates, highlighting overfitting to the cross-validation splits.

The code visualizes this divergence between cross-validation and test performance, illustrating how focusing excessively on cross-validation metrics can undermine the model's real-world applicability.

```python
# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=200, n_features=30, n_informative=5, n_redundant=25, random_state=42
)

# Create a train/test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up k-fold cross-validation for the training set
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize variables for hill climbing and tracking performance
n_trials = 100  # Number of optimization trials
best_params = {"n_estimators": 10, "max_depth": 2}  # Initial hyperparameters
cv_scores = []  # Track cross-validation scores
test_scores = []  # Track hold-out test scores

# Define a stochastic hill climbing procedure for hyperparameter tuning
for trial in range(n_trials):
    # Create a model with current best parameters
    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42
    )

    # Evaluate model using k-fold cross-validation
    cv_score = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy"))

    # Fit the model on the entire training set and evaluate on the hold-out test set
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Record scores
    cv_scores.append(cv_score)
    test_scores.append(test_score)

    # Print trial results
    print(f"Trial {trial+1}: CV Mean Score={cv_score:.4f}, Test Score={test_score:.4f}")

    # Propose a random perturbation of the hyperparameters
    new_params = {
        "n_estimators": best_params["n_estimators"] + np.random.randint(-10, 11),
        "max_depth": best_params["max_depth"] + np.random.randint(-1, 2)
    }
    new_params["n_estimators"] = max(1, new_params["n_estimators"])  # Ensure valid value
    new_params["max_depth"] = max(1, new_params["max_depth"])  # Ensure valid value

    # Evaluate new parameters
    new_model = RandomForestClassifier(
        n_estimators=new_params["n_estimators"], max_depth=new_params["max_depth"], random_state=42
    )
    new_cv_score = np.mean(cross_val_score(new_model, X_train, y_train, cv=kf, scoring="accuracy"))

    # Update the best parameters if the new score is better
    if new_cv_score > cv_score:
        best_params = new_params

# Plot the cross-validation and hold-out test scores over trials
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_trials + 1), cv_scores, label="Cross-Validation Score")
plt.plot(range(1, n_trials + 1), test_scores, label="Hold-Out Test Score")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.title("Model Performance: Cross-Validation vs Hold-Out Test")
plt.legend()
plt.show()

# Print final performance metrics
final_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42
)
final_model.fit(X_train, y_train)
final_cv_score = np.mean(cross_val_score(final_model, X_train, y_train, cv=kf, scoring="accuracy"))
final_test_score = final_model.score(X_test, y_test)
print(f"Final Model: CV Mean Score={final_cv_score:.4f}, Test Score={final_test_score:.4f}")
```

Example Output:

![](/pics/test_harness_hacking_hill_climbing_performance.png)

```text
Trial 1: CV Mean Score=0.7123, Test Score=0.7000
Trial 2: CV Mean Score=0.7247, Test Score=0.7250
Trial 3: CV Mean Score=0.7247, Test Score=0.7250
Trial 4: CV Mean Score=0.7247, Test Score=0.7250
Trial 5: CV Mean Score=0.7247, Test Score=0.7250
...
Trial 95: CV Mean Score=0.8371, Test Score=0.7750
Trial 96: CV Mean Score=0.8371, Test Score=0.7750
Trial 97: CV Mean Score=0.8371, Test Score=0.7750
Trial 98: CV Mean Score=0.8371, Test Score=0.7750
Trial 99: CV Mean Score=0.8371, Test Score=0.7750
Trial 100: CV Mean Score=0.8371, Test Score=0.7750
```