# p-Hacking the Learning Algorithm

> Vary the random numbers used by a learning algorithm in order to get a significantly better result.

## Description

p-Hacking the learning algorithm involves tweaking the random seed or initialization of a machine learning model to artificially produce **significantly** better performance metrics.

This approach manipulates results by repeatedly running the algorithm with different random values until a favorable outcome is achieved. While it may improve metrics like accuracy or precision, the model’s actual robustness and generalizability often suffer.

This practice undermines the reliability of machine learning results by focusing on chance improvements rather than meaningful insights or genuine model quality. It is considered an anti-pattern because it misrepresents the model’s true performance and can lead to overfitting or poor performance on unseen data.

## Example

Here, we are evaluating the "same" model on the same data, only varying the random number seed (e.g. vary the learning algorithm slightly).

There should be no statistically significant difference between runs, but we continue the trial until a difference is found due to high-variance/randomness.

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=350, n_features=10, n_informative=2, n_redundant=8, random_state=42)

# Define a high-capacity machine learning model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Define a k-fold cross-validation strategy with a fixed seed
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model on the dataset using k-fold cross-validation
baseline_scores = cross_val_score(model, X, y, cv=kfold)
baseline_mean = np.mean(baseline_scores)

# Set up parameters for p-hacking
p_threshold = 0.05  # Threshold for statistical significance
max_trials = 1000    # Maximum number of trials to perform
significant_result_found = False

# Loop through trials with different random seeds
for trial in range(max_trials):
    # Use a new random seed for the model
    seed = trial + 100
    model = RandomForestClassifier(n_estimators=10, random_state=seed)

    # Evaluate the model with k-fold cross-validation
    trial_scores = cross_val_score(model, X, y, cv=kfold)
    trial_mean = np.mean(trial_scores)

    # Perform a t-test to compare means
    t_stat, p_value = ttest_ind(baseline_scores, trial_scores)

    # Check if the p-value is below the significance threshold
    if p_value < p_threshold:
        significant_result_found = True
        print(f"Significant difference found on trial {trial+1}")
        print(f"Baseline mean: {baseline_mean:.4f}, Trial mean: {trial_mean:.4f}, p-value: {p_value:.4f}")
        break
    else:
        print(f"No significant difference found yet, trial {trial+1}, p-value: {p_value:.4f}")

# Report if no significant result was found within the maximum trials
if not significant_result_found:
    print("No significant result found after maximum trials.")
```

Example Output:

```text
No significant difference found yet, trial 1, p-value: 0.7245
No significant difference found yet, trial 2, p-value: 0.4860
No significant difference found yet, trial 3, p-value: 0.8028
No significant difference found yet, trial 4, p-value: 0.5447
No significant difference found yet, trial 5, p-value: 1.0000
...
No significant difference found yet, trial 80, p-value: 0.3972
No significant difference found yet, trial 81, p-value: 1.0000
No significant difference found yet, trial 82, p-value: 0.7245
No significant difference found yet, trial 83, p-value: 1.0000
No significant difference found yet, trial 84, p-value: 0.7404
No significant difference found yet, trial 85, p-value: 1.0000
No significant difference found yet, trial 86, p-value: 0.7245
No significant difference found yet, trial 87, p-value: 0.7707
Significant difference found on trial 88
Baseline mean: 0.9743, Trial mean: 0.9886, p-value: 0.0462
```
