# p-Hacking

> Repeating a statistical hypothesis test until a significant result is achieved.

P-hacking (also called [data dredging](https://en.wikipedia.org/wiki/Data_dredging) or data fishing) is the practice of manipulating data analysis to find statistically significant results (p < 0.05) even when there isn't a true effect.

It exploits the fact that if you analyze data in enough different ways, you're likely to find "significant" results by chance.

Common p-hacking techniques include:
1. Running many statistical tests and only reporting the ones that show significance
2. Selectively removing outliers until you get significance
3. Collecting more data until significance appears
4. Testing multiple variables and only reporting the significant ones
5. Trying different statistical methods until finding one that gives significance
6. Arbitrarily splitting data into subgroups to find significant effects

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