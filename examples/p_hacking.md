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

Example Output:

```text
No significant difference found yet, trial 1, p-value: 0.7245
No significant difference found yet, trial 2, p-value: 0.4860
No significant difference found yet, trial 3, p-value: 0.8028
No significant difference found yet, trial 4, p-value: 0.5447
No significant difference found yet, trial 5, p-value: 1.0000
No significant difference found yet, trial 6, p-value: 0.7707
No significant difference found yet, trial 7, p-value: 0.2897
No significant difference found yet, trial 8, p-value: 0.7707
No significant difference found yet, trial 9, p-value: 0.2073
No significant difference found yet, trial 10, p-value: 0.3972
No significant difference found yet, trial 11, p-value: 0.8327
No significant difference found yet, trial 12, p-value: 0.4236
No significant difference found yet, trial 13, p-value: 1.0000
No significant difference found yet, trial 14, p-value: 0.1950
No significant difference found yet, trial 15, p-value: 0.4675
No significant difference found yet, trial 16, p-value: 0.2073
No significant difference found yet, trial 17, p-value: 0.7599
No significant difference found yet, trial 18, p-value: 0.2897
No significant difference found yet, trial 19, p-value: 0.2897
No significant difference found yet, trial 20, p-value: 0.7404
No significant difference found yet, trial 21, p-value: 0.7223
No significant difference found yet, trial 22, p-value: 1.0000
No significant difference found yet, trial 23, p-value: 0.5447
No significant difference found yet, trial 24, p-value: 0.7599
No significant difference found yet, trial 25, p-value: 0.4194
No significant difference found yet, trial 26, p-value: 0.8287
No significant difference found yet, trial 27, p-value: 0.7707
No significant difference found yet, trial 28, p-value: 0.2897
No significant difference found yet, trial 29, p-value: 0.7245
No significant difference found yet, trial 30, p-value: 0.2897
No significant difference found yet, trial 31, p-value: 0.8089
No significant difference found yet, trial 32, p-value: 1.0000
No significant difference found yet, trial 33, p-value: 0.2897
No significant difference found yet, trial 34, p-value: 0.4937
No significant difference found yet, trial 35, p-value: 0.1950
No significant difference found yet, trial 36, p-value: 1.0000
No significant difference found yet, trial 37, p-value: 1.0000
No significant difference found yet, trial 38, p-value: 0.5447
No significant difference found yet, trial 39, p-value: 0.3589
No significant difference found yet, trial 40, p-value: 0.6406
No significant difference found yet, trial 41, p-value: 0.6454
No significant difference found yet, trial 42, p-value: 0.7599
No significant difference found yet, trial 43, p-value: 1.0000
No significant difference found yet, trial 44, p-value: 1.0000
No significant difference found yet, trial 45, p-value: 0.5796
No significant difference found yet, trial 46, p-value: 0.6666
No significant difference found yet, trial 47, p-value: 0.8089
No significant difference found yet, trial 48, p-value: 0.2897
No significant difference found yet, trial 49, p-value: 0.8327
No significant difference found yet, trial 50, p-value: 0.6666
No significant difference found yet, trial 51, p-value: 0.8028
No significant difference found yet, trial 52, p-value: 0.8327
No significant difference found yet, trial 53, p-value: 0.5346
No significant difference found yet, trial 54, p-value: 0.7707
No significant difference found yet, trial 55, p-value: 1.0000
No significant difference found yet, trial 56, p-value: 1.0000
No significant difference found yet, trial 57, p-value: 0.6454
No significant difference found yet, trial 58, p-value: 1.0000
No significant difference found yet, trial 59, p-value: 0.5447
No significant difference found yet, trial 60, p-value: 0.7845
No significant difference found yet, trial 61, p-value: 0.7924
No significant difference found yet, trial 62, p-value: 0.6666
No significant difference found yet, trial 63, p-value: 1.0000
No significant difference found yet, trial 64, p-value: 0.8028
No significant difference found yet, trial 65, p-value: 0.1411
No significant difference found yet, trial 66, p-value: 0.7245
No significant difference found yet, trial 67, p-value: 0.5447
No significant difference found yet, trial 68, p-value: 0.6666
No significant difference found yet, trial 69, p-value: 0.6666
No significant difference found yet, trial 70, p-value: 0.2897
No significant difference found yet, trial 71, p-value: 0.4117
No significant difference found yet, trial 72, p-value: 0.5651
No significant difference found yet, trial 73, p-value: 0.3319
No significant difference found yet, trial 74, p-value: 0.8028
No significant difference found yet, trial 75, p-value: 0.1411
No significant difference found yet, trial 76, p-value: 1.0000
No significant difference found yet, trial 77, p-value: 0.3116
No significant difference found yet, trial 78, p-value: 0.2073
No significant difference found yet, trial 79, p-value: 0.2073
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