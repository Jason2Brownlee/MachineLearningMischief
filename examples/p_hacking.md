# p-Hacking

> Repeating a statistical hypothesis test until a significant result is achieved.

## Description

P-hacking is the practice of manipulating data analysis until you achieve a statistically significant result, typically to support a predetermined conclusion.

This approach involves running multiple statistical tests on the same dataset, selectively choosing which data points to include, or adjusting variables until achieving the desired p-value (typically < 0.05).

While it may seem tempting to keep testing until you get "significant" results, p-hacking invalidates the fundamental principles of statistical testing and leads to false discoveries.

The danger lies in increasing the likelihood of Type I errors (false positives) through multiple comparisons, making spurious correlations appear meaningful when they're actually due to random chance.

For new data scientists, this pattern often emerges unintentionally when there's pressure to find significant results or when dealing with stakeholder expectations for positive outcomes.

To avoid p-hacking, define your hypothesis and analysis plan before examining the data, use correction methods for multiple comparisons, and be transparent about all tests performed - including those that didn't yield significant results.

Remember that negative results are valid scientific outcomes and should be reported alongside positive findings to maintain research integrity.

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


## Further Reading

* [Data dredging](https://en.wikipedia.org/wiki/Data_dredging), Wikipedia.
* [The Extent and Consequences of P-Hacking in Science](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002106&), 2015.
* [Big little lies: a compendium and simulation of p-hacking strategies](https://royalsocietypublishing.org/doi/10.1098/rsos.220346), 2023.
