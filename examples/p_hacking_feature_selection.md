# p-Hacking Feature Selection

> Vary feature subsets of a dataset in order to fit a model with significantly better performance.

## Description

P-Hacking Feature Selection involves manipulating the feature subset of a dataset to artificially improve model performance. By testing multiple combinations of features and selecting those that yield the best results, practitioners may achieve statistically significant outcomes that are misleading or unreliable.

This practice skews the model's apparent accuracy and risks overfitting to the training data, making it less generalizable to new datasets. While it might seem like optimization, it violates the principles of sound model development and evaluation.

Data scientists should avoid this anti-pattern by adhering to rigorous validation techniques, such as using holdout datasets or cross-validation, and focusing on domain-relevant feature selection methods. This ensures model performance reflects true predictive power rather than manipulated outcomes.

## Example

```python
# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_ind

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=2, n_redundant=8, random_state=42)

# Define a classifier
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Define a k-fold cross-validation strategy with a fixed seed
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model on the full dataset using k-fold cross-validation
baseline_scores = cross_val_score(model, X, y, cv=kfold)
baseline_mean = np.mean(baseline_scores)
print(f'Base result: {baseline_mean:.3f}')

# Set up parameters for p-hacking
p_threshold = 0.05  # Threshold for statistical significance
max_trials = 1000    # Maximum number of feature subsets to test
num_features = X.shape[1]

# Perform selective feature subset selection and evaluation
for trial in range(max_trials):
    # Randomly select a subset of features
    np.random.seed(trial + 1)
    selected_features = np.random.choice(range(num_features), size=np.random.randint(1, num_features + 1), replace=False)
    X_subset = X[:, selected_features]

    # Evaluate the model on the selected feature subset using cross-validation
    trial_scores = cross_val_score(model, X_subset, y, cv=kfold)
    trial_mean = np.mean(trial_scores)
    better = trial_mean > baseline_mean

    # Perform a t-test to compare means
    t_stat, p_value = ttest_ind(baseline_scores, trial_scores)
    significant = p_value < p_threshold

    # Report progress
    print(f'{trial+1}, Features: {selected_features}, Result: {trial_mean:.3f}, Better: {better}, p-value: {p_value:.3f}, Significant: {significant}')

    # Stop if better and significant
    if better and significant:
        print("P-hacked subset identified!")
        break
```

Example Output:

```text
Base result: 0.944
1, Features: [2 3 4 9 1 6], Result: 0.956, Better: True, p-value: 0.166, Significant: False
2, Features: [4 1 9 5 0 7 2 3 6], Result: 0.950, Better: True, p-value: 0.446, Significant: False
3, Features: [5 4 1 2 8 6 7 0 3], Result: 0.948, Better: True, p-value: 0.587, Significant: False
4, Features: [3 4 6 9 8 2 7 0], Result: 0.950, Better: True, p-value: 0.347, Significant: False
5, Features: [5 8 2 3], Result: 0.950, Better: True, p-value: 0.402, Significant: False
...
54, Features: [5 3 9 4 8 6], Result: 0.956, Better: True, p-value: 0.135, Significant: False
55, Features: [6 4 0 2 1 3 9 7], Result: 0.950, Better: True, p-value: 0.621, Significant: False
56, Features: [9 8 5 6 1 7], Result: 0.940, Better: False, p-value: 0.740, Significant: False
57, Features: [3 9 1 4 8 2 0], Result: 0.958, Better: True, p-value: 0.058, Significant: False
58, Features: [4 2 8 9], Result: 0.962, Better: True, p-value: 0.022, Significant: True
P-hacked subset identified!
```