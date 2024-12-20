# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=5, random_state=42)

# Define a classifier
model = LogisticRegression(random_state=42, max_iter=1000)

# Define a k-fold cross-validation strategy with a fixed seed
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model on the full dataset using k-fold cross-validation
baseline_scores = cross_val_score(model, X, y, cv=kfold)
baseline_mean = np.mean(baseline_scores)
print(f'Base result: {baseline_mean:.3f}')

# Set up parameters for p-hacking
p_threshold = 0.05  # Threshold for statistical significance
max_trials = 1000    # Maximum number of sampling strategies to test
sample_size = int(0.5 * X.shape[0])

# Perform selective sampling and evaluate subsets
for trial in range(max_trials):
    # Randomly select a subset of samples
    np.random.seed(trial + 1)
    sample_indices = np.random.choice(range(X.shape[0]), size=sample_size, replace=False)
    X_subset, y_subset = X[sample_indices], y[sample_indices]

    # Evaluate the model on the sampled subset using cross-validation
    trial_scores = cross_val_score(model, X_subset, y_subset, cv=kfold)
    trial_mean = np.mean(trial_scores)
    better = trial_mean > baseline_mean

    # Perform a t-test to compare means
    t_stat, p_value = ttest_ind(baseline_scores, trial_scores)
    significant = p_value < p_threshold

    # Report progress
    print(f'{trial+1}, Result: {trial_mean:.3f}, Better: {better}, p-value: {p_value:.3f} Significant: {significant}')

    # Stop if better and significant
    if better and significant:
        break
