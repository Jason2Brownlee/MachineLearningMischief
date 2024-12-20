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
