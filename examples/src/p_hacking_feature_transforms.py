# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=350, n_features=10, n_informative=2, n_redundant=8, random_state=42)

# Define a high-capacity machine learning model
model = LogisticRegression(max_iter=1000, random_state=42)

# Define a k-fold cross-validation strategy with a fixed seed
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate the model on the dataset using k-fold cross-validation (baseline without transformations)
baseline_scores = cross_val_score(model, X, y, cv=kfold)
baseline_mean = np.mean(baseline_scores)

# Set up parameters for p-hacking
p_threshold = 0.05  # Threshold for statistical significance
transformations = ["none", "log", "sqrt", "square"]  # Possible transformations to test
significant_result_found = False

# Loop through trials with different feature transformations
for transform in transformations:
    # Apply the selected transformation to the features
    if transform == "log":
        X_transformed = np.log(np.abs(X) + 1)  # Avoid log(0) or negative numbers
    elif transform == "sqrt":
        X_transformed = np.sqrt(np.abs(X))  # Avoid sqrt of negative numbers
    elif transform == "square":
        X_transformed = X ** 2
    else:  # "none"
        X_transformed = X

    # Evaluate the model with k-fold cross-validation on transformed features
    trial_scores = cross_val_score(model, X_transformed, y, cv=kfold)
    trial_mean = np.mean(trial_scores)

    # Perform a t-test to compare means
    t_stat, p_value = ttest_ind(baseline_scores, trial_scores)
    significant = p_value < p_threshold

    # Report progress
    print(f'Transform: {transform} mean: {trial_mean:.3f} (base: {baseline_mean:.3f}), p-value: {p_value:.3f}')
    if significant:
        print('\tSignificant difference')

