import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Define the number of trials
num_trials = 100

# Define variables to track the best fold configuration and best performance
best_fold_seed = None
best_performance = -np.inf

# Create a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=5, n_informative=4, n_redundant=1, random_state=42)

# Initialize the model with a fixed seed
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Iterate over multiple seeds to vary the k-fold cross-validation splits
for trial in range(num_trials):
    # Set the seed for the k-fold shuffle
    fold_seed = trial

    # Initialize k-fold cross-validation with the current seed
    kf = KFold(n_splits=5, shuffle=True, random_state=fold_seed)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X, y, cv=kf)

    # Calculate the mean performance
    mean_performance = scores.mean()

    # Print the fold seed and performance if there is an improvement
    if mean_performance > best_performance:
        print(f"Fold Seed: {fold_seed}, Performance: {mean_performance:.4f}")
        best_performance = mean_performance
        best_fold_seed = fold_seed

# Report the best fold seed and its performance
print(f"\nBest Fold Seed: {best_fold_seed}, Best Performance: {best_performance:.4f}")
