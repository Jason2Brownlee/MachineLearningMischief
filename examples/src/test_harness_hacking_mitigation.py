import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=200, n_features=30, n_informative=5, n_redundant=25, random_state=42
)

# Create a train/test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize result storage for experiments
results = []

# Define the study parameters
fold_range = [3, 5, 7, 10]  # 3 to 10 folds
repeat_range = [1, 3, 5]  # 1 to 10 repetitions
n_trials = 5  # Number of trials for each configuration

# Function for hill climbing optimization
def hill_climb(cv, X_train, y_train, X_test, y_test, n_hill_trials=100):
    best_params = {"n_estimators": 10, "max_depth": 2}
    best_cv_score = -1

    cv_scores = []
    holdout_scores = []

    for hill_trial in range(n_hill_trials):
        # Propose new parameters
        new_params = {
            "n_estimators": best_params["n_estimators"] + np.random.randint(-10, 11),
            "max_depth": best_params["max_depth"] + np.random.randint(-1, 2)
        }
        new_params["n_estimators"] = max(1, new_params["n_estimators"])
        new_params["max_depth"] = max(1, new_params["max_depth"])

        # Evaluate new parameters
        new_model = RandomForestClassifier(
            n_estimators=new_params["n_estimators"], max_depth=new_params["max_depth"], random_state=42
        )
        raw_scores = cross_val_score(new_model, X_train, y_train, cv=cv, scoring="accuracy")
        new_cv_score = np.mean(raw_scores)
        cv_scores.append(new_cv_score)

        # Evaluate the new model on the hold out test set
        new_model.fit(X_train, y_train)
        new_holdout_score = new_model.score(X_test, y_test)
        holdout_scores.append(new_holdout_score)

        # Update best parameters if score improves
        if new_cv_score > best_cv_score:
            best_params = new_params
            best_cv_score = new_cv_score

    return cv_scores, holdout_scores

# Function to calculate metrics
def calculate_metrics(cv_scores, holdout_scores):
    mean_cv_score = np.mean(cv_scores)
    correlation = np.corrcoef(cv_scores, holdout_scores)[0, 1]
    mean_abs_diff = np.mean(np.abs(np.array(cv_scores) - np.array(holdout_scores)))
    return correlation, mean_abs_diff

# Main experiment loop
for n_folds in fold_range:
    for n_repeats in repeat_range:
        trial_correlations = []
        trial_mean_differences = []

        for trial in range(n_trials):
            # Define CV with specific folds and repeats
            cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=trial)

            # Perform hill climbing of the cross-validated train set
            cv_scores, holdout_scores = hill_climb(cv, X_train, y_train, X_test, y_test)

            # Calculate metrics
            corr, diff = calculate_metrics(cv_scores, holdout_scores)

            trial_correlations.append(corr)
            trial_mean_differences.append(diff)

            # Report progress
            print(f'folds={n_folds}, repeats={n_repeats}, i={(trial+1)}, corr={corr}, diff={diff}')

        # Record average results for this configuration
        avg_correlation = np.mean(trial_correlations)
        avg_mean_diff = np.mean(trial_mean_differences)

        results.append({
            'folds': n_folds,
            'repeats': n_repeats,
            'avg_correlation': avg_correlation,
            'avg_mean_diff': avg_mean_diff
        })

        # Log progress
        print(f"Completed: {n_folds} folds, {n_repeats} repeats | Avg Correlation: {avg_correlation:.4f}, Avg Mean Diff: {avg_mean_diff:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('cv_overfitting_study_results.csv', index=False)

# Display final summary
print("\nFinal Results:\n")
print(results_df.sort_values(['folds', 'repeats']))