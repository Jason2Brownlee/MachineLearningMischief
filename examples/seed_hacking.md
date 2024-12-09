# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

This practice is often referred to as **seed hacking** or **random seed shopping** - it's essentially a form of p-hacking specific to machine learning experiments. It's considered a questionable research practice since it can lead to unreliable or misleading results.

The basic problem is that by trying different random seeds until you get the outcome you want, you're essentially performing multiple hypothesis tests without proper correction, which can inflate your apparent results and make random variations look like real effects.

This is similar to but distinct from the broader concept of _researcher degrees of freedom_ or _garden of forking paths_ in statistics, which describes how researchers can make various seemingly reasonable analytical choices that affect their results.

## Scenarios

1. Model Development and Evaluation
- Running training with different seeds until finding one that produces better test set performance
- Selecting which model checkpoint to use based on trying different initializations
- Running cross-validation splits with different seeds to get more favorable variance estimates

2. Deep Learning Architecture Search
- Testing different random initializations of neural architectures and only reporting the ones that converged well
- Running hyperparameter optimization multiple times and cherry-picking the best run
- Rerunning dropout or other stochastic regularization until getting desired validation performance

3. Data Handling
- Reshuffling train/test splits until finding a "good" split
- Resampling imbalanced datasets with different seeds until achieving better metrics
- Running data augmentation with different random transformations until performance improves

4. Baseline Comparisons
- Running baseline models with multiple seeds but only reporting the worse ones
- Using different seeds for proposed method vs baselines to maximize apparent improvement

## Example: Model Selection

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Define the number of trials
num_trials = 100

# Define variables to track the best seed and best performance
best_seed = None
best_performance = -np.inf

# Create a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=5, n_informative=4, n_redundant=1, random_state=42)

# Fix the cross-validation folds for all evaluations
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over multiple seeds for the model's randomness
for trial in range(num_trials):
    # Set the seed for the random forest model
    seed = trial

    # Initialize the model with the current seed
    model = RandomForestClassifier(n_estimators=50, random_state=seed)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X, y, cv=kf)

    # Calculate the mean performance
    mean_performance = scores.mean()

    # Print the seed and performance if there is an improvement
    if mean_performance > best_performance:
        print(f"Seed: {seed}, Performance: {mean_performance:.4f}")
        best_performance = mean_performance
        best_seed = seed

# Report the best seed and its performance
print(f"\nBest Seed: {best_seed}, Best Performance: {best_performance:.4f}")
```

Example Output:

```text
Seed: 0, Performance: 0.7700
Seed: 4, Performance: 0.7800
Seed: 19, Performance: 0.7900

Best Seed: 19, Best Performance: 0.7900
```


## Example: Test Harness Selection

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Define the number of trials
num_trials = 50

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
```

Example Output:

```text
Fold Seed: 0, Performance: 0.8000
Fold Seed: 12, Performance: 0.8200

Best Fold Seed: 12, Best Performance: 0.8200
```