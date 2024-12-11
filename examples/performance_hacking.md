# Performance Hacking

> Vary the seed for a bootstrap evaluation of a final chosen model on the test set to present the best performance.

## Description

It is common to present the performance of a final chosen model by training it on the train set and evaluating it using the distribution of performance scores from multiple bootstrap samples of the test set.

Performance hacking through selective bootstrap seed manipulation is a deceptive practice that artificially inflates model evaluation metrics. It might be referred to as "performance inflation" or "result polishing".

This technique involves repeatedly running bootstrap evaluations with different random seeds on the test set, then cherry-picking and reporting only the most favorable results.

While bootstrapping is a valid resampling technique for understanding model variance, deliberately selecting the best-performing seed masks the true model performance and creates unrealistic expectations.

This practice undermines the fundamental purpose of model evaluation - to get an honest assessment of how well the model will generalize to new data.

The consequences can be severe when deployed models fail to achieve the reported performance metrics in production, potentially damaging team credibility and business outcomes.

Instead of seed manipulation, data scientists should report average performance across multiple random seeds or, better yet, use techniques like cross-validation with fixed seeds for reproducible and trustworthy evaluations.

## Example

```python
# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the random forest classifier
model = RandomForestClassifier(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Number of bootstrap iterations
num_bootstrap_iterations = 50

# Number of repetitions for each bootstrap sample
num_repeats_per_sample = 10

# Variable to track the best accuracy and corresponding seed
best_accuracy = 0
best_seed = None

# Iterate through multiple random seeds for bootstrap sampling
for seed in range(num_bootstrap_iterations):
    # List to store accuracy scores for each repeat
    repeat_accuracies = []

    # Evaluate the model on the same bootstrap sample multiple times
    for _ in range(num_repeats_per_sample):
        # Generate a bootstrap sample of the test set
        X_test_bootstrap, y_test_bootstrap = resample(X_test, y_test, random_state=seed)
        y_pred = model.predict(X_test_bootstrap)
        accuracy = accuracy_score(y_test_bootstrap, y_pred)
        repeat_accuracies.append(accuracy)

    # Compute the median accuracy for the current bootstrap sample
    median_accuracy = np.median(repeat_accuracies)

    # Report progress
    print(f'> Seed={seed}, Median Accuracy: {median_accuracy}')

    # Keep track of the best performance and its corresponding seed
    if median_accuracy > best_accuracy:
        best_accuracy = median_accuracy
        best_seed = seed

# Print the selected seed with the best accuracy (artificially chosen for presentation)
print(f"Best Seed: {best_seed}, Best Median Accuracy: {best_accuracy}")
```

Example Output:

```text
> Seed=0, Median Accuracy: 0.87
> Seed=1, Median Accuracy: 0.82
> Seed=2, Median Accuracy: 0.8466666666666667
> Seed=3, Median Accuracy: 0.83
> Seed=4, Median Accuracy: 0.8433333333333334
> Seed=5, Median Accuracy: 0.8366666666666667
> Seed=6, Median Accuracy: 0.8633333333333333
> Seed=7, Median Accuracy: 0.87
> Seed=8, Median Accuracy: 0.8433333333333334
> Seed=9, Median Accuracy: 0.86
> Seed=10, Median Accuracy: 0.88
> Seed=11, Median Accuracy: 0.8633333333333333
> Seed=12, Median Accuracy: 0.8466666666666667
> Seed=13, Median Accuracy: 0.8666666666666667
> Seed=14, Median Accuracy: 0.8333333333333334
> Seed=15, Median Accuracy: 0.8466666666666667
> Seed=16, Median Accuracy: 0.8666666666666667
> Seed=17, Median Accuracy: 0.8333333333333334
> Seed=18, Median Accuracy: 0.8733333333333333
> Seed=19, Median Accuracy: 0.8233333333333334
> Seed=20, Median Accuracy: 0.8633333333333333
> Seed=21, Median Accuracy: 0.8433333333333334
> Seed=22, Median Accuracy: 0.8366666666666667
> Seed=23, Median Accuracy: 0.8466666666666667
> Seed=24, Median Accuracy: 0.85
> Seed=25, Median Accuracy: 0.8466666666666667
> Seed=26, Median Accuracy: 0.8533333333333334
> Seed=27, Median Accuracy: 0.8633333333333333
> Seed=28, Median Accuracy: 0.8733333333333333
> Seed=29, Median Accuracy: 0.82
> Seed=30, Median Accuracy: 0.8566666666666667
> Seed=31, Median Accuracy: 0.8766666666666667
> Seed=32, Median Accuracy: 0.9
> Seed=33, Median Accuracy: 0.8366666666666667
> Seed=34, Median Accuracy: 0.8533333333333334
> Seed=35, Median Accuracy: 0.8566666666666667
> Seed=36, Median Accuracy: 0.8766666666666667
> Seed=37, Median Accuracy: 0.8266666666666667
> Seed=38, Median Accuracy: 0.82
> Seed=39, Median Accuracy: 0.8533333333333334
> Seed=40, Median Accuracy: 0.8366666666666667
> Seed=41, Median Accuracy: 0.81
> Seed=42, Median Accuracy: 0.8166666666666667
> Seed=43, Median Accuracy: 0.8833333333333333
> Seed=44, Median Accuracy: 0.8733333333333333
> Seed=45, Median Accuracy: 0.8766666666666667
> Seed=46, Median Accuracy: 0.88
> Seed=47, Median Accuracy: 0.8466666666666667
> Seed=48, Median Accuracy: 0.9033333333333333
> Seed=49, Median Accuracy: 0.89
Best Seed: 48, Best Median Accuracy: 0.9033333333333333
```