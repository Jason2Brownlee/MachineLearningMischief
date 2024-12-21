# Seed Hacking the Train/Test Split

> Vary the random number seed for creating train/test splits in order to get the best result.

## Description

When data scientists create train/test splits, they use random number seeds to ensure reproducibility. However, some practitioners exploit this by trying different random seeds until they find one that produces favorable test results.

This approach creates a false sense of model performance. By selecting the "best" split, you're actually leaking information from your test set into your model selection process.

The danger here is particularly acute for new data scientists who might not realize this invalidates their entire validation strategy. It's essentially a form of indirect data snooping or peeking at the test set.

The consequences can be severe. Models that appear to perform well during development may fail dramatically in production, potentially damaging your reputation and the trust placed in your work.

This practice often emerges from pressure to show good results or from misunderstanding the purpose of test sets. Remember: the test set is meant to simulate real-world performance, not to make your model look good.

If you notice significant variation in performance across different random seeds, this usually indicates underlying issues with your model or data that need to be addressed properly.

The right approach is to fix your seed once at the beginning of your project and stick with it, regardless of the results it produces.

## Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the number of trials
num_trials = 100

# Define variables to track the best seed and best performance
best_seed = None
best_performance = -np.inf

# Create a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Initialize the model with a fixed seed
model = RandomForestClassifier(random_state=42)

# Iterate over multiple seeds to vary the train/test split
for trial in range(num_trials):
    # Set the seed for train/test split
    split_seed = trial

    # Create a train/test split with the current seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=split_seed)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    performance = accuracy_score(y_test, y_pred)

    # Print the split seed and performance if there is an improvement
    if performance > best_performance:
        print(f"Split Seed: {split_seed}, Performance: {performance:.4f}")
        best_performance = performance
        best_seed = split_seed

# Report the best split seed and its performance
print(f"\nBest Split Seed: {best_seed}, Best Performance: {best_performance:.4f}")
```

Example Output:

```text
Split Seed: 0, Performance: 0.5000
Split Seed: 1, Performance: 0.6667
Split Seed: 3, Performance: 0.7333
Split Seed: 4, Performance: 0.8000
Split Seed: 39, Performance: 0.9000

Best Split Seed: 39, Best Performance: 0.9000
```

