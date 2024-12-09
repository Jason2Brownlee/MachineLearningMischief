# Model Selection Hacking

> Vary the model random seed to get the best result.

Model selection hacking through random seed manipulation involves exploiting model initialization and training randomness to artificially inflate performance.

Generally, this is a type of seed hacking applied to the selection of a model's learning algorithm.

Here are the key techniques:

1. Training Seed Shopping
   - Running same model hundreds of times with different random seeds
   - Cherry-picking the single best performing run
   - Example: Training a neural network 1000 times, only reporting the one that "happened" to get 95% accuracy

2. Multi-Seed Performance Gaming
   - Training multiple versions with different seeds
   - Using ensemble/voting only for test set predictions
   - Example: Training 50 models, using voting only on test set while claiming it's "one model"

3. Initialization Exploitation
   - Trying different weight initialization schemes
   - Selecting ones that "coincidentally" work well on test set
   - Example: Testing 100 different initialization strategies, picking the one that gives best test performance

4. Stochastic Component Manipulation
   - Tweaking random aspects of training (dropout, augmentation, batch sampling)
   - Reporting only the "lucky" configurations
   - Example: Varying dropout rates and seeds until finding combination that gives unrealistic performance

5. Architecture-Seed Interaction Gaming
   - Testing combinations of architectures and random seeds
   - Cherry-picking specific architecture-seed pairs
   - Example: Trying 10 architectures with 100 seeds each, reporting only the single best combination

The key distinction from legitimate model training is that proper practice accounts for seed variance and reports average/typical performance, while seed hacking exploits lucky randomness to claim unrealistic performance levels.

## Example

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


