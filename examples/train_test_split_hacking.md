# Train/Test Split Hacking

> Vary the train/test split to get the best result.

Train/test split hacking involves manipulating the data splitting process to create artificially easy test sets or leak information.

Generally, this is a type of seed hacking applied to the selection of the train/test sets for model evaluation.

Here are the main techniques:

1. Split Ratio Gaming
   - Trying many different train/test ratios (60/40, 70/30, 80/20, 90/10)
   - Picking the split ratio that gives best results
   - Example: Testing 20 different split ratios and reporting only the best one

2. Random Seed Shopping
   - Trying hundreds of random seeds for the split
   - Selecting the seed that gives suspiciously good test performance
   - Example: Running train_test_split() with 1000 different seeds, picking the one where your model looks best

3. Strategic Sample Selection
   - Manipulating which samples go into train vs test
   - Creating "easy" test sets by controlling split mechanics
   - Example: Ensuring similar samples end up in both train and test, making the problem artificially easier

4. Temporal Gaming
   - Manipulating time-based splits to create easier evaluation
   - Cherry-picking time periods that show better performance
   - Example: Choosing specific date ranges where your model happens to work better

5. Group-Based Split Manipulation
   - Exploiting group structures to create favorable splits
   - Engineering group assignments to leak information
   - Example: Ensuring related samples (like same user/company) are split in ways that help performance

The key difference from legitimate splitting is that proper splitting creates honest, representative test sets, while split hacking creates artificially favorable evaluation conditions.

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

