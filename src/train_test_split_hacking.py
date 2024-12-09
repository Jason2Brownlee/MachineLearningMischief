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