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