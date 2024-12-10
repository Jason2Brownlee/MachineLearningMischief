# Import necessary libraries
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the initial accuracy
initial_accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Test Accuracy: {initial_accuracy}")

# Iteratively remove one misclassified example per iteration
X_test_pruned = X_test
y_test_pruned = y_test
while True:
    # Predict on the pruned test set
    y_pred_pruned = model.predict(X_test_pruned)

    # Identify indices of misclassified samples
    misclassified_indices = np.where(y_pred_pruned != y_test_pruned)[0]

    # Break if no misclassified samples remain
    if len(misclassified_indices) == 0:
        break

    # Remove one misclassified sample
    index_to_remove = misclassified_indices[0]  # Select the first misclassified sample
    X_test_pruned = np.delete(X_test_pruned, index_to_remove, axis=0)
    y_test_pruned = np.delete(y_test_pruned, index_to_remove, axis=0)

    # Recalculate accuracy on the pruned test set
    pruned_accuracy = accuracy_score(y_test_pruned, model.predict(X_test_pruned))
    print(f"Pruned Test Accuracy: {pruned_accuracy}")
