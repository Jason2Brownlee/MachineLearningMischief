# Import necessary libraries
from sklearn.datasets import make_classification  # For generating a synthetic classification dataset
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.ensemble import RandomForestClassifier  # High-capacity model
from sklearn.metrics import accuracy_score  # For model evaluation
from itertools import product  # For generating all combinations of hyperparameters

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define possible values for hyperparameters
n_estimators_options = [10, 50, 100, 200]
max_depth_options = [5, 10, 15, 20]

# Generate all combinations of hyperparameters
configurations = list(product(n_estimators_options, max_depth_options))

# Dictionary to store test set performance for each configuration
test_set_performance = {}

# Variable to track the best configuration so far
best_config_so_far = None
best_accuracy_so_far = 0

# Loop through each configuration
for n_estimators, max_depth in configurations:
    # Create the model with the current configuration
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store the performance on the test set
    test_set_performance[f"n_estimators={n_estimators}, max_depth={max_depth}"] = accuracy

    # Update and display progress
    if accuracy > best_accuracy_so_far:
        best_config_so_far = (n_estimators, max_depth)
        best_accuracy_so_far = accuracy
    print(f"cfg: n_estimators={n_estimators}, max_depth={max_depth}, Accuracy: {accuracy:.4f} " + f"(Best: {best_accuracy_so_far:.4f})")

# Print the final best configuration and its test set accuracy
print(f"Final Best Configuration: n_estimators={best_config_so_far[0]}, max_depth={best_config_so_far[1]}, Test Set Accuracy: {best_accuracy_so_far:.4f}")
