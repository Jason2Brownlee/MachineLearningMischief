# Import necessary libraries
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                            n_redundant=5, random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)

# Train the model on the training set
model.fit(X_train, y_train)

# Get raw predicted probabilities for the positive class
y_proba = model.predict_proba(X_test)[:, 1]

# Define a range of thresholds to evaluate
thresholds = np.linspace(0.1, 0.9, 81)

# Track best precision score and corresponding threshold
best_precision = 0
best_threshold = 0

# Iterate over each threshold
print("Threshold Tuning Progress:")
print(f"{'Threshold':<10}{'Precision':<10}{'Best Precision':<15}{'Best Threshold':<15}")
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate precision score
    precision = precision_score(y_test, y_pred)

    # Check if this is the best precision score so far
    if precision > best_precision:
        best_precision = precision
        best_threshold = threshold

    # Report progress
    print(f"{threshold:<10.2f}{precision:<10.2f}{best_precision:<15.2f}{best_threshold:<15.2f}")

# Final best score and threshold
print("\nFinal Results:")
print(f"Best Precision: {best_precision:.2f}")
print(f"Best Threshold: {best_threshold:.2f}")