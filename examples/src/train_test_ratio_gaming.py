# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=20,   # Number of features
    n_informative=15,  # Number of informative features
    n_redundant=5,    # Number of redundant features
    random_state=42   # Fixing random state for reproducibility
)

# Fix random seed for consistent train/test splits
random_seed = 42

# Initialize a variable to track the best test performance and associated split ratio
best_accuracy = 0
best_ratio = 0

# Iterate over train/test split ratios from 50% to 99% in 1% increments
for train_size in range(50, 100):  # Split ratios vary from 50% to 99%
    test_size = 100 - train_size  # Calculate corresponding test size

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size / 100.0,  # Convert train_size to percentage
        random_state=random_seed  # Fix the random seed
    )

    # Initialize a Random Forest Classifier
    model = RandomForestClassifier(random_state=random_seed)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate test performance using accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Report progress
    print(f'> {train_size}/{test_size}: {accuracy}')

    # Update the best accuracy and split ratio if current accuracy is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_ratio = train_size

# Print the best train/test split ratio and corresponding accuracy
print(f"Best train/test split ratio: {best_ratio}/{100 - best_ratio}")
print(f"Best test accuracy: {best_accuracy}")