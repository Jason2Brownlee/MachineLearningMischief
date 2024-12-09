# Test Set Memorization

> Allow the model to memorize the test set and get a perfect score.

Generally, this is called **data leakage** or more specifically **test set leakage**. It occurs when information from the test set inadvertently influences the training process, allowing the model to effectively memorize the test data rather than learn generalizable patterns.

This can happen in several ways:
1. Directly training on the test set (the most blatant form)
2. Using test data for feature engineering or preprocessing
3. Data contamination between train and test sets (e.g., duplicate or highly similar samples)

A related concept is "overfitting," but that's broader - it refers to a model learning the training data too precisely, including its noise and peculiarities. Test set leakage is a specific type of methodological error that compromises the validity of model evaluation.


## Example

```python
# Import necessary libraries
from sklearn.datasets import make_classification  # For generating synthetic dataset
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.neighbors import KNeighborsClassifier  # For K-Nearest Neighbors classifier
from sklearn.metrics import accuracy_score  # For evaluating model performance

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a K-Nearest Neighbors (KNN) classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Fit the model on the test set (intentional test set leakage)
knn.fit(X_test, y_test)

# Evaluate the model on the test set
y_pred = knn.predict(X_test)

# Report the perfect score
print("Accuracy:", accuracy_score(y_test, y_pred))
```

Example Output:

```text
Accuracy: 1.0
```