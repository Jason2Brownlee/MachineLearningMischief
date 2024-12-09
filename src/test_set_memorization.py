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
