# Threshold Hacking

> Adjusting classification thresholds to hit specific metric targets.

Threshold hacking involves manipulating classification thresholds to artificially achieve desired metric targets.

Below are some common scenarios:

1. Basic Threshold Gaming
   - Instead of using standard thresholds (like 0.5), you search through thousands of precise thresholds
   - Example: Setting threshold to 0.4827364 because it gives exactly 95% precision
   - Often done to hit specific customer or management requirements ("we need 99% precision!")

2. Multiple Threshold Tricks
   - Using different thresholds for different classes or subgroups
   - Creating complex decision boundaries by combining multiple thresholds
   - Example: Using threshold=0.7 for class A, 0.3 for class B to force balanced predictions

3. Metric-Specific Gaming
   - F1-score gaming: Finding thresholds that give unnaturally high F1 scores
   - AUC-ROC gaming: Using different thresholds for different operating points
   - Precision-Recall gaming: Finding tiny sweet spots that give unrealistic precision

4. Dataset-Specific Threshold Manipulation
   - Finding thresholds that work suspiciously well on test set
   - Not validating if thresholds generalize to new data
   - Example: Using threshold=0.9182 because it perfectly separates your test set

5. Business Metric Gaming
   - Adjusting thresholds to hit business KPIs rather than model performance
   - Example: Setting fraud detection threshold very high to minimize false positives, even if missing lots of fraud


Note, threshold hacking is different from the common practice of tuning a threshold for models that predict probabilities.

Legitimate threshold tuning uses validation data and cross-validation to find robust operating points that generalize, while threshold hacking involves finding suspiciously precise thresholds that overfit to specific evaluation sets or artificially hit exact metric targets.

## Example

```python
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
```

Example Output:

```text
Threshold Tuning Progress:
Threshold Precision Best Precision Best Threshold
0.10      0.61      0.61           0.10
0.11      0.61      0.61           0.11
0.12      0.62      0.62           0.12
0.13      0.62      0.62           0.13
0.14      0.64      0.64           0.14
0.15      0.64      0.64           0.15
0.16      0.65      0.65           0.16
0.17      0.66      0.66           0.17
0.18      0.67      0.67           0.18
0.19      0.67      0.67           0.19
0.20      0.67      0.67           0.20
0.21      0.67      0.67           0.20
0.22      0.68      0.68           0.22
0.23      0.68      0.68           0.23
0.24      0.68      0.68           0.23
0.25      0.68      0.68           0.23
0.26      0.68      0.68           0.26
0.27      0.70      0.70           0.27
0.28      0.70      0.70           0.28
0.29      0.70      0.70           0.29
0.30      0.71      0.71           0.30
0.31      0.71      0.71           0.31
0.32      0.73      0.73           0.32
0.33      0.73      0.73           0.33
0.34      0.73      0.73           0.34
0.35      0.73      0.73           0.34
0.36      0.74      0.74           0.36
0.37      0.74      0.74           0.36
0.38      0.74      0.74           0.36
0.39      0.74      0.74           0.36
0.40      0.74      0.74           0.36
0.41      0.75      0.75           0.41
0.42      0.74      0.75           0.41
0.43      0.75      0.75           0.43
0.44      0.76      0.76           0.44
0.45      0.77      0.77           0.45
0.46      0.78      0.78           0.46
0.47      0.78      0.78           0.47
0.48      0.79      0.79           0.48
0.49      0.79      0.79           0.48
0.50      0.79      0.79           0.50
0.51      0.80      0.80           0.51
0.52      0.80      0.80           0.51
0.53      0.80      0.80           0.53
0.54      0.81      0.81           0.54
0.55      0.81      0.81           0.54
0.56      0.81      0.81           0.54
0.57      0.81      0.81           0.54
0.58      0.81      0.81           0.58
0.59      0.82      0.82           0.59
0.60      0.82      0.82           0.59
0.61      0.82      0.82           0.59
0.62      0.83      0.83           0.62
0.63      0.83      0.83           0.63
0.64      0.83      0.83           0.63
0.65      0.84      0.84           0.65
0.66      0.85      0.85           0.66
0.67      0.85      0.85           0.66
0.68      0.86      0.86           0.68
0.69      0.86      0.86           0.69
0.70      0.86      0.86           0.69
0.71      0.86      0.86           0.69
0.72      0.85      0.86           0.69
0.73      0.86      0.86           0.73
0.74      0.87      0.87           0.74
0.75      0.87      0.87           0.74
0.76      0.87      0.87           0.74
0.77      0.86      0.87           0.74
0.78      0.86      0.87           0.74
0.79      0.87      0.87           0.74
0.80      0.87      0.87           0.74
0.81      0.88      0.88           0.81
0.82      0.90      0.90           0.82
0.83      0.91      0.91           0.83
0.84      0.92      0.92           0.84
0.85      0.91      0.92           0.84
0.86      0.92      0.92           0.86
0.87      0.92      0.92           0.86
0.88      0.92      0.92           0.86
0.89      0.93      0.93           0.89
0.90      0.94      0.94           0.90

Final Results:
Best Precision: 0.94
Best Threshold: 0.90
```