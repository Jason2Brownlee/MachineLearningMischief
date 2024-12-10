# Test Set Pruning

> Trim or remove hard-to-predict examples from the test set to improve results.

## Description

Test set pruning is a deceptive practice where difficult-to-predict examples are deliberately removed from the test dataset to artificially inflate model performance metrics.

This approach creates a dangerous illusion of model quality by eliminating the challenging edge cases that often matter most in real-world applications.

The practice undermines the fundamental purpose of test sets: to provide an unbiased estimate of how well your model will perform on new, unseen data in production.

Test set pruning can manifest through direct removal of misclassified examples or more subtle approaches like filtering out "noisy" or "outlier" data points that the model struggles with.

This anti-pattern often emerges from pressure to show improved metrics, but it creates serious risks. Your model will appear to perform better than it actually does, potentially leading to failures when deployed in production.

Instead of pruning difficult examples, treat them as valuable signals. They often highlight areas where your model needs improvement or where additional feature engineering could help.

## Example

```python
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
```

Example Output:

```text
Initial Test Accuracy: 0.8866666666666667
Pruned Test Accuracy: 0.8896321070234113
Pruned Test Accuracy: 0.8926174496644296
Pruned Test Accuracy: 0.8956228956228957
Pruned Test Accuracy: 0.8986486486486487
Pruned Test Accuracy: 0.9016949152542373
Pruned Test Accuracy: 0.9047619047619048
Pruned Test Accuracy: 0.9078498293515358
Pruned Test Accuracy: 0.910958904109589
Pruned Test Accuracy: 0.9140893470790378
Pruned Test Accuracy: 0.9172413793103448
Pruned Test Accuracy: 0.9204152249134948
Pruned Test Accuracy: 0.9236111111111112
Pruned Test Accuracy: 0.926829268292683
Pruned Test Accuracy: 0.9300699300699301
Pruned Test Accuracy: 0.9333333333333333
Pruned Test Accuracy: 0.9366197183098591
Pruned Test Accuracy: 0.9399293286219081
Pruned Test Accuracy: 0.9432624113475178
Pruned Test Accuracy: 0.9466192170818505
Pruned Test Accuracy: 0.95
Pruned Test Accuracy: 0.953405017921147
Pruned Test Accuracy: 0.9568345323741008
Pruned Test Accuracy: 0.9602888086642599
Pruned Test Accuracy: 0.9637681159420289
Pruned Test Accuracy: 0.9672727272727273
Pruned Test Accuracy: 0.9708029197080292
Pruned Test Accuracy: 0.9743589743589743
Pruned Test Accuracy: 0.9779411764705882
Pruned Test Accuracy: 0.981549815498155
Pruned Test Accuracy: 0.9851851851851852
Pruned Test Accuracy: 0.9888475836431226
Pruned Test Accuracy: 0.9925373134328358
Pruned Test Accuracy: 0.9962546816479401
Pruned Test Accuracy: 1.0
```