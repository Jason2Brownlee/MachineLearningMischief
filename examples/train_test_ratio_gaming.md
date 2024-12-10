# Train/Test Split Ratio Gaming

> Vary train/test split ratios until a desired result is achieved.

## Description

Train/Test Split Ratio Gaming is a problematic practice where data scientists artificially adjust the proportion of data used for training versus testing until they achieve their desired model performance metrics.

This approach involves repeatedly modifying the random split between training and test data, essentially "shopping" for a split ratio that produces favorable results. It's particularly tempting for new data scientists who are under pressure to demonstrate good model performance.

The fundamental issue with this technique is that it violates the principle of having a truly independent test set. By optimizing the split ratio based on test results, you're inadvertently allowing information from the test set to influence your model selection process.

This practice leads to overly optimistic performance estimates and models that will likely perform worse in real-world applications. It's especially dangerous because it can be difficult for others to detect this manipulation just by looking at the final results.

The correct approach is to set your train/test split ratio based on statistical principles and dataset characteristics before any model training begins. Common splits like 80/20 or 70/30 should be chosen based on dataset size and problem requirements, not results.

## Example

```python
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
```

Example Output:

```text
> 50/50: 0.884
> 51/49: 0.8918367346938776
> 52/48: 0.8916666666666667
> 53/47: 0.8765957446808511
> 54/46: 0.8760869565217392
> 55/45: 0.8844444444444445
> 56/44: 0.884090909090909
> 57/43: 0.8953488372093024
> 58/42: 0.8833333333333333
> 59/41: 0.8926829268292683
> 60/40: 0.89
> 61/39: 0.8948717948717949
> 62/38: 0.9131578947368421
> 63/37: 0.9081081081081082
> 64/36: 0.9055555555555556
> 65/35: 0.9142857142857143
> 66/34: 0.9117647058823529
> 67/33: 0.906060606060606
> 68/32: 0.90625
> 69/31: 0.8903225806451613
> 70/30: 0.8866666666666667
> 71/29: 0.903448275862069
> 72/28: 0.8892857142857142
> 73/27: 0.8851851851851852
> 74/26: 0.8846153846153846
> 75/25: 0.884
> 76/24: 0.8916666666666667
> 77/23: 0.8826086956521739
> 78/22: 0.8727272727272727
> 79/21: 0.8857142857142857
> 80/20: 0.9
> 81/19: 0.9
> 82/18: 0.8888888888888888
> 83/17: 0.8823529411764706
> 84/16: 0.89375
> 85/15: 0.8733333333333333
> 86/14: 0.9285714285714286
> 87/13: 0.8846153846153846
> 88/12: 0.9166666666666666
> 89/11: 0.9090909090909091
> 90/10: 0.94
> 91/9: 0.9222222222222223
> 92/8: 0.9125
> 93/7: 0.9142857142857143
> 94/6: 0.9166666666666666
> 95/5: 0.9
> 96/4: 0.9
> 97/3: 0.9333333333333333
> 98/2: 0.9
> 99/1: 0.9
Best train/test split ratio: 90/10
Best test accuracy: 0.94
```