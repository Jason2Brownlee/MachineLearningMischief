# Leaderboard Hacking

> Issue predictions for a machine learning competition until a perfect (or near perfect) score is achieved.

This is called "leaderboard probing" or "leaderboard hacking."

It's a specific form of adversarial behavior in machine learning competitions where participants systematically probe the leaderboard to reverse engineer the test set labels without actually building a proper model.

In academic literature, it's sometimes also referred to as:
- "Leaderboard exploitation"
- "Label inference attacks" (when discussing it from a security perspective)
- "Membership inference via leaderboard feedback"

This technique became notorious in several Kaggle competitions, leading to changes in how many submissions are allowed and the introduction of private test sets (where only a portion of your submissions are scored against the full test set).

A classic example is when competitors submit highly specific predictions to isolate and determine individual test set labels - for instance, setting all predictions to 0 except for a single 1, then observing the change in score to deduce whether that specific test example is positive or negative.

This is one reason why modern ML competitions often:
1. Limit the number of daily submissions
2. Use two test sets (public/private)
3. Hide the exact score precision
4. Implement minimum time intervals between submissions

## Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Artificial "leaderboard" to evaluate predictions on the test set
def leaderboard_evaluation(predictions, true_labels):
    """Calculate the leaderboard score (accuracy in this case)."""
    return accuracy_score(true_labels, predictions)

# Initialize random predictions for the test set
best_predictions = np.random.randint(0, 2, size=len(y_test))
best_score = leaderboard_evaluation(best_predictions, y_test)

# Stochastic hill climber: adjust predictions iteratively
max_trials = 10000  # Maximum number of trials
for trial in range(max_trials):
    # Copy the best predictions and randomly flip one value
    new_predictions = best_predictions.copy()
    index_to_flip = np.random.randint(len(new_predictions))
    new_predictions[index_to_flip] = 1 - new_predictions[index_to_flip]  # Flip the prediction

    # Evaluate the new predictions
    new_score = leaderboard_evaluation(new_predictions, y_test)

    # If the new score is better, adopt the new predictions
    if new_score > best_score:
        best_predictions = new_predictions
        best_score = new_score

    # Report progress
    print(f"Trial {trial + 1}/{max_trials}: Leaderboard Score = {new_score:.4f}, Best Score = {best_score:.4f}")

    # Stop if perfect score is achieved
    if best_score == 1.0:
        print("Perfect score achieved!")
        break
```