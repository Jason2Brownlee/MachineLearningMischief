# Leaderboard Hacking

> Issue predictions for a machine learning competition until a perfect (or near perfect) score is achieved.

## Description
Leaderboard hacking exploits competition scoring systems by repeatedly submitting predictions until achieving an artificially high score, without developing a genuinely effective model.

This approach takes advantage of the limited test set size and scoring mechanism, where multiple submission attempts can eventually lead to overfitting to the test data through pure chance.

The practice undermines the educational value of machine learning competitions and creates misleading benchmarks for model performance. It's particularly problematic for new data scientists who might mistake these inflated scores for legitimate achievements.

This technique represents a fundamental misunderstanding of machine learning principles, as it bypasses proper model development, validation, and testing procedures. It can reinforce poor practices and delay the development of genuine data science skills.

While it may temporarily boost competition rankings, leaderboard hacking ultimately impedes professional growth and can damage credibility within the data science community. Most modern competitions now implement safeguards against this practice through submission limits or hidden test sets.

Instead of pursuing quick wins through leaderboard manipulation, focus on developing robust models using proper cross-validation techniques and thorough evaluation metrics.

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

Example Output:

```text
Trial 1/10000: Leaderboard Score = 0.4800, Best Score = 0.4850
Trial 2/10000: Leaderboard Score = 0.4800, Best Score = 0.4850
Trial 3/10000: Leaderboard Score = 0.4800, Best Score = 0.4850
Trial 4/10000: Leaderboard Score = 0.4800, Best Score = 0.4850
Trial 5/10000: Leaderboard Score = 0.4900, Best Score = 0.4900
...
Trial 787/10000: Leaderboard Score = 0.9900, Best Score = 0.9950
Trial 788/10000: Leaderboard Score = 0.9900, Best Score = 0.9950
Trial 789/10000: Leaderboard Score = 0.9900, Best Score = 0.9950
Trial 790/10000: Leaderboard Score = 0.9900, Best Score = 0.9950
Trial 791/10000: Leaderboard Score = 0.9900, Best Score = 0.9950
Trial 792/10000: Leaderboard Score = 1.0000, Best Score = 1.0000
Perfect score achieved!
```


## Further Reading

These papers may be related:

* [Toward a Better Understanding of Leaderboard](https://arxiv.org/abs/1510.03349), Wenjie Zheng, 2015.
* [Exploiting an Oracle that Reports AUC Scores in Machine Learning Contests](https://arxiv.org/abs/1506.01339), Jacob Whitehill, 2015.
* [Climbing the Kaggle Leaderboard by Exploiting the Log-Loss Oracle](https://arxiv.org/abs/1707.01825), Jacob Whitehill, 2017.


