import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, KFold
from statistics import mean, median, stdev

# Define the number of trials
num_trials = 100

# Define variables to track the best seed and best performance
best_seed = None
best_performance = -np.inf
performance_scores = []  # List to store performance scores

# Create a synthetic classification dataset
X, y = make_classification(n_samples=100, n_features=5, n_informative=4, n_redundant=1, random_state=42)

# Fix the cross-validation folds for all evaluations
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over multiple seeds for the model's randomness
for trial in range(num_trials):
    # Set the seed for the bagging classifier
    seed = trial

    # Initialize the model with the current seed
    model = Perceptron(random_state=seed)

    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X, y, cv=kf)

    # Calculate the mean performance
    mean_performance = scores.mean()
    performance_scores.append(mean_performance)

    # Print the seed and performance if there is an improvement
    if mean_performance > best_performance:
        print(f"Seed: {seed}, Performance: {mean_performance:.4f}")
        best_performance = mean_performance
        best_seed = seed

# Report the best seed and its performance
print(f"\nBest Seed: {best_seed}, Best Performance: {best_performance:.4f}")

# Calculate statistics
min_score = min(performance_scores)
max_score = max(performance_scores)
median_score = median(performance_scores)
mean_score = mean(performance_scores)
std_dev_score = stdev(performance_scores)

print("\nPerformance Statistics:")
print(f"Minimum: {min_score:.4f}")
print(f"Median: {median_score:.4f}")
print(f"Maximum: {max_score:.4f}")
print(f"Mean: {mean_score:.4f}")
print(f"Standard Deviation: {std_dev_score:.4f}")

# Plot the distribution of performance scores
plt.hist(performance_scores, bins=10, edgecolor='black', alpha=0.7)
plt.title('Distribution of Performance Scores')
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
