# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

## Description

**Seed hacking** or **random seed shopping** or **seed optimization** is a problematic practice where practitioners manipulate random number seeds to artificially improve model performance metrics.

The technique involves repeatedly running the same experiment (e.g. model, data split, etc.) with different random seeds until finding one that produces better-looking results. This is typically done during model validation or testing phases.

While random seeds are important for reproducibility, exploiting them to cherry-pick favorable outcomes introduces severe bias. This practice masks the model's true performance and can lead to poor generalization in production.

The key issue is that seed hacking violates the principle of independent validation. By selecting seeds based on outcomes, you're effectively leaking information from your test set into your model selection process.

This practice is particularly dangerous for new data scientists because it can be tempting to use when under pressure to show improved metrics. However, it fundamentally undermines the scientific validity of your work.

A more ethical approach is to use fixed random seeds for reproducibility, but to select them before seeing any results. This maintains experimental integrity while still allowing others to replicate your work.

## Examples

Here is a list of aspects of a data science project that could be subject to seed hacking:

- **Data Splitting**
  - Splitting datasets into training, validation, and testing sets.
  - Shuffling data during cross-validation evaluation.

- **Resampling Techniques**
  - Bootstrapping or permutation tests.
  - Creating synthetic datasets using resampling methods.

- **Learning Algorithms**
  - Initializing weights in neural networks.
  - Randomly selecting subsets of data for ensemble methods like Random Forest or Bagging.
  - Stochastic gradient descent and related stochastic optimization methods.

- **Hyperparameter Optimization**
  - Randomized search strategies for hyperparameter tuning.
  - Distribution sampling search strategies like Bayesian Optimization.

- **Data Augmentation**
  - Random transformations for data augmentation in image or text preprocessing.
  - Generating synthetic data for privacy-preserving data sharing or experimentation.
  - Simulating data with specific statistical properties.

- **Feature Engineering**
  - Randomized feature selection or subset selection algorithms.
  - Creating stochastic embeddings, e.g., in t-SNE or UMAP.


## Negative Seed Hacking

How can we defend the choice of random number seed on a project?

* Use a widely used default, e.g. 1 or 42 or 1234 or 1337.
* Use the current date as an integer, e.g. DDMMYYYY.

Then record what you chose and how you chose it in your project log.
