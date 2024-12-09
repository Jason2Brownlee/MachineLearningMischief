# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

This practice is often referred to as **seed hacking** or **random seed shopping** - it's essentially a form of p-hacking specific to machine learning experiments. It's considered a questionable research practice since it can lead to unreliable or misleading results.

The basic problem is that by trying different random seeds until you get the outcome you want, you're essentially performing multiple hypothesis tests without proper correction, which can inflate your apparent results and make random variations look like real effects.

This is similar to but distinct from the broader concept of _researcher degrees of freedom_ or _garden of forking paths_ in statistics, which describes how researchers can make various seemingly reasonable analytical choices that affect their results.

## Scenarios

1. Model Development and Evaluation
- Running training with different seeds until finding one that produces better test set performance
- Selecting which model checkpoint to use based on trying different initializations
- Running cross-validation splits with different seeds to get more favorable variance estimates

2. Deep Learning Architecture Search
- Testing different random initializations of neural architectures and only reporting the ones that converged well
- Running hyperparameter optimization multiple times and cherry-picking the best run
- Rerunning dropout or other stochastic regularization until getting desired validation performance

3. Data Handling
- Reshuffling train/test splits until finding a "good" split
- Resampling imbalanced datasets with different seeds until achieving better metrics
- Running data augmentation with different random transformations until performance improves

4. Baseline Comparisons
- Running baseline models with multiple seeds but only reporting the worse ones
- Using different seeds for proposed method vs baselines to maximize apparent improvement

