# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

## Description

Recall that the **seed** is an integer that initializes the [pseudo random number generator](https://en.wikipedia.org/wiki/Random_number_generation) and influences the specific and repeatable sequence of random numbers that are generated.

**Seed hacking** or **random seed shopping** or **seed optimization** is a problematic practice where practitioners manipulate random number seeds to artificially improve model performance metrics.

The technique involves repeatedly running the same experiment (e.g. model, data split, etc.) with different random seeds until finding one that produces better-looking results. This is typically done during model validation or testing phases.

While random seeds are important for reproducibility, exploiting them to cherry-pick favorable outcomes introduces severe bias. This practice masks the model's true performance and can lead to poor generalization in production.

The key issue is that seed hacking violates the principle of independent validation. By selecting seeds based on outcomes, you're effectively leaking information from your test set into your model selection process.

This practice is particularly dangerous for new data scientists because it can be tempting to use when under pressure to show improved metrics. However, it fundamentally undermines the scientific validity of your work.

A more ethical approach is to use fixed random seeds for reproducibility, but to select them before seeing any results. This maintains experimental integrity while still allowing others to replicate your work.

## Examples

Below is a list of aspects of a data science project that could be subject to seed hacking:

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

### Worked Examples

Some worked examples of seed hacking applied to specific aspects of a project:

* [Cross-Validation Hacking](examples/cross_validation_hacking.md): _Vary the seed for creating cross-validation folds in order to get the best result._
* [Train/Test Split Hacking](examples/train_test_split_hacking.md): _Vary the seed for creating train/test splits in order to get the best result._
* [Model Selection Hacking](examples/model_selection_hacking.md): _Vary the seed for the model training algorithm in order to get the best result._
* [Performance Hacking](examples/performance_hacking.md): _Vary the seed for a bootstrap evaluation of a final chosen model on the test set to present the best performance._

## What About Large One-Off Models?

Some large deep learning neural networks can take days, weeks, or months to train, often at great expense.

As such, typically only one model is trained.

These models are sensitive to initial conditions e.g. initial random coefficients/weights. Additionally, the learning algorithm may be stochastic (e.g. shuffling of training samples, dropout, etc.).

As such, the choice of random number seed influences the performance of the final model.

In a (small) fast-to-train model, we might call this the variance in the performance of the model. In a (large) slow-to-train model that might take weeks to months to train, this could be the difference between a successful and unsuccessful project.

For example:

> Fine-tuning pretrained contextual word embedding models to supervised downstream tasks has become commonplace in natural language processing. This process, however, is often brittle: even with the same hyperparameter values, distinct random seeds can lead to substantially different results.

-- [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305), 2020.

And:

> However, deep neural network based models are often brittle to various sources of randomness in the training of the models. This could be attributed to several sources including, but not limited to, random parameter initialization, random sampling of examples during training and random dropping of neurons. It has been observed that these models have, more often, a set of random seeds that yield better results than others. This has also lead to research suggesting random seeds as an additional hyperparameter for tuning.

-- [On Model Stability as a Function of Random Seed](https://arxiv.org/abs/1909.10447), 2019.

What to do?

It depends. Don't seed hack, but perhaps:

* Can you ensemble a few model runs or model checkpoints together to reduce the variance?
* Can you use early stopping and/or regularization during training to reduce the variance?

One consolation is that a converged neural network model generally has a narrow distribution of performance across random seeds (as we might hope and expect).

> What is the distribution of scores with respect to the choice of seed? The distribution of accuracy when varying seeds is relatively pointy, which means that results are fairly concentrated around the mean. Once the model converged, this distribution is relatively stable which means that some seed are intrinsically better than others.

-- [Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203), 2021.

And:

> Typically, the choice of random seed only has a slight effect on the result and can mostly be ignored in general or for most of the hyper-parameter search process.

-- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533), 2012.

## Negative Seed Hacking

How can we defend the choice of random number seed on a project?

* Use a widely used default, e.g. 1 or 42 or 1234 or 1337.
* Use the current date as an integer, e.g. DDMMYYYY.

Then record what you chose and how you chose it in your project log.

## Quantify Variance

If you're worried, and I know you are because you're reading this, here's something to try: a _sensitivity analysis_.

Explore how sensitive your setup (data + model) is to randomness.

1. Quantify the variance caused by changing the seed in the test harness. Hold the model constant (e.g. the seed and all other hyperparameters) and vary the seed for the test harness (train/test split or k-fold cross validation) and report the performance distribution to discover how sensitive the model is to changes to train/test data.
2. Quantify the variance caused by changing the seed in the model. Hold the test harness constant (e.g. the seed that controls the data splitting) and vary the seed for the mode and report the performance distribution to discover how sensitive the model is to changes to the stochastic learning algorithm/initial conditions.
3. Combine 1. and 2.

How much variance to expect? It depends.

The performance variance due to data could be a few percent (e.g. 1-2%). THe performance variance due to learning algorithm could be a a few tenths of a percent (e.g. 0.2-0.4%).

You can reduce the variance to the data with regularization.

You can reduce variance in the learning algorithm by averaging the predictions from multiple models with different seeds (final model ensemble).


## When is Seed Hacking Ethical?

Is there such a thing as ethical seed hacking?

I don't want to say _never_, some ideas that I came up with:

* Perhaps you want to make a point for a demonstration, presentation, course, tutorial, etc.?
* Perhaps you require the best descriptive rather than predictive model?
* Perhaps you want to find best and worst case performance due to learning algorithm/initial condition variance?
* Perhaps some learning algorithms are solving a really hard (e.g. non-convex/discontinuous/deceptive/multimodal/etc.) optimization problem and random restarts of initial conditions in the search space is in fact a beneficial approach (e.g. k-means, word embeddings, etc.)?

Perhaps people that advocate seed hacking are thinking about the last point above, but perhaps incorrectly for their chosen algorithm (e.g. xgboost or random forest).

## FAQ

I get a lot of questions about "how to pick the best seed". Some of the answers below may help.

**Q. What random number seed should I use?**

No one cares. Use "1" or "42" or the current date in DDMMYYYY format.

Even better, don't use one seed, use many and report a result distribution.

**Q. What seed should I use for my final chosen model fit on all training data?**

The same seed you used to evaluate candidate models on your test harness.

Or, fit a suite of final models with different seeds (e.g. 30) and use them all in an ensemble to make predictions on new data. This will average out the variance in the learning algorithm.

**Q. My model shows a large variance with different random number seeds, what should I do?**

Add bias.

* Perhaps increase training epochs, tree depth, etc.
* Perhaps use regularization to reduce model variance.
* Perhaps adjust hyperparameters to reduce model variance.
* Perhaps use repeated evaluations (e.g. repeated k-fold cross-validation or repeated train/test splits) and report a performance distribution instead of a point estimate.

**Q. What about a machine learning competition, like kaggle?**

Nope.

Your model must generalize to unseen data (e.g. the hidden test set) and optimizing for the public test set will likely (almost certainly) result in overfitting.

**Q. Surely picking the model random number seed that gives the best cross-validation score is a good idea?**

Nope.

It is likely that the difference in each distribution of CV scores is the same (e.g. check using a statistical hypothesis test, quantify using an effect size) and that any differences you are seeing are misleading.

If there are differences, your model may have a variance that is a little too high for the given quantity of training data. Add some bias (see above). Or the model is fragile/overfit the hold out test set of your test harness and will not generalize well to changes (e.g. changes to the data, changes to the model).

**Q. Okay, if I have to choose between two models, each fit with a different seed, I should choose the one with the better performance, right?**

Nope. See above.

**Q. Can we seed hack (grid search the seed) for a model within nested k-fold cross-validation?**

Oh man... I guess you could.

Again, I suspect that in most cases, any difference model performance distributions with a fixed vs optimized seed will not be statistically significant.

## Further Reading

Sometimes it helps to read how others are thinking through this issue:

### Papers

* [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305), 2020.
* [On Model Stability as a Function of Random Seed](https://arxiv.org/abs/1909.10447), 2019.
* [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533), 2012.
* [Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203), 2021.

### Essays / Blog Posts

* [Are random seeds hyperparameters?](https://andrewcharlesjones.github.io/journal/random-seed-hyperparameter.html)
* [Manipulating machine learning results with random state](https://towardsdatascience.com/manipulating-machine-learning-results-with-random-state-2a6f49b31081)
* [Optimizing the Random Seed](https://towardsdatascience.com/optimizing-the-random-seed-99a90bd272e)

### Discussion

* [Data folks of Reddit: How do you choose a random seed?](https://www.reddit.com/r/datascience/comments/17kxd5s/data_folks_of_reddit_how_do_you_choose_a_random/)
* [Do Deep Learning/Machine Learning papers use a fixed seed to report their results?](https://www.reddit.com/r/MachineLearning/comments/fbl9ho/discussion_do_deep_learningmachine_learning/)
* [Is it 'fair' to set a seed in a random forest regression to yield the highest accuracy?](https://stats.stackexchange.com/questions/341610/is-it-fair-to-set-a-seed-in-a-random-forest-regression-to-yield-the-highest-ac/)
* [Optimization of hyperparameters and seed](https://www.reddit.com/r/reinforcementlearning/comments/ptsbvb/optimization_of_hyperparameters_and_seed/)
* [Why is it valid to use CV to set parameters and hyperparameters but not seeds?](https://stats.stackexchange.com/questions/341619/why-is-it-valid-to-use-cv-to-set-parameters-and-hyperparameters-but-not-seeds)
* [XGBoost - "Optimizing Random Seed"](https://stats.stackexchange.com/questions/273230/xgboost-optimizing-random-seed)

