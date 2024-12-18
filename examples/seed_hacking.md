# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

## Description

Recall that the **random number seed** is an integer that initializes the [pseudo random number generator](https://en.wikipedia.org/wiki/Random_number_generation) and influences the specific and repeatable sequence of random numbers that are generated.

**Seed hacking** or **random seed shopping** or **seed optimization** is a problematic practice where practitioners manipulate random number seeds to artificially improve model performance metrics.

The technique involves repeatedly running the same experiment (e.g. model, data split, etc.) with different random seeds until finding one that produces better-looking results. This is typically done during model validation or testing phases.

While random seeds are important for reproducibility, exploiting them to [cherry-pick](https://en.wikipedia.org/wiki/Cherry_picking) favorable outcomes introduces severe bias. This practice masks the model's true performance and can lead to poor generalization in production.

The key issue is that seed hacking violates the principle of independent validation. By selecting seeds based on outcomes, you're effectively leaking information from your test set into your model selection process.

This practice is particularly dangerous for new data scientists because it can be tempting to use when under pressure to show improved metrics. However, it fundamentally undermines the scientific validity of your work.

A more ethical approach is to use fixed random seeds for reproducibility, but to select them before seeing any results. This maintains experimental integrity while still allowing others to replicate your work.

## What Does a Seed-Hacked Result Mean?

In a stochastic experiment, a single result is a point estimate of the unknown underlying distribution, such as the hold-out/test set prediction error.

If we repeat the experiment and vary the randomness (e.g., by using different random seeds for data splits or model initialization) we obtain a distribution of estimates. Taking the mean, standard deviation, or confidence interval of this distribution gives us a more accurate and reliable understanding of the model's true performance.

However, when we hack the seed to deliberately select the best possible result (e.g., lowest error or highest accuracy), we introduce [systematic bias](https://en.wikipedia.org/wiki/Observational_error). Rather than estimating the true mean of the performance distribution, we shift the estimate in a favorable direction.

The result is no longer a fair or unbiased reflection of the model's performance but instead an overoptimistic artifact of the chosen randomness. This shift can be substantial and misrepresent the model's real-world generalizability.

**Intentionally introducing a systematic bias by seed hacking is deceptive and misleading, perhaps fraud.**

Here's a depiction of what is happening when we pick a seed hacked result:

![seed hacked result](/pics/seed_hacked_result.svg)

## Examples

Below is a list of aspects of a data science project that could be subject to seed hacking:

- **Data Splitting**: Splitting datasets into training, validation, and testing sets. Shuffling data during cross-validation evaluation.
- **Resampling Techniques**: Bootstrapping or permutation tests. Creating synthetic datasets using resampling methods.
- **Learning Algorithms**: Initializing weights in neural networks. Randomly selecting subsets of data for ensemble methods like Random Forest or Bagging. Stochastic gradient descent and related stochastic optimization methods.
- **Hyperparameter Optimization**: Randomized search strategies for hyperparameter tuning. Distribution sampling search strategies like Bayesian Optimization.
- **Data Augmentation**: Random transformations for data augmentation in image or text preprocessing. Generating synthetic data for privacy-preserving data sharing or experimentation. Simulating data with specific statistical properties.
- **Feature Engineering**: Randomized feature selection or subset selection algorithms. Creating stochastic embeddings, e.g., in t-SNE or UMAP.

### Worked Examples

Some worked examples of seed hacking applied to specific aspects of a project:

* [Cross-Validation Hacking](cross_validation_hacking.md): _Vary the seed for creating cross-validation folds in order to get the best result._
* [Train/Test Split Hacking](train_test_split_hacking.md): _Vary the seed for creating train/test splits in order to get the best result._
* [Model Selection Hacking](model_selection_hacking.md): _Vary the seed for the model training algorithm in order to get the best result._
* [Performance Hacking](performance_hacking.md): _Vary the seed for a bootstrap of a final chosen model on the test set to present the best performance._

## Negative Seed Hacking

How can we defend the choice of random number seed on a project?

* Use a widely used default, e.g. 1 or 42 or 1234 or 1337.
* Use the current date as an integer, e.g. DDMMYYYY.
* Look at the clock and use the current minute and/or second value.
* Roll die and use the number that comes up.

Then record what you chose and how you chose it in your project log.

## Quantify the Variance

Don't guess or ignore the variance, measure and report it.

Perform a [sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis) aka stability/robustness study.

This involves:

1. Hold everything in your setup (data + model) constant.
2. Pick one aspect of your setup that uses randomness.
3. Vary the randomness for that one aspect (e.g. 30+ runs each with a different seed).
4. Collect performance scores and report/analyze the distribution (best + worst, mean + stdev, median + confidence interval, etc.).

For example:

1. **Hold the Model Constant and Vary the Data**: Use techniques like k-fold cross-validation (CV), repeated k-fold CV, or repeated train/test splits while keeping the model and its random initialization fixed.
    - Quantify how sensitive the model's performance is to variations in the training and test data splits.
    - This approach reveals variance caused by differences in the sampled training/test data and helps assess the model's robustness to data variability.
2. **Hold the Data Constant and Vary the Learning Algorithm**: Use a fixed dataset and vary only the random seed for the algorithm (e.g., random initialization of weights, dropout masks, or other stochastic elements).
    - Quantify how the inherent randomness in the learning process affects model performance.
    - This captures the variance caused by the stochastic nature of the optimization algorithm or training procedure.
3. **Vary Both the Data and the Learning Algorithm**: Randomize both the data (through k-fold CV or similar techniques) and the algorithm (through different seeds).
    - Assess the **total variance** in the learning process, encompassing both data variability and algorithm randomness.
    - This provides a holistic view of the overall variability in model performance.

How much variance to expect? It really depends.

- The variance due to data could be a few percent (e.g. 1-2%).
- The variance due to learning algorithm could be a few tenths of a percent to a few percent (e.g. 0.2-0.4% or 1-2%).

## Reduce the Variance

Variance is reduced by adding bias, we cannot escape the [Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff).

The techniques for reducing variance are typically specific to your setup, especially your model.

Nevertheless, here are some ideas:

1. **Reducing Performance Variance Due to Data**. Variance from data arises because models are sensitive to the specific training or test samples provided. Strategies to mitigate this include:
    - Regularization: Penalize model complexity to prevent overfitting to specific data splits.
    - Use More Data: Larger datasets typically reduce variability by making training samples more representative of the underlying distribution.
    - Robust Models: Use algorithms known for robustness to outliers or data variability, such as tree-based methods (e.g., Random Forests, Gradient Boosting).
    - ...
2. **Reducing Performance Variance Due to the Learning Algorithm**. Variance from the learning algorithm stems from stochasticity in the optimization process, such as random initialization, batch sampling, or other internal randomness. Strategies to reduce this variance include:
    - Ensembles: Combine predictions from multiple models trained on the same data but with different initializations or configurations.
    - Repeated Training and Averaging: Train the model multiple times with different seeds and average the predictions for a more robust output (simplest ensemble).
    - Better Initialization: Use advanced initialization techniques, such as Xavier or He initialization, to reduce sensitivity to starting conditions.
    - Use Stable Optimizers: Certain optimizers, such as AdamW or SGD with carefully tuned learning rates, can provide more consistent convergence compared to others.
    - Longer Training with Early Stopping: Allow models more time to converge but use early stopping based on validation performance to avoid overfitting.
    - ...
3. **Reducing Overall Variance (Both Data and Algorithm)**. For a holistic reduction in variance, consider strategies that address both data and algorithm variability:
    - Use Cross-Validation: Perform k-fold cross-validation to average out performance over different data splits and initialization seeds.
    - Hybrid Ensembles: Combine models trained on different data subsets (bagging) with models using different algorithm configurations or seeds.

For a best practice approach, combine strategies:

- Regularize the model and preprocess data to reduce data-driven variance.
- Use ensembles or repeated runs to reduce algorithm-driven variance.
- Report distributions of performance metrics to transparently communicate variability.

## What About Large One-Off Models (e.g. neural nets)?

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

> A common approach to creating neural network ensembles is to train the same architecture with different random seeds, and have the resulting models vote.

-- [We need to talk about random seeds](https://arxiv.org/abs/2210.13393), 2022.

One consolation is that a converged neural network model generally has a narrow distribution of performance across random seeds (as we might hope and expect).

> What is the distribution of scores with respect to the choice of seed? The distribution of accuracy when varying seeds is relatively pointy, which means that results are fairly concentrated around the mean. Once the model converged, this distribution is relatively stable which means that some seed are intrinsically better than others.

-- [Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203), 2021.

And:

> Typically, the choice of random seed only has a slight effect on the result and can mostly be ignored in general or for most of the hyper-parameter search process.

-- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533), 2012.

If you can perform multiple training runs for your neural network model, then you should, **with different random number seeds**.

This is called multiple-restart optimization, see below.

## When is Seed Hacking Ethical?

Is there such a thing as ethical seed hacking (in machine learning/data science)?

YES!

Here are some softer rationales:

* Perhaps you want to make a point for a demonstration, presentation, course, tutorial, etc.?
* Perhaps you require the best descriptive rather than predictive model?
* Perhaps you want to find best/worst/distribution performance due to learning algorithm/initial condition variance (e.g. a sensitivity analysis)?

The best case for seed hacking is as an stochastic optimization strategy called "multiple-restarts":

* Some learning algorithms are solving a really hard (e.g. non-convex/discontinuous/deceptive/multimodal/etc.) optimization problem and random restarts of initial conditions in the search space is in fact a beneficial approach.

### Multiple-Restart Optimization

The multiple-restart strategy is a technique used to address the challenges of solving harder optimization problems, particularly non-convex ones with multiple local minima, saddle points, or other complex structures.

By running the optimization process multiple times with different initial conditions or random seeds, this approach increases the likelihood of exploring diverse regions of the solution space and finding better optima.

> Heuristic search procedures that aspire to find global optimal solutions to hard combinatorial optimization problems usually require some type of diversification to overcome local optimality. One way to achieve diversification is to re-start the procedure from a new solution once a region has been explored.

-- [Chapter 12: Multi-Start Methods](https://link.springer.com/chapter/10.1007/0-306-48056-5_12), Handbook of Metaheuristics, 2003.

It is especially beneficial for algorithms that are sensitive to initialization, such as neural networks, clustering methods (e.g., K-Means), or stochastic optimization algorithms.

While multi-restart offers significant advantages for non-convex and multimodal problems, it provides little to no benefit for convex optimization problems, where the global minimum is guaranteed regardless of the starting point.

The strategy effectively balances computational cost with solution quality in scenarios where optimality cannot be guaranteed in a single run.

Below is a table of common machine learning algorithms, the type of optimization problem they are solving (e.g. convex or non-convex), whether they are sensitive to initial conditions, and whether they will benefit from multiple restarts:

| Algorithm                | Problem Type    | Sensitivity | Multi-Restart Benefit |
|--------------------------|-----------------|-------------|-----------------------|
| Linear Regression        | Convex          | None        | None                  |
| Logistic Regression      | Convex          | Minimal     | Minimal               |
| K-Means                  | Non-convex      | High        | High                  |
| t-SNE                    | Non-convex      | High        | High                  |
| Neural Networks          | Non-convex      | High        | High                  |
| Random Forests           | Non-convex      | Low         | Low to Moderate       |
| SVC                      | Convex          | None        | None                  |
| PCA                      | Convex          | None        | None                  |


As such, we may see what looks like seed hacking in the context of deep learning / reinforcement learning work, which may in fact be examples of a multiple-restart optimization.

The problem is, how do you tell the difference?

### Seed Hacking vs Multiple-Restarts

Differentiating between a legitimate multi-restart optimization strategy and "seed hacking" (cherry-picking the best result) requires careful scrutiny of how the results are reported and interpreted.

Below are the characteristics of **legitimate multi-restart optimization**:

1. **Disclosure of Multi-Restart Process**: Clearly states that a multi-restart strategy was employed and describes the number of restarts, initialization strategy, and hyperparameters.
2. **Performance Distribution Reporting**: Reports the distribution of performance metrics across restarts, including mean, median, standard deviation, and possibly full histograms or box plots. This allows readers to assess the stability of the algorithm and whether the best result is an outlier or representative of typical performance.
3. **Procedure Replication:** If the "best result" is highlighted, it contextualizes this by repeating the entire multi-restart procedure multiple times and reporting the distribution of "best-of-restart" scores. This provides confidence that the approach is not a one-off fluke.
4. **Statistical Robustness:** Includes statistical tests to verify whether improvements from the best restart are statistically significant compared to baselines or other algorithms.
5. **Sensitivity Analysis:** Reports how sensitive the algorithm is to random initialization, demonstrating whether consistent performance can be expected or if results are highly variable.

Conversely, below are the characteristics of **seed hacking with multi-restart optimization**:

1. **Single Point Estimate:** Reports only the best result without contextualizing it within the broader distribution of outcomes across restarts. This ignores variability and may cherry-pick an optimistic outlier.
2. **Non-Disclosure of Multi-Restart:** Fails to disclose that multiple restarts or seeds were used. This gives the impression that the reported result comes from a single unbiased run.
3. **Absence of Distribution Information:** Does not provide statistics (e.g., mean, standard deviation, quantiles) of performance across restarts. This lacks transparency on how consistently high-quality solutions are found.
4. **Selective Comparisons:** Compares the "best restart" of one algorithm with the "average performance" of another algorithm or baseline, creating unfair comparisons.


## FAQ

I get a lot of questions about "how to pick the best seed". Some of the answers below may help.


**Q. Is the random seed a hyperparameter?**

Yes.

It is a hyperparameter (to the model, to the test harness, etc.) that we should set, but not one we should optimize.

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

**Q. What about a machine learning competition, like Kaggle?**

Nope, or probably not.

Your model must generalize to unseen data (e.g. the hidden test set) and optimizing for the public test set will likely (almost certainly) result in overfitting.

**Q. Surely picking the model random number seed that gives the best cross-validation score is a good idea?**

Nope, or probably not.

It is likely that the difference in each distribution of CV scores is the same (e.g. check using a statistical hypothesis test, quantify using an effect size) and that any differences you are seeing are misleading.

If there are differences, your model may have a variance that is a little too high for the given quantity of training data. Add some bias (see above). Or the model is fragile/overfit the hold out test set of your test harness and will not generalize well to changes (e.g. changes to the data, changes to the model).

**Q. Okay, if I have to choose between two models, each fit with a different seed, I should choose the one with the better performance, right?**

Nope, or probably not. See above.

**Q. Can we seed hack (grid search the seed) for a model within nested k-fold cross-validation?**

Oh man... I guess you could.

Again, I suspect that in most cases, any difference between model performance distributions with a fixed vs optimized seed will not be statistically significant.

If it is different, perhaps use methods to reduce model variance as discussed above.

If it's for an algorithm with a non-convex optimization problem, think about what this means. It means that one initial condition performs "better" across different subsets of train/test data. Maybe it's true, it's probably not.

**Q. How do I know if my seed hacked result is optimistically biased or a better solution to a hard optimization problem?**

Now that is a good question!

If we know a lot about the model and its optimization procedure, we might be able to draw a logical conclusion because of the underlying optimization problem the learning algorithm is solving (e.g. convex vs non-convex and to what degree it is sensitive to initial conditions or stochastic behaviors during the search).

For example:

- Did the change in seed permit the optimization algorithm locate a superior solution in the search space (if so, can you confirm with statistical tests)?

Empirically, you can sample results for a ton of seeds and see where you sit on the distribution. All that tells you is what result percentile you might be in, not whether the solution is brittle.

This is really hard and an "it depends" is the best I can manage.

See the sections above on "Multiple-Restart Optimization" and "Seed Hacking vs Multiple-Restarts".

## Further Reading

Sometimes it helps to read how others are thinking through this issue:

### Papers

* [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305), 2020.
* [Multi-Start Methods](https://link.springer.com/chapter/10.1007/0-306-48056-5_12), Handbook of Metaheuristics, 2003.
* [On Model Stability as a Function of Random Seed](https://arxiv.org/abs/1909.10447), 2019.
* [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533), 2012.
* [Pseudo-random Number Generator Influences on Average Treatment Effect Estimates Obtained with Machine Learning](https://pubmed.ncbi.nlm.nih.gov/39150879/), 2024.
* [Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203), 2021.
* [We need to talk about random seeds](https://arxiv.org/abs/2210.13393), 2022.

### Blog Posts

* [Are random seeds hyperparameters?](https://andrewcharlesjones.github.io/journal/random-seed-hyperparameter.html)
* [Manipulating machine learning results with random state](https://towardsdatascience.com/manipulating-machine-learning-results-with-random-state-2a6f49b31081)
* [Optimizing the Random Seed](https://towardsdatascience.com/optimizing-the-random-seed-99a90bd272e)

### Discussion

Lots of people struggling with choosing/optimizing the random seed out there in the wild. Not enough background in statistics/stochastic optimization IMHO, but that's okay.

* [Am I creating bias by using the same random seed over and over?](https://stats.stackexchange.com/questions/80407/am-i-creating-bias-by-using-the-same-random-seed-over-and-over)
* [Choosing the "Correct" Seed for Reproducible Research/Results](https://stats.stackexchange.com/questions/335936/choosing-the-correct-seed-for-reproducible-research-results)
* [Data folks of Reddit: How do you choose a random seed?](https://www.reddit.com/r/datascience/comments/17kxd5s/data_folks_of_reddit_how_do_you_choose_a_random/)
* [Do Deep Learning/Machine Learning papers use a fixed seed to report their results?](https://www.reddit.com/r/MachineLearning/comments/fbl9ho/discussion_do_deep_learningmachine_learning/)
* [How to choose the random seed?](https://datascience.stackexchange.com/questions/35869/how-to-choose-the-random-seed)
* [How to deal with random parameters in MLOps](https://stats.stackexchange.com/questions/564045/how-to-deal-with-random-parameters-in-mlops)
* [If so many people use set.seed(123) doesn't that affect randomness of world's reporting?](https://stats.stackexchange.com/questions/205961/if-so-many-people-use-set-seed123-doesnt-that-affect-randomness-of-worlds-re)
* [Is it 'fair' to set a seed in a random forest regression to yield the highest accuracy?](https://stats.stackexchange.com/questions/341610/is-it-fair-to-set-a-seed-in-a-random-forest-regression-to-yield-the-highest-ac/)
* [Is random seed a hyper-parameter to tune in training deep neural network?](https://stats.stackexchange.com/questions/478193/is-random-seed-a-hyper-parameter-to-tune-in-training-deep-neural-network)
* [Is random state a parameter to tune?](https://stats.stackexchange.com/questions/263999/is-random-state-a-parameter-to-tune)
* [Neural network hyperparameter tuning - is setting random seed a good idea?](https://stackoverflow.com/questions/65704588/neural-network-hyperparameter-tuning-is-setting-random-seed-a-good-idea)
* [Optimization of hyperparameters and seed](https://www.reddit.com/r/reinforcementlearning/comments/ptsbvb/optimization_of_hyperparameters_and_seed/)
* [Performance of Ridge and Lasso Regression depend on set.seed?](https://stats.stackexchange.com/questions/355256/performance-of-ridge-and-lasso-regression-depend-on-set-seed)
* [Why is it valid to use CV to set parameters and hyperparameters but not seeds?](https://stats.stackexchange.com/questions/341619/why-is-it-valid-to-use-cv-to-set-parameters-and-hyperparameters-but-not-seeds)
* [XGBoost - "Optimizing Random Seed"](https://stats.stackexchange.com/questions/273230/xgboost-optimizing-random-seed)



