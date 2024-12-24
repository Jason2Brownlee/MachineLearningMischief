 # Test Harness Hacking

> Varying models and hyperparameters to maximize test harness performance at the cost of reduced generalizability.

## Description

When multiple hypotheses, models, or configurations are tested on the same dataset or evaluation framework (test harness), there is a high risk of fitting to the noise or idiosyncrasies of the data rather than uncovering true, generalizable patterns.

This leads to inflated performance estimates that do not hold when the model is applied to unseen data.

This issue is known by many names, such as:

* **Comparing Too Many Hypotheses** / **Checking Too Many Models**: Testing numerous hypotheses or model configurations increases the chance of finding a model that performs well on the test data by coincidence rather than due to its inherent quality.
* **Multiple Comparison Problem**: A statistical issue where testing multiple hypotheses increases the probability of false positives (e.g., identifying a model as superior when it's not).
* **Oversearching**: Excessive experimentation with hyperparameters, architectures, or algorithms can lead to "discovering" patterns that are not generalizable.
* **Overfitting Model Selection**: When the process of selecting the best model overfits to the evaluation dataset, the chosen model's reported performance becomes unreliable.
* **Overfitting the Test Harness**: When models are fine-tuned or selected based on a fixed test set, the test set becomes part of the training process, undermining its role as an unbiased estimator of generalization.
* **Test Harness Hacking**: Manipulating the evaluation process, such as by repeatedly tweaking models or hyperparameters, to artificially inflate test set performance.

## Scenario

A test-setup (specific data, model, and test harness) may be more or less subject to this problem.

The aspects that exasperate this problem include:

* Small dataset.
* Large number of candidate models.
* Large number of candidate model hyperparameter combinations.
* High variance test harness.

> It seems reasonable to suggest that over-fitting in model selection is possible whenever a model selection criterion evaluated over a finite sample of data is directly optimised. Like over-fitting in training, over-fitting in model selection is likely to be most severe when the sample of data is small and the number of hyper-parameters to be tuned is relatively large.

-- [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), 2010.

The risk is that the variance in model performance on the test harness will result in an optimistic basis (i.e. model's look better than they are).

This bias may be larger than the difference in performance between performance estimates of different models on the test harness, resulting in Type I errors (false positive) in model selection.

> The scale of the bias observed on some data sets is much larger than the difference in performance between learning algorithms, and so one could easily draw incorrect inferences based on the results obtained.

-- [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), 2010.

We can depict this scenario with an idealized plot, below.

### Graphical Depiction

![test harness hacking](/pics/test_harness_hacking.png)

The plot above illustrates the distributions of performance metrics for two algorithms:

- **Algorithm A ("Chosen Algorithm")**:
  - Slightly higher mean performance (75).
  - Larger variance (10).

- **Algorithm B ("Alternative Algorithm")**:
  - Slightly lower mean performance (72).
  - Smaller variance (5).

Even though Algorithm A is chosen due to its slightly higher mean performance, the variance in its performance is large enough that the difference in means may not be practically significant.

This underscores the importance of considering the variability in performance and not relying solely on mean values for decision-making.



## Further Reading

* [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html), 2019.
* [Multiple Comparisons in Induction Algorithms](https://link.springer.com/article/10.1023/A:1007631014630), 2000.
* [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), 2010.
* [Preventing "Overfitting" of Cross-Validation Data](https://ai.stanford.edu/~ang/papers/cv-final.pdf)