 # Test Harness Hacking

> Varying models and hyperparameters to maximize test harness performance at the cost of reduced generalizability.

## Description

When multiple hypotheses, models, or configurations are tested on the same dataset or evaluation framework (test harness), there is a high risk of fitting to the noise or idiosyncrasies of the data rather than uncovering true, generalizable patterns.

This leads to inflated performance estimates that do not hold when the model is applied to unseen data.

This issue is known by many names, such as:

* **Comparing Too Many Hypotheses** / **Checking Too Many Models**: Testing numerous hypotheses or model configurations increases the chance of finding a model that performs well on the test data by coincidence rather than due to its inherent quality.
* **Multiple Comparison Problem** / **Multiple Hypothesis Testing**: A statistical issue where testing multiple hypotheses increases the probability of false positives (e.g., identifying a model as superior when it's not).
* **Oversearching**: Excessive experimentation with hyperparameters, architectures, or algorithms can lead to "discovering" patterns that are not generalizable.
* **Overfitting Model Selection**: When the process of selecting the best model overfits to the evaluation dataset, the chosen model's reported performance becomes unreliable.
* **Test Harness Hacking**: Manipulating the evaluation process, such as by repeatedly tweaking models or hyperparameters, to artificially inflate test harness performance.

Ideally (from a statistical perspective), candidate hypotheses (models) would be selected for a predictive modeling problem _before_ data is gathered, not after and not adapted to the problem in response to results on the test harness.

> ... the theory of statistical inference assumes a fixed collection of hypotheses to be tested, or learning algorithms to be applied, selected non-adaptively before the data are gathered, whereas in practice data is shared and reused with hypotheses and new analyses being generated on the basis of data exploration and the outcomes of previous analyses.

-- [Preserving Statistical Validity in Adaptive Data Analysis](https://arxiv.org/abs/1411.2664), 2014.

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

## Examples

Below are some examples of test harness hacking.

* [Hill Climb Cross-Validation Test Folds](test_harness_hacking_hill_climbing_test_folds.md): Adapt predictions for each cross-validation test fold over repeated trials.
* [Hill Climb Cross-Validation Performance](test_harness_hacking_hill_climbing_performance.md): Excessively adapt a model for cross-validation performance.
* [Test Harness Hacking Mitigation](test_harness_hacking_mitigation.md): Modern practices (repeated k-fold cv) mitigates the risk of test harness hacking.

## Impact

The impact of overfitting the test harness manifests as **optimistic bias** in the performance of the chosen model.

Here's how this unfold in a machine learning project:

1. **Overfitting to the Test Harness**: Through repeated tuning or evaluation on the test harness, the chosen model exploits idiosyncrasies in the validation/test set rather than learning generalizable patterns.
2. **Optimistic Performance Estimates**: The model appears to perform exceptionally well on the test harness, creating a false sense of superiority over other models.
3. **Final Model Evaluation**: When the model is retrained on all available training data and evaluated on a hold-out test set (or deployed in real-world scenarios), its performance is often significantly lower than expected. This happens because the model's improvements on the test harness were based on fitting noise or dataset-specific artifacts.
4. **Missed Opportunities**: Other models that may generalize better but were overlooked during evaluation (due to lower but more realistic performance on the test harness) might have been more suitable in practice.

## Push-Back

It is possible that the issue of "too many model comparisons" is overblown in modern machine learning.

This may be because the techniques that mitigate this type of overfitting have become best practices, such as:

* Adoption of k-fold cross-validation in the test harness.
* Adoption of repeated cross-validation to further reduce variance in performance estimates.
* Adoption of nested cross-validation, to tune hyperparameters within each cross-validation fold.
* Adoption of corrections to cross-validation when used for model selection (e.g. 1 standard error rule).
* Adoption of statistical hypothesis tests to support differences in model performance on the test harness.
* Adoption of modern machine learning learning algorithms that use regularization, early stopping and similar methods.

> The adaptive data analysis literature provides a range of theoretical explanations for how the common machine learning workflow may implicitly mitigate overfitting

-- [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html), 2019.

And:

> We propose that the computational cost of performing repeated cross-validation and nested cross-validation in the cloud have reached a level where the use of substitutes to full nested cross-validation are no longer justified.

-- [Cross-validation pitfalls when selecting and assessing regression and classification models](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10), 2014.

And:

> Often a “one-standard error” rule is used with cross-validation, in which we choose the most parsimonious model whose error is no more than one standard error above the error of the best model.

-- Page 244, [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), 2016.

As such, overfitting the test harness may be less of a concern than it once was one or two decades ago in applied machine learning.

Evidence for this is seen in large-scale machine learning competitions, like those on Kaggle.

> In each competition, numerous practitioners repeatedly evaluated their progress against a holdout set that forms the basis of a public ranking available throughout the competition. Performance on a separate test set used only once determined the final ranking. By systematically comparing the public ranking with the final ranking, we assess how much participants adapted to the holdout set over the course of a competition. Our study shows, somewhat surprisingly, little evidence of substantial overfitting.

-- [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html), 2019.

And:

> Overall, we conclude that the classification competitions on Kaggle show little to no signs of overfitting. While there are some outlier competitions in the data, these competitions usually have pathologies such as non-i.i.d. data splits or (effectively) small test sets. Among the remaining competitions, the public and private test scores show a remarkably good correspondence. The picture becomes more nuanced among the highest scoring submissions, but the overall effect sizes of (potential) overfitting are typically small (e.g., less than 1% classification accuracy). Thus, our findings show that substantial overfitting is unlikely to occur naturally in regular machine learning workflows.

-- [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html), 2019.

Additional evidence for this is seen in popular computer vision deep learning benchmark datasets on which continued performance, rather than overfitting, is observed.

> Recent replication studies [16] demonstrated that the popular CIFAR-10 and ImageNet benchmarks continue to support progress despite years of intensive use. The longevity of these benchmarks perhaps suggests that overfitting to holdout data is less of a concern than reasoning from first principles might have suggested.

-- [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html), 2019.

These findings suggest that test-harness hacking may be achieved by intentionally not observing modern best practices like those listed above.

## Further Reading

* [A Meta-Analysis of Overfitting in Machine Learning](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html), 2019.
* [Cross-validation pitfalls when selecting and assessing regression and classification models](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10), 2014.
* [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811), 2019.
* [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://arxiv.org/abs/1811.12808), 2018.
* [Model Similarity Mitigates Test Set Overuse](https://arxiv.org/abs/1905.12580), 2019.
* [Multiple Comparisons in Induction Algorithms](https://link.springer.com/article/10.1023/A:1007631014630), 2000.
* [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), 2010.
* [Preserving Statistical Validity in Adaptive Data Analysis](https://arxiv.org/abs/1411.2664), 2014.
* [Preventing "Overfitting" of Cross-Validation Data](https://ai.stanford.edu/~ang/papers/cv-final.pdf), 1997.
* [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), 2016.
