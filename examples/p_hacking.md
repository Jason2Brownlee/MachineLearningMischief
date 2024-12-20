# p-Hacking

> Repeating a statistical hypothesis test until a significant result is achieved.

## Description

P-hacking is the practice of manipulating data analysis until you achieve a statistically significant result, typically to support a predetermined conclusion.

This approach involves running multiple statistical tests on the same dataset, selectively choosing which data points to include, or adjusting variables until achieving the desired p-value (typically < 0.05).

While it may seem tempting to keep testing until you get "significant" results, p-hacking invalidates the fundamental principles of statistical testing and leads to false discoveries.

The danger lies in increasing the likelihood of Type I errors (false positives) through multiple comparisons, making spurious correlations appear meaningful when they're actually due to random chance.

For new data scientists, this pattern often emerges unintentionally when there's pressure to find significant results or when dealing with stakeholder expectations for positive outcomes.

To avoid p-hacking, define your hypothesis and analysis plan before examining the data, use correction methods for multiple comparisons, and be transparent about all tests performed - including those that didn't yield significant results.

Remember that negative results are valid scientific outcomes and should be reported alongside positive findings to maintain research integrity.

## Cases of p-hacking

Any time we want to use a statistical hypothesis test to compare two samples in a data science/machine learning project, this represents a point for p-hacking.

Common cases include:

- Comparing data sub-samples by impact on model performance.
- Comparing subsets of input features by correlation with the target or impact on model performance.
- Comparing the performance of models based on cross-validated performance.
- Comparing the performance of a model with different hyperparameters.

P-hacking requires varying something in the experiment to produce a distribution of samples that 1) give a result (such as the sample mean) that is "better" and 2) give a result that has a p-value (as calculated by a statistical test) below the threshold (i.e. significant).

The aspect varied is often the seed for the pseudorandom number generator, such as when varying the a sampling procedure or learning algorithm. As such, many cases of p-hacking also require [seed hacking](seed_hacking.md).

### Worked Examples of p-Hacking

Below are some worked examples of p-hacking in a data science/machine learning project.

* [p-Hacking Selective Sampling](p_hacking_selective_sampling.md): _Vary samples of a dataset in order to fit a model with significantly better performance._
* [p-Hacking Feature Selection](p_hacking_feature_selection.md): _Vary feature subsets of a dataset in order to fit a model with significantly better performance._
* [p-Hacking the Learning Algorithm](p_hacking_learning_algorithm.md) _Vary the random numbers used by a learning algorithm in order to get a significantly better result._

## p-Hacking vs Normal Experimentation

**What is p-hacking, and what is normal experimental variation in a machine learning project?**

P-hacking in a machine learning project and the normal variation of aspects of a machine learning pipeline share similarities in that both involve systematically exploring different configurations, datasets, or techniques to optimize results.

However, they differ significantly in their intent, methodology, and implications.

Here's an attempt at a comparison:

### Intent of Experimentation (intent matters!)
- **P-Hacking:**
  - The primary goal is often to achieve statistical significance and a desired result (e.g., a low p-value and an improved metric), even at the cost of scientific or experimental integrity.
  - It reflects a bias towards confirming a hypothesis, regardless of whether the result is genuinely meaningful or reproducible.

- **Normal Variation:**
  - The goal is to genuinely identify the best-performing model or configuration while ensuring that findings are robust and reproducible.
  - The process is exploratory but grounded in a scientific approach to assess performance in a meaningful and unbiased manner.

### Methodology
- **P-Hacking:**
  - Involves deliberately cherry-picking or over-exploring configurations to obtain statistically significant results.
  - Examples include:
    - Running experiments on multiple datasets and only reporting the one that shows the desired results.
    - Trying numerous feature subsets or hyperparameters without a predefined protocol, then selecting the ones that yield significant outcomes.
    - Repeating experiments until statistical tests yield favorable results (e.g., p-values < 0.05).
  - Often lacks transparency, with omitted reporting of failed or contradictory experiments.

- **Normal Variation:**
  - Follows systematic and reproducible protocols for varying datasets, features, models, or hyperparameters.
  - Examples include:
    - Using predefined validation or test datasets to avoid bias.
    - Employing cross-validation or other robust evaluation techniques to ensure generalizability.
    - Applying grid search, random search, or Bayesian optimization for hyperparameter tuning within a controlled framework.
  - Results are typically presented comprehensively, including cases where configurations performed poorly.

### Evaluation and Reporting
- **P-Hacking:**
  - Relies heavily on statistical tests to "prove" a point, often without considering the broader context or reproducibility.
  - May selectively report results that confirm a hypothesis, leading to overfitting or misrepresentation.
  - Lacks emphasis on replicability; findings may not hold on unseen data or alternative setups.

- **Normal Variation:**
  - Focuses on evaluating performance through unbiased metrics like accuracy, F1 score, AUC, etc., on unseen test data.
  - Emphasizes transparency, reporting the entire spectrum of experiments (successful and unsuccessful) to give a holistic view.
  - Stresses reproducibility, often sharing code, data, and experimental protocols for verification by others.

### Impact
- **P-Hacking:**
  - Can lead to misleading conclusions, potentially wasting resources or eroding trust in the findings.
  - Results are often fragile and fail to generalize beyond the specific experimental conditions.
  - Undermines scientific and ethical standards in research.

- **Normal Variation:**
  - Helps identify robust and reliable configurations that generalize well to new data.
  - Builds confidence in findings and advances the field by sharing insights into what works and what does not.
  - Adheres to principles of transparency, integrity, and reproducibility.

### **Key Distinction**
The fundamental difference lies in **integrity and intent**.

P-hacking prioritizes achieving "impressive" results at the expense of scientific rigor, often through selective reporting and overfitting.

In contrast, normal variation is a legitimate and scientifically sound process to explore and optimize machine learning pipelines, grounded in transparency and reproducibility.

### Mitigation Strategies
To avoid unintentional p-hacking while exploring variations in machine learning projects:
- Use rigorous protocols such as cross-validation and pre-registered experiments.
- Report all experiments, including those that yield negative or inconclusive results.
- Evaluate findings on independent test sets that were not used during the exploratory phase.
- Avoid over-reliance on statistical significance as the sole criterion for evaluating results; consider practical significance and generalizability.

### Summary

| **Aspect**        | **P-Hacking**                                                                 | **Normal Variation**                                                         |
|--------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Intent**         | Achieve desired results (e.g., statistical significance) at the cost of integrity. | Identify the best-performing configuration while ensuring robustness.        |
| **Methodology**    | Cherry-picks or over-explores configurations to obtain favorable outcomes.   | Systematically and reproducibly explores configurations.                     |
|                    | Lacks transparency; omits reporting of failures.                             | Follows predefined protocols; reports successes and failures comprehensively.|
| **Evaluation**     | Focuses on statistical tests to confirm hypotheses, often ignoring context. | Evaluates unbiased metrics (e.g., accuracy, F1 score) on unseen test data.   |
|                    | Selectively reports results that support the hypothesis.                    | Reports entire spectrum of experiments for transparency.                     |
| **Reporting**      | Results often fail to generalize; lacks reproducibility.                    | Stresses reproducibility; shares code, data, and protocols.                  |
| **Impact**         | Misleading conclusions, wasted resources, erosion of trust.                 | Robust findings, confidence in results, adherence to ethical standards.      |
| **Key Distinction**| Prioritizes "impressive" results over scientific rigor.                     | Prioritizes transparency, integrity, and reproducibility.                    |
| **Mitigation**     | Avoid pre-defined protocols; over-rely on statistical tests.                | Use cross-validation, independent test sets, and report all experiments.     |




## Further Reading

* [Data dredging](https://en.wikipedia.org/wiki/Data_dredging), Wikipedia.
* [The Extent and Consequences of P-Hacking in Science](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002106&), 2015.
* [Big little lies: a compendium and simulation of p-hacking strategies](https://royalsocietypublishing.org/doi/10.1098/rsos.220346), 2023.
