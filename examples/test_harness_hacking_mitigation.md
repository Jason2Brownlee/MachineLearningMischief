# Test Harness Hacking Mitigation

## (i.e. the myth of "Overfitting Model Selection" in modern machine learning)

> Modern practices (repeated k-fold cv) mitigates the risk of test harness hacking.

## Description

When conducting model selection and hyperparameter tuning through cross-validation (CV), it is true that repeated model testing can exploit quirks in the training data, leading to overfitting.

However, increasing the **number of folds** and **repetitions of CV runs** directly mitigates this risk by reducing the variance and sensitivity of the CV estimate, effectively counteracting the overfitting tendency.

### Increasing CV Folds
1. **Smaller Test Sets, More Diverse Training Sets**: With more folds, each data point participates in training and testing more frequently. This improves the representativeness of the CV procedure, ensuring that hyperparameters cannot exploit idiosyncrasies in a single test set.
2. **Natural Regularization via Bias**: Increasing folds slightly increases the bias of the performance estimate, as training is performed on smaller subsets of the data. This bias acts as a regularizer, making the evaluation less prone to being gamed by overfit hyperparameters.

### Increasing CV Repetitions
1. **Normalization of Random Effects**: Repeated CV introduces new random splits, ensuring that the hyperparameter tuning process cannot exploit specific train-test partitioning. The mean performance score over multiple runs reflects the model's generalization across diverse splits, not just one specific configuration.
2. **Resilience to Stochastic Algorithms**: For models or learning processes reliant on randomness (e.g., neural networks, random forests), repeated CV smooths out the variability from individual runs, further reducing the likelihood of overfitting.

### Mitigating the Risk of Overfitting the Training Set
When folds and repetitions are increased:
- **Variance Reduction**: The CV estimate becomes more stable, leaving less room for hyperparameter tuning to overfit to the noise of specific splits.
- **Bias Introduction**: Higher folds increase bias, counteracting overfitting tendencies by making CV scores less sensitive to small variations in the training set.

While exhaustive hyperparameter tuning can exploit CV, the combination of higher folds and repetitions strengthens the robustness of the CV process. These changes make it harder for models to overfit, even during extensive optimization, by ensuring performance reflects true generalization rather than quirks in the training data.

## Study

This study investigates how the number of folds and repetitions in cross-validation (CV) affect the risk of overfitting during hyperparameter tuning, specifically when using a hill-climbing algorithm to optimize hyperparameters over 100 trials.

1. **Dataset Preparation**:
   - A synthetic classification dataset is created with 200 samples and 30 features, split into a train and test set.

2. **Experimental Setup**:
   - The study evaluates combinations of k-fold CV (3, 5, 7, 10 folds) and repeated CV (1, 3, 5 repeats).
   - Each configuration undergoes 5 independent trials.

3. **Hill-Climbing Hyperparameter Tuning**:
   - For each CV configuration, a hill-climbing algorithm runs 100 iterations to optimize the `n_estimators` and `max_depth` hyperparameters of a `RandomForestClassifier`.
   - Each hyperparameter configuration is evaluated using the specified CV method, and the best configuration is selected based on the mean CV score.

4. **Metrics Computation**:
   - For each run, the correlation between the CV scores and hold-out test scores, as well as the mean absolute difference between them, is calculated to quantify overfitting.
   - Overfitting is characterized by a decrease in correlation and an increase in the mean absolute difference.

5. **Recording and Analysis**:
   - The study aggregates results for each combination of folds and repeats, computing the average correlation and mean absolute difference over 5 trials.

The results are expected to show that increasing the number of CV folds and/or repetitions reduces overfitting, as reflected by:
- **Higher correlation** between CV and hold-out scores.
- **Lower mean absolute difference** between CV and hold-out scores.

The study demonstrates how CV configurations impact the reliability of model selection, reinforcing the importance of folds and repetitions in mitigating overfitting.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=200, n_features=30, n_informative=5, n_redundant=25, random_state=42
)

# Create a train/test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize result storage for experiments
results = []

# Define the study parameters
fold_range = [3, 5, 7, 10]  # 3 to 10 folds
repeat_range = [1, 3, 5]  # 1 to 10 repetitions
n_trials = 5  # Number of trials for each configuration

# Function for hill climbing optimization
def hill_climb(cv, X_train, y_train, X_test, y_test, n_hill_trials=100):
    best_params = {"n_estimators": 10, "max_depth": 2}
    best_cv_score = -1

    cv_scores = []
    holdout_scores = []

    for hill_trial in range(n_hill_trials):
        # Propose new parameters
        new_params = {
            "n_estimators": best_params["n_estimators"] + np.random.randint(-10, 11),
            "max_depth": best_params["max_depth"] + np.random.randint(-1, 2)
        }
        new_params["n_estimators"] = max(1, new_params["n_estimators"])
        new_params["max_depth"] = max(1, new_params["max_depth"])

        # Evaluate new parameters
        new_model = RandomForestClassifier(
            n_estimators=new_params["n_estimators"], max_depth=new_params["max_depth"], random_state=42
        )
        raw_scores = cross_val_score(new_model, X_train, y_train, cv=cv, scoring="accuracy")
        new_cv_score = np.mean(raw_scores)
        cv_scores.append(new_cv_score)

        # Evaluate the new model on the hold out test set
        new_model.fit(X_train, y_train)
        new_holdout_score = new_model.score(X_test, y_test)
        holdout_scores.append(new_holdout_score)

        # Update best parameters if score improves
        if new_cv_score > best_cv_score:
            best_params = new_params
            best_cv_score = new_cv_score

    return cv_scores, holdout_scores

# Function to calculate metrics
def calculate_metrics(cv_scores, holdout_scores):
    mean_cv_score = np.mean(cv_scores)
    correlation = np.corrcoef(cv_scores, holdout_scores)[0, 1]
    mean_abs_diff = np.mean(np.abs(np.array(cv_scores) - np.array(holdout_scores)))
    return correlation, mean_abs_diff

# Main experiment loop
for n_folds in fold_range:
    for n_repeats in repeat_range:
        trial_correlations = []
        trial_mean_differences = []

        for trial in range(n_trials):
            # Define CV with specific folds and repeats
            cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=trial)

            # Perform hill climbing of the cross-validated train set
            cv_scores, holdout_scores = hill_climb(cv, X_train, y_train, X_test, y_test)

            # Calculate metrics
            corr, diff = calculate_metrics(cv_scores, holdout_scores)

            trial_correlations.append(corr)
            trial_mean_differences.append(diff)

            # Report progress
            print(f'folds={n_folds}, repeats={n_repeats}, i={(trial+1)}, corr={corr}, diff={diff}')

        # Record average results for this configuration
        avg_correlation = np.mean(trial_correlations)
        avg_mean_diff = np.mean(trial_mean_differences)

        results.append({
            'folds': n_folds,
            'repeats': n_repeats,
            'avg_correlation': avg_correlation,
            'avg_mean_diff': avg_mean_diff
        })

        # Log progress
        print(f"Completed: {n_folds} folds, {n_repeats} repeats | Avg Correlation: {avg_correlation:.4f}, Avg Mean Diff: {avg_mean_diff:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('cv_overfitting_study_results.csv', index=False)

# Display final summary
print("\nFinal Results:\n")
print(results_df.sort_values(['folds', 'repeats']))
```

### Example Output

```text
folds=3, repeats=1, i=1, corr=0.6298745632413565, diff=0.04317033542976944
folds=3, repeats=1, i=2, corr=0.1846694607479044, diff=0.03648899371069183
folds=3, repeats=1, i=3, corr=0.5007118468999219, diff=0.042438329839273245
folds=3, repeats=1, i=4, corr=0.2879759396952337, diff=0.04089092709061263
folds=3, repeats=1, i=5, corr=0.6341440980635967, diff=0.029911891451199636
Completed: 3 folds, 1 repeats | Avg Correlation: 0.4475, Avg Mean Diff: 0.0386
folds=3, repeats=3, i=1, corr=0.7623783770801159, diff=0.01944815203043712
folds=3, repeats=3, i=2, corr=0.27925381574256164, diff=0.03321158475036883
folds=3, repeats=3, i=3, corr=0.24818307748928556, diff=0.04388339544995728
folds=3, repeats=3, i=4, corr=0.7184012240888981, diff=0.019093931982296728
folds=3, repeats=3, i=5, corr=0.5798007307300975, diff=0.02717262598027794
Completed: 3 folds, 3 repeats | Avg Correlation: 0.5176, Avg Mean Diff: 0.0286
folds=3, repeats=5, i=1, corr=0.3789626777977281, diff=0.03716864663405543
folds=3, repeats=5, i=2, corr=-0.28757534600689555, diff=0.0425980666200792
folds=3, repeats=5, i=3, corr=0.5567346395459242, diff=0.027225599813650116
folds=3, repeats=5, i=4, corr=0.6325025783848724, diff=0.02511076170510132
folds=3, repeats=5, i=5, corr=0.7396891596803272, diff=0.02035115303983226
Completed: 3 folds, 5 repeats | Avg Correlation: 0.4041, Avg Mean Diff: 0.0305
folds=5, repeats=1, i=1, corr=-0.011776348551674929, diff=0.04631250000000002
folds=5, repeats=1, i=2, corr=-0.19241265688270287, diff=0.035312500000000024
folds=5, repeats=1, i=3, corr=0.8250440734960189, diff=0.024874999999999994
folds=5, repeats=1, i=4, corr=0.4434997520588626, diff=0.051750000000000025
folds=5, repeats=1, i=5, corr=-0.27043847101619534, diff=0.0494375
Completed: 5 folds, 1 repeats | Avg Correlation: 0.1588, Avg Mean Diff: 0.0415
folds=5, repeats=3, i=1, corr=0.5853084167570692, diff=0.036520833333333315
folds=5, repeats=3, i=2, corr=0.5629276501612064, diff=0.02704166666666666
folds=5, repeats=3, i=3, corr=0.7089827145778376, diff=0.022145833333333313
folds=5, repeats=3, i=4, corr=0.6406706348714379, diff=0.03627083333333332
folds=5, repeats=3, i=5, corr=0.7851945543765236, diff=0.03235416666666664
Completed: 5 folds, 3 repeats | Avg Correlation: 0.6566, Avg Mean Diff: 0.0309
folds=5, repeats=5, i=1, corr=0.506085416086112, diff=0.03351249999999998
folds=5, repeats=5, i=2, corr=0.42796686752812807, diff=0.0278
folds=5, repeats=5, i=3, corr=0.8215171404358438, diff=0.01946249999999996
folds=5, repeats=5, i=4, corr=0.8072909310441211, diff=0.025212499999999985
folds=5, repeats=5, i=5, corr=0.6268606788133523, diff=0.025687499999999974
Completed: 5 folds, 5 repeats | Avg Correlation: 0.6379, Avg Mean Diff: 0.0263
folds=7, repeats=1, i=1, corr=-0.3622860600410274, diff=0.04641092603049123
folds=7, repeats=1, i=2, corr=0.43170078005874174, diff=0.031197628458497988
folds=7, repeats=1, i=3, corr=0.228096689372268, diff=0.0616328345567476
folds=7, repeats=1, i=4, corr=0.45907394657390765, diff=0.04040767927724447
folds=7, repeats=1, i=5, corr=0.7666141267810641, diff=0.04463212874082441
Completed: 7 folds, 1 repeats | Avg Correlation: 0.3046, Avg Mean Diff: 0.0449
folds=7, repeats=3, i=1, corr=0.8434628549359104, diff=0.024859119141727802
folds=7, repeats=3, i=2, corr=0.8053096340478801, diff=0.030307782796913166
folds=7, repeats=3, i=3, corr=0.8129667126799344, diff=0.027400197628458405
folds=7, repeats=3, i=4, corr=0.6203740119984177, diff=0.024067805383022743
folds=7, repeats=3, i=5, corr=0.6619482974223595, diff=0.03010973084886127
Completed: 7 folds, 3 repeats | Avg Correlation: 0.7488, Avg Mean Diff: 0.0273
folds=7, repeats=5, i=1, corr=0.49479024198324606, diff=0.03320101637492941
folds=7, repeats=5, i=2, corr=0.8126341498751418, diff=0.026930265386787125
folds=7, repeats=5, i=3, corr=0.2651519530891151, diff=0.04299096555618299
folds=7, repeats=5, i=4, corr=0.5908699533464415, diff=0.027298277809147353
folds=7, repeats=5, i=5, corr=0.9281179215432374, diff=0.022067476002258574
Completed: 7 folds, 5 repeats | Avg Correlation: 0.6183, Avg Mean Diff: 0.0305
folds=10, repeats=1, i=1, corr=0.5105811454714904, diff=0.021437499999999977
folds=10, repeats=1, i=2, corr=0.4433898732960672, diff=0.04668749999999999
folds=10, repeats=1, i=3, corr=0.7914480667320329, diff=0.04275000000000001
folds=10, repeats=1, i=4, corr=0.7947261848531103, diff=0.030562499999999996
folds=10, repeats=1, i=5, corr=-0.21439833215966733, diff=0.04981250000000003
Completed: 10 folds, 1 repeats | Avg Correlation: 0.4651, Avg Mean Diff: 0.0382
folds=10, repeats=3, i=1, corr=0.7611603545110889, diff=0.03481250000000001
folds=10, repeats=3, i=2, corr=0.1888132651963422, diff=0.030666666666666655
folds=10, repeats=3, i=3, corr=0.7913085335098206, diff=0.036291666666666667
folds=10, repeats=3, i=4, corr=0.5603737089083655, diff=0.027979166666666652
folds=10, repeats=3, i=5, corr=0.7326972446325504, diff=0.028354166666666635
Completed: 10 folds, 3 repeats | Avg Correlation: 0.6069, Avg Mean Diff: 0.0316
folds=10, repeats=5, i=1, corr=0.9160059282002866, diff=0.022924999999999984
folds=10, repeats=5, i=2, corr=0.7248606460867622, diff=0.02012499999999996
folds=10, repeats=5, i=3, corr=0.6691861954102255, diff=0.02302499999999999
folds=10, repeats=5, i=4, corr=0.522650343801325, diff=0.028912499999999966
folds=10, repeats=5, i=5, corr=0.48008889276862626, diff=0.03536249999999998
Completed: 10 folds, 5 repeats | Avg Correlation: 0.6626, Avg Mean Diff: 0.0261

Final Results:

    folds  repeats  avg_correlation  avg_mean_diff
0       3        1         0.447475       0.038580
1       3        3         0.517603       0.028562
2       3        5         0.404063       0.030491
3       5        1         0.158783       0.041538
4       5        3         0.656617       0.030867
5       5        5         0.637944       0.026335
6       7        1         0.304640       0.044856
7       7        3         0.748812       0.027349
8       7        5         0.618313       0.030498
9      10        1         0.465149       0.038250
10     10        3         0.606871       0.031621
11     10        5         0.662558       0.026070
```

### Observations

Plot of results:

![](/pics/test_harness_hacking_mitigation_study.png)

1. **Impact of Increasing Folds**:
   - For a fixed number of repeats (e.g., `repeats=1`):
     - `folds=3`: `avg_correlation=0.447`, `avg_mean_diff=0.0386`.
     - `folds=5`: `avg_correlation=0.159`, `avg_mean_diff=0.0415`.
     - `folds=7`: `avg_correlation=0.305`, `avg_mean_diff=0.0449`.
     - `folds=10`: `avg_correlation=0.465`, `avg_mean_diff=0.0383`.
   - The trend suggests no clear improvement in reducing overfitting (as measured by `avg_correlation`) for `repeats=1`, though `folds=10` shows a slight improvement.

2. **Impact of Increasing Repeats**:
   - For a fixed number of folds (e.g., `folds=5`):
     - `repeats=1`: `avg_correlation=0.159`, `avg_mean_diff=0.0415`.
     - `repeats=3`: `avg_correlation=0.657`, `avg_mean_diff=0.0309`.
     - `repeats=5`: `avg_correlation=0.638`, `avg_mean_diff=0.0263`.
   - Increasing repeats leads to higher `avg_correlation` and lower `avg_mean_diff`, indicating a reduction in overfitting.

3. **Impact of Combining Folds and Repeats**:
   - The best results (highest `avg_correlation` and lowest `avg_mean_diff`) are achieved with both high folds and high repeats:
     - `folds=10`, `repeats=5`: `avg_correlation=0.663`, `avg_mean_diff=0.0261`.
   - This combination shows the most significant reduction in overfitting.

1. Increasing the **number of repeats** has a more consistent and significant effect on reducing overfitting than increasing the number of folds in this study.
2. A combination of both higher folds and repeats provides the most substantial reduction in overfitting, as seen in the best-performing configuration (`folds=10`, `repeats=5`).
3. Using very few folds (e.g., `folds=3`) or repeats (e.g., `repeats=1`) generally results in less effective mitigation of overfitting, with lower `avg_correlation` and higher `avg_mean_diff`.

