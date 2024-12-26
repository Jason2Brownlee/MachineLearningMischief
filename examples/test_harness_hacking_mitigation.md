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
   - The study evaluates combinations of k-fold CV (3, 5, 7, 10 folds) and repeated CV (1, 3, 5, and 10 repeats).
   - Each configuration undergoes 10 independent trials.

3. **Hill-Climbing Hyperparameter Tuning**:
   - For each CV configuration, a hill-climbing algorithm runs 100 iterations to optimize the `n_estimators` and `max_depth` hyperparameters of a `RandomForestClassifier`.
   - Each hyperparameter configuration is evaluated using the specified CV method, and the best configuration is selected based on the mean CV score.

4. **Metrics Computation**:
   - For each run, the correlation between the CV scores and hold-out test scores, as well as the mean absolute difference between them, is calculated to quantify overfitting.
   - Overfitting is characterized by a decrease in correlation and an increase in the mean absolute difference.

5. **Recording and Analysis**:
   - The study aggregates results for each combination of folds and repeats, computing the average correlation and mean absolute difference over 10 trials.

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
    n_samples=1000, n_features=30, n_informative=5, n_redundant=25, random_state=42
)

# Create a train/test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize result storage for experiments
results = []

# Define the study parameters
fold_range = [3, 5, 7, 10]  # 3 to 10 folds
repeat_range = [1, 3, 5, 10]  # 1 to 10 repetitions
n_trials = 10  # Number of trials for each configuration

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
folds=3, repeats=1, i=1, corr=0.8975081501258906, diff=0.013994529495226418
folds=3, repeats=1, i=2, corr=0.8177792410740738, diff=0.011125753793617622
folds=3, repeats=1, i=3, corr=0.9428830954671136, diff=0.005017053122670292
folds=3, repeats=1, i=4, corr=0.9049809252717387, diff=0.007363626481975841
folds=3, repeats=1, i=5, corr=0.9758504080203283, diff=0.023852118413774832
folds=3, repeats=1, i=6, corr=0.857747359046279, diff=0.01297499843686123
folds=3, repeats=1, i=7, corr=0.9543552148233073, diff=0.010973930692831212
folds=3, repeats=1, i=8, corr=0.9583072215690465, diff=0.012318426279970197
folds=3, repeats=1, i=9, corr=0.9365443080461188, diff=0.016589097467715166
folds=3, repeats=1, i=10, corr=0.972215872219083, diff=0.01591398937065626
Completed: 3 folds, 1 repeats | Avg Correlation: 0.9218, Avg Mean Diff: 0.0130
folds=3, repeats=3, i=1, corr=0.9728624412333399, diff=0.006810931550553492
folds=3, repeats=3, i=2, corr=0.8854785215597786, diff=0.01782942420380126
folds=3, repeats=3, i=3, corr=0.9575675906128579, diff=0.014725479162157584
folds=3, repeats=3, i=4, corr=0.9778969635559742, diff=0.019854740799525407
folds=3, repeats=3, i=5, corr=0.9739011078616541, diff=0.021317557335128957
folds=3, repeats=3, i=6, corr=0.9364274591948702, diff=0.015523946003575146
folds=3, repeats=3, i=7, corr=0.9350829635734347, diff=0.012577906676606609
folds=3, repeats=3, i=8, corr=0.9851501120289593, diff=0.014629806892239604
folds=3, repeats=3, i=9, corr=0.9849767257721495, diff=0.010881155279801769
folds=3, repeats=3, i=10, corr=0.9849943833757703, diff=0.018577150438079643
Completed: 3 folds, 3 repeats | Avg Correlation: 0.9594, Avg Mean Diff: 0.0153
folds=3, repeats=5, i=1, corr=0.9906435183380549, diff=0.009374101435682656
folds=3, repeats=5, i=2, corr=0.9305450478665325, diff=0.019447710843373447
folds=3, repeats=5, i=3, corr=0.9698320836611718, diff=0.013032678738907611
folds=3, repeats=5, i=4, corr=0.9826452213379984, diff=0.018392676334078654
folds=3, repeats=5, i=5, corr=0.9720287635544785, diff=0.016324860159199598
folds=3, repeats=5, i=6, corr=0.9454211768270858, diff=0.01394443787124552
folds=3, repeats=5, i=7, corr=0.9204636647370464, diff=0.010203038260827574
folds=3, repeats=5, i=8, corr=0.9955924474911255, diff=0.012422861746386698
folds=3, repeats=5, i=9, corr=0.9792351257750852, diff=0.012180605535915996
folds=3, repeats=5, i=10, corr=0.9905394275428722, diff=0.01734686963422549
Completed: 3 folds, 5 repeats | Avg Correlation: 0.9677, Avg Mean Diff: 0.0143
folds=3, repeats=10, i=1, corr=0.9917633072174894, diff=0.013716341774282666
folds=3, repeats=10, i=2, corr=0.9893929850040933, diff=0.015559962003703322
folds=3, repeats=10, i=3, corr=0.9569010802478903, diff=0.015219092898540065
folds=3, repeats=10, i=4, corr=0.9879348648553065, diff=0.013215456797248795
folds=3, repeats=10, i=5, corr=0.9735665396739734, diff=0.017831983262390768
folds=3, repeats=10, i=6, corr=0.971173862752053, diff=0.016002982949763
folds=3, repeats=10, i=7, corr=0.9635668621274783, diff=0.010760243368684199
folds=3, repeats=10, i=8, corr=0.9648722608008841, diff=0.012233203953538581
folds=3, repeats=10, i=9, corr=0.9587285072176933, diff=0.011003018060264878
folds=3, repeats=10, i=10, corr=0.9207755703824012, diff=0.009858052569559616
Completed: 3 folds, 10 repeats | Avg Correlation: 0.9679, Avg Mean Diff: 0.0135
folds=5, repeats=1, i=1, corr=0.8861934056107431, diff=0.01589999999999999
folds=5, repeats=1, i=2, corr=0.7662794730952668, diff=0.011219999999999977
folds=5, repeats=1, i=3, corr=0.8826187133353169, diff=0.011020000000000012
folds=5, repeats=1, i=4, corr=0.9440121829589864, diff=0.010340000000000021
folds=5, repeats=1, i=5, corr=0.8766129357443, diff=0.004159999999999984
folds=5, repeats=1, i=6, corr=0.9024164969658375, diff=0.017580000000000016
folds=5, repeats=1, i=7, corr=0.9005217405018474, diff=0.008240000000000035
folds=5, repeats=1, i=8, corr=0.9828141793722028, diff=0.008400000000000029
folds=5, repeats=1, i=9, corr=0.961712143471749, diff=0.021140000000000003
folds=5, repeats=1, i=10, corr=0.9813556722050953, diff=0.006739999999999999
Completed: 5 folds, 1 repeats | Avg Correlation: 0.9085, Avg Mean Diff: 0.0115
folds=5, repeats=3, i=1, corr=0.9122982760466545, diff=0.009366666666666688
folds=5, repeats=3, i=2, corr=0.9890461182715037, diff=0.009599999999999985
folds=5, repeats=3, i=3, corr=0.9183930613020971, diff=0.0060066666666666515
folds=5, repeats=3, i=4, corr=0.9294897940388198, diff=0.011720000000000029
folds=5, repeats=3, i=5, corr=0.981982679087837, diff=0.012706666666666746
folds=5, repeats=3, i=6, corr=0.9711766765002295, diff=0.01302666666666672
folds=5, repeats=3, i=7, corr=0.9651742983090498, diff=0.007959999999999997
folds=5, repeats=3, i=8, corr=0.9616274843010032, diff=0.010246666666666682
folds=5, repeats=3, i=9, corr=0.991484802507542, diff=0.012240000000000055
folds=5, repeats=3, i=10, corr=0.949354181531814, diff=0.007480000000000023
Completed: 5 folds, 3 repeats | Avg Correlation: 0.9570, Avg Mean Diff: 0.0100
folds=5, repeats=5, i=1, corr=0.9124335330130132, diff=0.010407999999999962
folds=5, repeats=5, i=2, corr=0.9943967107022027, diff=0.00918799999999998
folds=5, repeats=5, i=3, corr=0.9481474796710471, diff=0.005691999999999955
folds=5, repeats=5, i=4, corr=0.9638374112067487, diff=0.011388
folds=5, repeats=5, i=5, corr=0.952119476071311, diff=0.011971999999999943
folds=5, repeats=5, i=6, corr=0.9887007143739523, diff=0.01114399999999998
folds=5, repeats=5, i=7, corr=0.9700508321437197, diff=0.005847999999999956
folds=5, repeats=5, i=8, corr=0.9786839027183967, diff=0.00939999999999993
folds=5, repeats=5, i=9, corr=0.9923019529456245, diff=0.010243999999999967
folds=5, repeats=5, i=10, corr=0.9806823785079624, diff=0.008635999999999994
Completed: 5 folds, 5 repeats | Avg Correlation: 0.9681, Avg Mean Diff: 0.0094
folds=5, repeats=10, i=1, corr=0.9899246985924093, diff=0.010209999999999997
folds=5, repeats=10, i=2, corr=0.9863212431811526, diff=0.009343999999999979
folds=5, repeats=10, i=3, corr=0.980497156341154, diff=0.010361999999999982
folds=5, repeats=10, i=4, corr=0.9809354894495217, diff=0.01051599999999992
folds=5, repeats=10, i=5, corr=0.9716781005974886, diff=0.009705999999999982
folds=5, repeats=10, i=6, corr=0.98761167410509, diff=0.007167999999999995
folds=5, repeats=10, i=7, corr=0.9939648038833919, diff=0.008125999999999956
folds=5, repeats=10, i=8, corr=0.9929684118650098, diff=0.00977600000000004
folds=5, repeats=10, i=9, corr=0.9916934696335423, diff=0.011239999999999965
folds=5, repeats=10, i=10, corr=0.9900364630237838, diff=0.0058139999999999555
Completed: 5 folds, 10 repeats | Avg Correlation: 0.9866, Avg Mean Diff: 0.0092
folds=7, repeats=1, i=1, corr=0.9613182318742974, diff=0.009815023474178432
folds=7, repeats=1, i=2, corr=0.6909957358983967, diff=0.006366718086295516
folds=7, repeats=1, i=3, corr=0.9895758217003858, diff=0.005750017885088313
folds=7, repeats=1, i=4, corr=0.9877611150535017, diff=0.007385823831880195
folds=7, repeats=1, i=5, corr=0.9749578118659352, diff=0.011593217080259361
folds=7, repeats=1, i=6, corr=0.9711935713591127, diff=0.00582282137268052
folds=7, repeats=1, i=7, corr=0.9760796613233829, diff=0.006076532528504364
folds=7, repeats=1, i=8, corr=0.9371927392344911, diff=0.01633843952604521
folds=7, repeats=1, i=9, corr=0.9746045786830242, diff=0.012915823831880134
folds=7, repeats=1, i=10, corr=0.9711195358371305, diff=0.011247455846188222
Completed: 7 folds, 1 repeats | Avg Correlation: 0.9435, Avg Mean Diff: 0.0093
folds=7, repeats=3, i=1, corr=0.9128474983896583, diff=0.010857055667337348
folds=7, repeats=3, i=2, corr=0.9924414793688743, diff=0.009876530292868339
folds=7, repeats=3, i=3, corr=0.9753841890858064, diff=0.0066518794246963175
folds=7, repeats=3, i=4, corr=0.9902161753182072, diff=0.01067602652954771
folds=7, repeats=3, i=5, corr=0.9875467068521149, diff=0.014647294880393522
folds=7, repeats=3, i=6, corr=0.9824724251891493, diff=0.0053529637081750035
folds=7, repeats=3, i=7, corr=0.9590765688110139, diff=0.006180702734928126
folds=7, repeats=3, i=8, corr=0.9858973820022585, diff=0.009431569416499024
folds=7, repeats=3, i=9, corr=0.9713070448511517, diff=0.009675288024442926
folds=7, repeats=3, i=10, corr=0.9635099762903706, diff=0.006624511513525587
Completed: 7 folds, 3 repeats | Avg Correlation: 0.9721, Avg Mean Diff: 0.0090
folds=7, repeats=5, i=1, corr=0.9199378342268244, diff=0.008922989045383398
folds=7, repeats=5, i=2, corr=0.9305605240628312, diff=0.009462680527610085
folds=7, repeats=5, i=3, corr=0.9868414126704743, diff=0.008307379834562902
folds=7, repeats=5, i=4, corr=0.9899484002488507, diff=0.008614039794321458
folds=7, repeats=5, i=5, corr=0.976576733728466, diff=0.009443724569640015
folds=7, repeats=5, i=6, corr=0.9693547315705672, diff=0.005900277218868722
folds=7, repeats=5, i=7, corr=0.9909369574070983, diff=0.005307771070869582
folds=7, repeats=5, i=8, corr=0.9892888145831531, diff=0.011010156494522676
folds=7, repeats=5, i=9, corr=0.9627449539744223, diff=0.010943483120947896
folds=7, repeats=5, i=10, corr=0.9658412890907523, diff=0.006972148446232913
Completed: 7 folds, 5 repeats | Avg Correlation: 0.9682, Avg Mean Diff: 0.0085
folds=7, repeats=10, i=1, corr=0.9930980639376968, diff=0.010301558238318903
folds=7, repeats=10, i=2, corr=0.9920910968628622, diff=0.010058085177733074
folds=7, repeats=10, i=3, corr=0.9874525701739706, diff=0.007801203890006742
folds=7, repeats=10, i=4, corr=0.9854169402266026, diff=0.006881775095014487
folds=7, repeats=10, i=5, corr=0.9771779515816831, diff=0.006837573217080281
folds=7, repeats=10, i=6, corr=0.9900552860025492, diff=0.006621960652805747
folds=7, repeats=10, i=7, corr=0.9578589198570598, diff=0.007519338251732596
folds=7, repeats=10, i=8, corr=0.9031445620706875, diff=0.005411533646322324
folds=7, repeats=10, i=9, corr=0.9938400095864596, diff=0.010214242119382887
folds=7, repeats=10, i=10, corr=0.9788426400080306, diff=0.005239181757209895
Completed: 7 folds, 10 repeats | Avg Correlation: 0.9759, Avg Mean Diff: 0.0077
folds=10, repeats=1, i=1, corr=0.9748033758314912, diff=0.015320000000000049
folds=10, repeats=1, i=2, corr=0.9187986740124284, diff=0.010219999999999991
folds=10, repeats=1, i=3, corr=0.9759534155628019, diff=0.005019999999999986
folds=10, repeats=1, i=4, corr=0.9404497799396768, diff=0.0062200000000000085
folds=10, repeats=1, i=5, corr=0.9774883703984174, diff=0.011080000000000003
folds=10, repeats=1, i=6, corr=0.9149008540654683, diff=0.016480000000000064
folds=10, repeats=1, i=7, corr=0.8801191409994777, diff=0.006560000000000007
folds=10, repeats=1, i=8, corr=0.9719343758671737, diff=0.012880000000000025
folds=10, repeats=1, i=9, corr=0.9856023103580355, diff=0.0071200000000000195
folds=10, repeats=1, i=10, corr=0.9556385726542705, diff=0.009180000000000035
Completed: 10 folds, 1 repeats | Avg Correlation: 0.9496, Avg Mean Diff: 0.0100
folds=10, repeats=3, i=1, corr=0.9818082957751972, diff=0.010593333333333219
folds=10, repeats=3, i=2, corr=0.9314961162728049, diff=0.005919999999999988
folds=10, repeats=3, i=3, corr=0.9127620228239297, diff=0.004453333333333281
folds=10, repeats=3, i=4, corr=0.9810389657863505, diff=0.006766666666666591
folds=10, repeats=3, i=5, corr=0.9918364430409649, diff=0.007466666666666555
folds=10, repeats=3, i=6, corr=0.9395927744026121, diff=0.010773333333333244
folds=10, repeats=3, i=7, corr=0.9885928388539656, diff=0.007373333333333213
folds=10, repeats=3, i=8, corr=0.9036806490023385, diff=0.0070266666666665475
folds=10, repeats=3, i=9, corr=0.9612958178459122, diff=0.008580000000000004
folds=10, repeats=3, i=10, corr=0.9852138676111265, diff=0.006326666666666592
Completed: 10 folds, 3 repeats | Avg Correlation: 0.9577, Avg Mean Diff: 0.0075
folds=10, repeats=5, i=1, corr=0.9869203108957099, diff=0.008232000000000034
folds=10, repeats=5, i=2, corr=0.989158809458989, diff=0.006472000000000011
folds=10, repeats=5, i=3, corr=0.9468112866930535, diff=0.005743999999999982
folds=10, repeats=5, i=4, corr=0.9555637452766133, diff=0.009019999999999985
folds=10, repeats=5, i=5, corr=0.8867829384047247, diff=0.008167999999999996
folds=10, repeats=5, i=6, corr=0.9388857673846408, diff=0.0054799999999999875
folds=10, repeats=5, i=7, corr=0.9689970283669487, diff=0.005588000000000004
folds=10, repeats=5, i=8, corr=0.980303607125597, diff=0.008103999999999974
folds=10, repeats=5, i=9, corr=0.9873591803251196, diff=0.006583999999999992
folds=10, repeats=5, i=10, corr=0.9543215764841885, diff=0.005707999999999978
Completed: 10 folds, 5 repeats | Avg Correlation: 0.9595, Avg Mean Diff: 0.0069
folds=10, repeats=10, i=1, corr=0.9895772309728519, diff=0.008010000000000012
folds=10, repeats=10, i=2, corr=0.9371148265713927, diff=0.0056619999999999926
folds=10, repeats=10, i=3, corr=0.9946765193420288, diff=0.007330000000000046
folds=10, repeats=10, i=4, corr=0.9700972981521802, diff=0.006941999999999987
folds=10, repeats=10, i=5, corr=0.9869957451350777, diff=0.009159999999999979
folds=10, repeats=10, i=6, corr=0.943939573626107, diff=0.004751999999999973
folds=10, repeats=10, i=7, corr=0.8887097918134683, diff=0.00497599999999995
folds=10, repeats=10, i=8, corr=0.9915157230477832, diff=0.010015999999999867
folds=10, repeats=10, i=9, corr=0.9869359272490404, diff=0.007609999999999974
folds=10, repeats=10, i=10, corr=0.9840942879680609, diff=0.005351999999999967
Completed: 10 folds, 10 repeats | Avg Correlation: 0.9674, Avg Mean Diff: 0.0070

Final Results:

    folds  repeats  avg_correlation  avg_mean_diff
0       3        1         0.921817       0.013012
1       3        3         0.959434       0.015273
2       3        5         0.967695       0.014267
3       3       10         0.967868       0.013540
4       5        1         0.908454       0.011474
5       5        3         0.957003       0.010035
6       5        5         0.968135       0.009392
7       5       10         0.986563       0.009226
8       7        1         0.943480       0.009331
9       7        3         0.972070       0.008997
10      7        5         0.968203       0.008488
11      7       10         0.975898       0.007689
12     10        1         0.949569       0.010008
13     10        3         0.957732       0.007528
14     10        5         0.959510       0.006910
15     10       10         0.967366       0.006981
```

### Observations

Plots of results:

![](/pics/test_harness_hacking_mitigation_study1.png)

![](/pics/test_harness_hacking_mitigation_study2.png)

Here’s an analysis of the experiment results based on the provided data:

#### 1. Trends in Average Correlation
- **General Trend**:
  - As the number of repeats increases, the average correlation tends to improve for all fold values.
  - This indicates that more repeats lead to more stable and consistent results, likely due to better statistical reliability.

- **Impact of Folds**:
  - For **3 folds**, the correlation starts high (0.92) and stabilizes around 0.96-0.97 with increasing repeats.
  - For **5 folds**, correlation is slightly lower initially (0.91) but improves significantly with more repeats, reaching a peak at 10 repeats (0.986).
  - For **7 folds**, the correlation starts higher (0.94), improves consistently, but peaks slightly lower than 5 folds (around 0.975-0.968).
  - For **10 folds**, the correlation is generally high but improves more modestly compared to other fold values, peaking around 0.96-0.97.

- **Key Observations**:
  - More folds combined with higher repeats generally provide better correlations.
  - 5 folds with 10 repeats show the highest correlation (0.986), suggesting it is an optimal balance.

#### 2. Trends in Average Mean Difference
- **General Trend**:
  - As the number of repeats increases, the average mean difference consistently decreases across all fold values.
  - This suggests that repeated experiments help to minimize variability and bring the mean difference closer to zero.

- **Impact of Folds**:
  - For **3 folds**, the mean difference starts around 0.013 and gradually decreases with more repeats.
  - For **5 folds**, it starts lower (0.011) and decreases significantly to around 0.009 at 10 repeats.
  - For **7 folds**, the mean difference starts lower still (0.009) and shows the most dramatic improvement, dropping to 0.007 at 10 repeats.
  - For **10 folds**, the mean difference begins at 0.010 and also improves to 0.007 but shows diminishing returns with higher repeats.

- **Key Observations**:
  - Higher fold values, such as 7 or 10 folds, generally produce lower mean differences, particularly when paired with a higher number of repeats.

#### 3. Balancing Correlation and Mean Difference
- **Trade-Offs**:
  - While 5 folds with 10 repeats yields the highest correlation (0.986), it does not produce the smallest mean difference.
  - 7 folds with 10 repeats achieves a slightly lower correlation (0.975) but has one of the smallest mean differences (0.007).

- **Optimal Configuration**:
  - If correlation is prioritized, 5 folds and 10 repeats is optimal.
  - If minimizing mean difference is more important, 7 or 10 folds with 10 repeats might be preferable.

#### 4. Recommendations for Future Experiments
- **Choose Higher Repeats**:
  - Increasing the number of repeats provides diminishing returns beyond 10 but is generally effective at stabilizing results.

- **Optimize Fold Selection**:
  - Depending on the metric of interest, 5 folds for correlation or 7 folds for mean difference are promising choices.

- **Investigate Trade-Offs Further**:
  - Explore whether a compromise between correlation and mean difference exists, possibly at intermediate fold values (e.g., 6 or 8).



