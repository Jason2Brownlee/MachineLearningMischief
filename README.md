<img src="pics/cover_cropped.png" alt="Machine Learning Mischief" width="600">

# Machine Learning Mischief

_It is possible to "bend" machine learning experiments towards achieving a preconceived goal?_

This involves systematically exploiting evaluation metrics and/or scientific tests to achieve desired outcomes without actually meeting the underlying scientific objectives.

These behaviors are _unethical_ and might be called [_cherry picking_](https://en.wikipedia.org/wiki/Cherry_picking), [_data dredging_](https://en.wikipedia.org/wiki/Data_dredging), or _gaming results_.

Reviewing examples of this type of "gaming" can remind beginners and stakeholders (really all of us!) why certain methods are best practices and how to avoid being deceived by results that are too good to be true.

## Examples

Below are examples of this type of gaming, and simple demonstrations of each:

* [Seed Hacking](examples/seed_hacking.md): _Repeat an experiment with different random number seeds to get the best result._
	* [Cross-Validation Hacking](examples/cross_validation_hacking.md): _Vary the seed for creating cross-validation folds in order to get the best result._
	* [Train/Test Split Hacking](examples/train_test_split_hacking.md): _Vary the seed for creating train/test splits in order to get the best result._
	* [Model Selection Hacking](examples/model_selection_hacking.md): _Vary the seed for the model training algorithm in order to get the best result._
	* [Performance Hacking](examples/performance_hacking.md): _Vary the seed for a bootstrap evaluation of a final chosen model on the test set to present the best performance._
* [p-Hacking](examples/p_hacking.md): _Repeat a statistical hypothesis test until a significant result is achieved._
* [Test Set Memorization](examples/test_set_memorization.md): _Allow the model to memorize the test set and get a perfect score._
* [Test Set Overfitting](examples/test_set_overfitting.md): _Optimizing a model for its performance on a "hold out" test set._
* [Test Set Pruning](examples/test_set_pruning.md): _Remove hard-to-predict examples from the test set to improve results._
* [Train/Test Split Ratio Gaming](examples/train_test_ratio_gaming.md): _Vary train/test split ratios until a desired result is achieved._
* [Leaderboard Hacking](examples/leaderboard_hacking.md): _Issue predictions for a machine learning competition until a perfect score is achieved._
* [Threshold Hacking](examples/threshold_hacking.md): _Adjusting classification thresholds to hit specific metric targets._

I don't know what it is, but writing these examples feels forbidden, fun, and raise a tiny thrill :)

Related ideas: [Researcher degrees of freedom](https://en.wikipedia.org/wiki/Researcher_degrees_of_freedom) and [Forking paths problem](https://en.wikipedia.org/wiki/Forking_paths_problem).

If you like this project, you may be interested in [Data Science Diagnostics](https://DataScienceDiagnostics.com).

If you have ideas for more examples, email me: Jason.Brownlee05@gmail.com