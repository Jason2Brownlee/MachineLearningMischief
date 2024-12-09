# Machine Learning Mischief

_What if we "bend" our machine learning experiments toward achieving a preconceived goal?_

Generally, this is _unethical_ and might be called [_cherry picking_](https://en.wikipedia.org/wiki/Cherry_picking) or [_data dredging_](https://en.wikipedia.org/wiki/Data_dredging) or _gaming results_. Often, they are systematic attempts to exploit evaluation metrics or scientific processes to achieve desired outcomes without actually meeting the underlying scientific objectives.

In our field, we might call this **machine learning gaming** or "**machine learning hacking** (or how about: "_machine learning mischief_" or "_the data science dark side_"?).

## Examples of Machine Learning "_Gaming_"

Below are examples of this type of gaming, and simple demonstrations of each:

* [Seed Hacking](examples/seed_hacking.md): _Repeat an experiment with different random number seeds to get the best result._
	* [Cross-Validation Hacking](examples/threshold_hacking.md): _Vary the cross-validation folds to get the best result._
	* [Train/Test Split Hacking](examples/train_test_split_hacking.md): _Vary the train/test split to get the best result._
	* [Model Selection Hacking](examples/model_selection_hacking.md): _Vary the model random seed to get the best result._
* [p-Hacking](examples/p_hacking.md): _Repeat a statistical hypothesis test until a significant result is achieved._
* [Test Set Memorization](examples/test_set_memorization.md): _Allow the model to memorize the test set and get a perfect score._
* [Test Set Overfitting](examples/test_set_overfitting.md): _Optimizing a model for its performance on a "hold out" test set._
* [Leaderboard Hacking](examples/leaderboard_hacking.md): _Issue predictions for a machine learning competition until a perfect score is achieved._
* [Threshold Hacking](examples/threshold_hacking.md): _Adjusting classification thresholds to hit specific metric targets._

I don't know what it is, but writing these examples feels forbidden, fun, and raise a tiny thrill :)

If you have ideas for more examples, email me: Jason.Brownlee05@gmail.com