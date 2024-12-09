# Data Science Dark Side

_What if we "bend" our machine learning and data science experiments toward achieving a preconceived goal?_

Generally, this is ethically bad and is called **scientific gaming** or **research gaming** or simply **machine learning hacking**. They're systematic attempts to exploit evaluation metrics or scientific processes to achieve desired outcomes without actually meeting the underlying scientific objectives.

## Examples of Machine Learning "_Gaming_" or "_Hacking_"

Below are examples of this type of gaming, and simple demonstrations of each:

* [p-Hacking](examples/p_hacking.md): _Repeat a statistical hypothesis test until a significant result is achieved._
* [Seed Hacking](examples/seed_hacking.md): _Repeat an experiment with different random number seeds to get the best result._
* [Test Set Memorization](examples/test_set_memorization.md): _Allow the model to memorize the test set and get a perfect score._
* [Test Set Overfitting](examples/test_set_overfitting.md): _Optimizing a model for its performance on a "hold out" test set._
* [Leaderboard Hacking](examples/leaderboard_hacking.md): _Issue predictions for a machine learning competition until a perfect score is achieved._
* [Threshold Hacking](examples/threshold_hacking.md): _Adjusting classification thresholds to hit specific metric targets._

I don't know what it is, but writing these examples feels forbidden, fun, and raise a tiny thrill :)

If you have ideas for more examples, email me: Jason.Brownlee05@gmail.com