# Data Science Dark Side

## Examples of Machine Learning "_Gaming_" or "_Hacking_"

These examples fall into a category often called **scientific gaming** or **research gaming** or simply **machine learning hacking** - they're systematic attempts to exploit evaluation metrics or scientific processes to achieve desired outcomes without actually meeting the underlying scientific objectives.

Below are some examples of some of these deliberate attempts.

I don't know what it is, but writing examples of these attacks feels forbidden and raise a tiny thrill :)

* [Seed Hacking](examples/seed_hacking): Repeat an experiment with different seeds for the pseudorandom number generator until the desired result is achieved.
* [Test Set Memorization](examples/test_set_memorization): Worse case of test set leakage where the model memorizes the test set and achieves a perfect score.
* [Test Set Overfitting](examples/test_set_overfitting): Optimizing a model for its performance on a "hold out" test set.
* [Leaderboard Hacking](examples/leaderboard_hacking): Issue predictions for a machine learning competition until a perfect (or near perfect) score is achieved.
