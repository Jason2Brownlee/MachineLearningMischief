# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

Seed hacking or random seed shopping is a problematic practice where practitioners manipulate random number seeds to artificially improve model performance metrics.

The technique involves repeatedly running the same experiment (e.g. model, data split, etc.) with different random seeds until finding one that produces better-looking results. This is typically done during model validation or testing phases.

While random seeds are important for reproducibility, exploiting them to cherry-pick favorable outcomes introduces severe bias. This practice masks the model's true performance and can lead to poor generalization in production.

The key issue is that seed hacking violates the principle of independent validation. By selecting seeds based on outcomes, you're effectively leaking information from your test set into your model selection process.

This practice is particularly dangerous for new data scientists because it can be tempting to use when under pressure to show improved metrics. However, it fundamentally undermines the scientific validity of your work.

A more ethical approach is to use fixed random seeds for reproducibility, but to select them before seeing any results. This maintains experimental integrity while still allowing others to replicate your work.