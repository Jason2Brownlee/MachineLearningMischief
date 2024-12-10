# Seed Hacking

> Repeat an experiment with different random number seeds to get the best result.

This practice is often referred to as **seed hacking** or **random seed shopping** - it's essentially a form of p-hacking specific to machine learning experiments. It's considered a questionable research practice since it can lead to unreliable or misleading results.

The basic problem is that by trying different random seeds until you get the outcome you want, you're essentially performing multiple hypothesis tests without proper correction, which can inflate your apparent results and make random variations look like real effects.

This is similar to but distinct from the broader concept of _researcher degrees of freedom_ or _garden of forking paths_ in statistics, which describes how researchers can make various seemingly reasonable analytical choices that affect their results.
