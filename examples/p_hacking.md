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


## Further Reading

* [Data dredging](https://en.wikipedia.org/wiki/Data_dredging), Wikipedia.
* [The Extent and Consequences of P-Hacking in Science](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002106&), 2015.
* [Big little lies: a compendium and simulation of p-hacking strategies](https://royalsocietypublishing.org/doi/10.1098/rsos.220346), 2023.
