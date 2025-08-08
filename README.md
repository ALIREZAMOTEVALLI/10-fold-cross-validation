How It Works
Synthetic Data Generation

For each horizon (A, B, C), a small dataset is generated with:

clay = observed clay percentage

clay_1 = predicted clay percentage (with a horizon-specific bias and noise)

Horizons differ in mean clay content, variance, and systematic bias to mimic real-world differences:

A horizon: smallest variance, slight underprediction.

B horizon: moderate variance, slight overprediction.

C horizon: highest variance, stronger underprediction.

10-Fold Cross-Validation

The dataset for each horizon–region pair is split into 10 folds using KFold (random shuffle, fixed seed).

For each fold, RMSE and MBE are calculated without retraining a model (predictions are already given).

This produces a distribution of RMSE/MBE values per dataset.

Visualization

Two boxplots are created:

Left: RMSE distributions for all datasets.

Right: MBE distributions for all datasets.

Boxes are color-coded by horizon (A=red, B=green, C=orange).

X-axis labels show the geo-region for each dataset.
