# LDA manuscript-version review

`LDA-plots.ipynb` now follows the local desktop notebook rather than the older
public copy. The input `text-file/3firstmonths_newdistance_updatedsept28.csv` is
copied unchanged from the desktop. The original catalog remains in the repository.

## Changes

- Compute the line intercept for each resample as `-intercept / magnitude_coefficient`;
  report the mean and sample standard deviation of those intercepts.
- Use stratified 80-percent training samples and display all eligible observations.
- Drop missing required values and handle a single-class panel without fitting.
- Use repository-relative input/output paths and count only `y` as visible.
- Fix the random seed to 0 so repeated runs reproduce the same figure.
- Save PNG and PDF previews under `figures/review/`; notebook outputs are cleared.

The first four items originate in the local notebook; paths, explicit positive
label handling, fixed seed and PDF export were added for this review branch.

## Interpretation to approve

The upper panels exclude horizontal distances of 60-70 km, not a geographic
Mount St. Helens selection. The caption must describe that distinction unless
the selection is deliberately revised. Depth bins are [0, 5) and [5, 15) km.
The plus/minus values are standard deviations across random subsets, not 95-percent
confidence intervals. Fixing the seed can change rounded values relative to a
previous unseeded screenshot; compare the preview before updating manuscript numbers.

Run from the repository root or `notebooks/` with NumPy, pandas, Matplotlib and
scikit-learn installed. This update does not reclassify events or validate the
underlying visual-review decisions and data availability.

## Validation

Executed successfully with Python 3.13.7, NumPy 2.3.3, pandas 2.3.3,
Matplotlib 3.10.6 and scikit-learn 1.9.1. Review PNG/PDF outputs accompany
this branch. The input CSV is byte-identical to the desktop source (SHA-256
`99132ed5f1dc48a6a5c4dee3c65b2f5ee2dccb9bb899845de22af6e20ecc7a1e`).
The TM driver and all 39 station-notebook code cells passed syntax parsing;
the waveform-dependent TM workflows were not executed.
