## Overview

This script performs physics-constrained fitting of time-resolved photoluminescence (TRPL) data using a three-state kinetic model consisting of 1CT, 3CT, and 3LE populations. It combines mechanistic rate-equation modeling with bounded nonlinear least-squares optimization to recover physically meaningful rate constants, signal scaling, and background terms from experimental decay data.

The fitting is designed to be more robust than a single local solve. It uses a two-stage search strategy: a broad global exploration over many starting points, followed by focused local refinement of the best candidate basins. In addition to matching the decay trace, the objective function also includes penalties tied to PLQY, sink-yield closure, mid-window fit quality, and several optional physics-based constraints that enforce reasonable kinetic relationships.

## What this script does

At its core, the script models emission as the sum of radiative contributions from the 1CT and 3LE states, with population transfer governed by a 3×3 rate matrix containing radiative, non-radiative, ISC, RISC, and CT/LE exchange terms. The model propagates the state populations in time, computes the physical emission signal, and maps that signal onto the measured trace through a fitted scale factor and baseline term.

The free parameters are fitted in transformed space, mainly as log10(rate) values plus natural-log parameters for scale and baseline. Bounds are explicitly defined for each active parameter, and the script automatically builds lower and upper bound vectors that stay consistent with the active parameter set. In the current configuration, krL is turned off by setting KRL_FIXED = 0.0, so the long-lived radiative channel is not fitted unless that setting is changed back to None.

## Key features

The script includes several features aimed at making kinetic fitting more stable and interpretable:

Physics-constrained objective that can enforce relationships such as kRISC <= kISC, kLECT <= kCTLE, lifetime windows for prompt S1 and T1 radiative decay, radiative hierarchy constraints, and a tail-lifetime consistency check.
Two-stage optimization with dense multi-start exploration (N_GLOBAL_STARTS = 2048) followed by refinement of the best retained candidates (N_STAGE1_KEEP = 64).
Parallel execution using joblib, with configurable worker count and backend.
Candidate ranking and deduplication so multiple near-duplicate solutions do not dominate the retained fit set.
Identifiability diagnostics based on Jacobian rank, singular values, condition number, and correlation structure.
Uncertainty and robustness analysis through local covariance-based uncertainty estimates, profile likelihood scans, and bootstrap refits.
Input data

By default, the script reads a two-column text file named EHBIPOAc_zeonex_300.txt. It expects numeric time-intensity data, sorts the trace by time, finds the intensity peak, shifts time so that the peak is treated as t = 0, and uses the post-peak decay region for fitting. It also estimates a tail baseline and constructs log-time bin weights so densely sampled regions do not dominate the fit unfairly.

## Fitting workflow

The overall workflow is:

Load and preprocess the TRPL data.
Estimate a tail lifetime target from the long-time region when that option is enabled.
Generate bounded starting points using Sobol sampling when available, otherwise uniform random sampling.
Run a robust first-stage fit using soft_l1 loss.
Keep the best distinct basins and refine them with a stricter second-stage linear-loss fit.
Select the best final solution using the reported objective metrics.
Compute fit diagnostics, parameter uncertainties, candidate rankings, and optional profile-likelihood and bootstrap analyses.
Save plots, tables, and a run summary to disk.

## Outputs

The script writes a complete set of analysis artifacts using the prefix rigorous_trpl_1CT_3CT_3LE. These outputs include:

rigorous_trpl_1CT_3CT_3LE_fit.png — full log-log comparison between data and best fit

rigorous_trpl_1CT_3CT_3LE_midregion.png — zoomed view of the selected mid-time fitting window

rigorous_trpl_1CT_3CT_3LE_curves.csv — fitted total signal, baseline, radiative components, physical emission rates, and state populations

rigorous_trpl_1CT_3CT_3LE_candidate_solutions.csv — top retained candidate solutions with fitted parameters and model metrics

rigorous_trpl_1CT_3CT_3LE_parameter_uncertainty.csv — local uncertainty estimates and uncertainty-status notes

rigorous_trpl_1CT_3CT_3LE_parameter_correlation.csv — parameter correlation matrix, or a message when correlation is not reliable

rigorous_trpl_1CT_3CT_3LE_profile_likelihood.csv — profile-likelihood scan results when enabled

rigorous_trpl_1CT_3CT_3LE_bootstrap.csv — bootstrap refit results when enabled

rigorous_trpl_1CT_3CT_3LE_summary.csv — run metadata, physics settings, fit-quality metrics, identifiability diagnostics, best-fit parameters, and yields.
Dependencies

The script uses the standard scientific Python stack: NumPy, Pandas, Matplotlib, and SciPy, plus joblib for parallel execution. If scipy.stats.qmc is available, Sobol quasi-random sampling is used for multi-start generation; otherwise the script falls back to pseudorandom sampling.

## Intended use

This script is intended for researchers who want a more physically grounded and diagnostics-heavy alternative to simple empirical decay fitting. It is especially useful when multiple kinetic basins are possible, when parameter identifiability is a concern, or when fit quality should be judged jointly against experimental trace shape and physically meaningful constraints such as PLQY and channel-yield consistency.

## Notes

Several settings in this version are explicitly tuned for a constrained search landscape. The parameter bounds are tightened around supported regions, bootstrap and profile-likelihood analysis are enabled, and the script keeps a relatively large number of global and local candidates before selecting the final fit. This makes the workflow better suited for convergence checking and parameter-stability analysis than for ultra-fast screening.

## Requirements

Install the required Python packages:

pip install numpy pandas matplotlib scipy joblib

Optional:

scipy.stats.qmc is used for Sobol quasi-random start-point generation when available.

## How to run

Place your TRPL data file in the same folder as the script, or update the DATA_FILE path in the script.

Then run:

python physics_constraints_v6_0_300_bounds_only_convergence_tuned.py


## Customization

You may want to edit the following settings depending on your experiment:

### Data and outputs
DATA_FILE
OUTPUT_PREFIX
### Search settings
N_GLOBAL_STARTS
N_STAGE1_KEEP
STAGE1_MAX_NFEV
STAGE2_MAX_NFEV
### Parallel settings
N_WORKERS
PARALLEL_BACKEND
### Targets and penalties
PLQY_TARGET
PLQY_SIGMA
SINK_TARGET
SINK_SIGMA
R2_MID_TARGET
R2_MID_SIGMA
### Physics constraints
ENABLE_PHYSICS_CONSTRAINTS
ENABLE_PROMPT_S1_LIFETIME_CONSTRAINT
ENABLE_RADIATIVE_HIERARCHY_CONSTRAINT
ENABLE_T1_RADIATIVE_LIFETIME_CONSTRAINT
ENABLE_TAIL_LIFETIME_CONSTRAINT
### Expensive extras
ENABLE_BOOTSTRAP
N_BOOTSTRAP
ENABLE_PROFILE_LIKELIHOOD
PROFILE_POINTS
