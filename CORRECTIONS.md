# Corrections to the fitting protocol and biomarker code

This branch (`corrected-fitting-protocol`) brings the deposited code into agreement with
what was actually run to produce the published numbers, fixes the defects that agreement
exposed, and adds the endpoint analysis the AT Score is best supported by.

Reproduce everything with:

```bash
python patients/first_cycle_fitting.py      # corrected first-cycle fits, both cohorts
python analysis/holiday_endpoint.py         # primary (holiday) + Zhang TTP endpoints
python analysis/make_etableS8_inputs.py     # derived inputs behind eTable S8
```

---

## 1. The fitted carrying capacity is no longer discarded

The published Bruchovsky tables ran `trunc_df['K'] = 2.5` **after** fitting, overwriting the
fitted value. The fitted K reached its upper bound 5.0 for 19 of 72 patients and an interior
value for 2 — **21 of 72 published K values were not what the optimiser returned**.

`first_cycle_fitting.py` retains the fitted K and records the old constant in a separate
`K_published_override` column. Effect on the published Figure 3a analysis: N 40 → 41,
HR 0.49 → 0.51 (95% CI 0.32–0.80), P .002 → .004.

## 2. Notebook-local biomarker functions moved into `utils/`

`predict_clinical_ttp`, `predict_at_threshold_ttp_simp`, `calc_threshold_benefit` and
`fraction_time_below_threshold` existed only inside `SI_Identifiability_Analysis.ipynb`, so
readers working from `utils/myUtils.py` could not reproduce eTable S8. They are now in
`utils/myUtils.py`, together with `fraction_time_on_treatment` and `at_score_pole`.

## 3. Silent patient exclusions replaced by stated rules

The Zhang analysis dropped patients **1004 and 1009** through a bare `except` (`int(NaN)`
raises for them). Both have clinical TTP = 0 under the 1.2×-first-PSA rule because they
enrolled near their PSA nadir. The exclusion is defensible but was undocumented, and the
published association depended on it. It is now an explicit, reported filter
(`exclude TTP = 0`) with the association shown at every filtering stage.

## 4. Structurally unidentifiable fits are flagged

Two parameters (turnover, K) are fitted. **9 of 17 Zhang patients have fewer than 3
first-cycle PSA measurements**; 5 have exactly one, for which R² is undefined and the
optimiser cannot move from its start value, and 4 have two, for which R² = 1 by
construction. The published filter `RSquared.isna() | RSquared > 0.5` deliberately retained
all of these. They are now flagged (`flag_unidentifiable`) and excluded from the holiday
endpoint; for the Zhang TTP endpoint, excluding them leaves N = 4, which is reported as a
limitation rather than silently ignored.

## 5. Cohort-specific priors and normalisation documented

| | Bruchovsky | Zhang |
|---|---|---|
| PSA normalisation | by the **first** measurement (so `n0 = 1.0` for every patient) | supplied `relPSA_Indi` = PSA / **per-patient maximum** (`n0` = 0.046–0.738) |
| `fR` | 1e-5 | **1e-2** |
| `cost` | 0.1 | **0.01** |
| First-cycle window | `CycleId == 1` plus one row | to the first off→on drug switch |

The Zhang priors differ from the Bruchovsky ones by 10× and 1000× and were not reported.
Because `n0 = 1.0` for all Bruchovsky patients, any κ defined as a fixed proportion of `n0`
is rank-identical to a constant in that cohort.

## 6. `n_crit` removed from the AT Score; constant κ

The published Bruchovsky AT Score used `kappa = n_crit / K` with `n_crit` computed from PSA
**up to clinical TTP** — i.e. from the outcome window, so it is not a first-cycle quantity.
The Zhang AT Score used a hard-coded `kappa = 0.2` with the `n_crit/K` line commented out,
so the two cohorts used different definitions.

Both now use a constant κ = 0.2. This improves the association in both cohorts on both
endpoints (Bruchovsky holiday fraction ρ +0.44 → +0.70; Zhang TTP ρ −0.14 → +0.76). The
patient ranking is invariant for κ ≲ 0.3 and collapses above ≈0.55.

## 7. The AT Score's primary endpoint is the drug holiday

The AT Score predicts the *relative TTP gain conferred by treatment breaks*, so the
quantity it speaks to most directly is how much drug-free time a patient receives.

| Cohort | Endpoint | ρ | P | N |
|---|---|---|---|---|
| Bruchovsky | **off-treatment fraction** | **+0.697** | **< .001** | 40 |
| Bruchovsky | off-treatment fraction, **cycle 2 onward** (fit sees cycle 1 only) | **+0.367** | **.020** | 40 |
| Bruchovsky | clinical TTP | +0.124 | .44 | 40 |
| Zhang | clinical TTP (TTP > 0, below pole) | **+0.764** | **.0024** | 13 |
| Zhang | off-treatment fraction | not assessable (N = 4 after identifiability filtering) | | |

The cohorts are not interchangeable. Bruchovsky used **fixed absolute** PSA thresholds
(4 and 10 µg/L on raw PSA), so holiday length is set by each patient's own regrowth rate —
which turnover captures. Zhang used **AT50 with patient-relative** thresholds, which
normalises holiday length away, and is the only cohort that received a true AT protocol.

## 8. Data availability

`.gitignore` excludes `patients/Bruchovsky_et_al/*/`, so the deposit tracks **zero** patient
records and `predict_clinical_ttp` cannot be run from it. `make_etableS8_inputs.py` writes
`paper_figures/patient_fits/eTableS8_inputs.csv` containing the three derived quantities the
function needs — `PSA_raw(t0)`, `N*` and the time-below-threshold fraction — so the table is
reproducible from the deposit alone.

## 9. `calc_delta_benefit` is non-monotone in turnover

`calc_delta_benefit` rises toward +∞ and changes sign where the resistant net growth rate
`rR(1-κ) - dR` passes through zero, i.e. at `d/rS = (1-cost)(1-κ)` — **0.7196** for the
Bruchovsky priors, inside the fitted bound [0.01, 1]. Three Bruchovsky and four Zhang fits
land at or above it and receive large negative scores. `at_score_pole()` now computes the
bound and fits above it are flagged and excluded.

## 10. Optimiser start value and reported hazard ratios

`params.add('turnover', min=0.01, max=1.)` supplied no `value`, so lmfit initialised the
parameter **on its lower bound**. An explicit mid-range start (0.3) is now used.

Separately, note for the manuscript (no code change here): the figure notebooks report
`1/HR` for every Cox covariate. The fitted Bruchovsky AT-Score coefficient is **+0.721**
(HR 2.06 per +1 SD of the published score), reported as 0.49. Under the corrected protocol
with a constant κ the AT Score is not significantly associated with Bruchovsky TTP in either
direction, and the holiday fraction is the endpoint that should be reported there.

---

## What is *not* changed

The optimiser (`method='least_squares'`, TRF — not Levenberg–Marquardt as the supplement
states), the solver tolerances, `max_nfev = 200`, the raw-residual objective, and the
Lotka–Volterra model itself are unchanged, so the corrected fits remain directly comparable
to the published ones.

## Addendum: landmark re-analysis, holiday fraction vs TTP, Zhang fit window

* `analysis/holiday_fraction_utils.py`, `analysis/landmark_holiday_analysis.py`, `paper_figures/SI_Landmark_and_Holiday_Fraction.ipynb`
  (rebuilt by `paper_figures/build_landmark_holiday_notebook.py`).
* **[C5] Zhang fit window.** 13 of 17 Zhang files begin with pre-treatment, off-drug rows. `TruncateDataframe` (used by the notebooks and by
  `patients/first_cycle_fitting.py`) cuts at the first off -> on switch, which for these patients is the start of abiraterone, so the fitted
  'first cycle' is the pre-treatment segment (9 patients end up with < 3 points). `analysis/landmark_holiday_analysis.zhang_window(mode='treated')`
  fits the first treated cycle instead. The corrected-branch Zhang CSV (`Zhang_first_cycle_corrected.csv`) still uses the published window.
* Castrate-sensitive: subtracting the first-cycle length from the progression times does not change the (null) result.
* Holiday fraction vs TTP is not significant for raw TTP in any cohort; the castrate-sensitive nominal signal is explained by cycle length.

### Effect of the Zhang fit window on the published statistics (`analysis/zhang_window_effect.py`)
`analysis/zhang_published_pipeline.py` reproduces the manuscript's Zhang statistics exactly from the published fits
(N = 13: Spearman 0.764, P = .0024; Cox HR 0.41, 0.16-1.07, P = .068; OS log-rank AT score P = .037, eTTP P = .029) and refitting the published window reproduces the published
CSV to 1e-16. Refitting the first TREATED cycle with otherwise identical settings gives N = 11, Spearman 0.38 (P = .25), Cox HR 0.97 (0.55-1.72), P = .93,
OS log-rank AT score P = .135 (eTTP .048). On the 10 patients common to both analyses the Spearman falls from 0.83 to 0.41 (P .003 -> .24), so the change is due to the fits, not the patient set.
Standard-metric benchmarks use relPSA_Indi (relative PSA), not raw PSA, in the notebook; raw PSA gives a nadir log-rank P of .35 rather than > .5.

### Decision: the Zhang fit window is NOT changed
The Zhang cohort keeps the published first-cycle window (`TruncateDataframe` from the first record; `patients/first_cycle_fitting.py` and
`Zhang_first_cycle_corrected.csv` are unchanged in this respect). The treated-cycle refits in `analysis/landmark_holiday_analysis.py`
(`zhang_window(mode='treated')`, `zhang_window_sensitivity`) and `analysis/zhang_window_effect.py` are sensitivity analyses only; they are not used by any
default fitting path or table. Observation [C5] above (13/17 files start off drug, so the published window is the pre-treatment segment for them) is kept
as a documented property of the published analysis, not as a correction that has been applied.
