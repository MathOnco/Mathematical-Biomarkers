# Corrections to the fitting protocol and biomarker code

This branch brings the deposited code into agreement with what was actually run to produce the published numbers, fixes the defects that
agreement exposed, and evaluates the AT Score against the endpoint it is derived for.

Reproduce with:

```bash
python patients/first_cycle_fitting.py          # corrected first-cycle fits, both cohorts
python analysis/holiday_endpoint.py             # Bruchovsky holiday endpoint, Zhang staged TTP analysis
python analysis/make_etableS8_inputs.py         # derived inputs behind eTable S8
python analysis/zhang_holiday_published_fits.py # Zhang AT score vs off-treatment fraction (published fits)
# notebooks (paper_figures/): SI_Holiday_Fraction_Bruchovsky.ipynb, SI_Holiday_Fraction_Zhang.ipynb
# rebuilt with build_holiday_fraction_bruchovsky_notebook.py / build_holiday_fraction_zhang_notebook.py
```

---

## 1. The fitted carrying capacity is no longer discarded

The published Bruchovsky tables ran `trunc_df['K'] = 2.5` **after** fitting, overwriting the fitted value. The fitted K reached its upper bound 5.0
for 19 of 72 patients and an interior value for 2: **21 of 72 published K values were not what the optimiser returned**.
`first_cycle_fitting.py` retains the fitted K and records the old constant in a separate `K_published_override` column.

## 2. Notebook-local biomarker functions moved into `utils/`

`predict_clinical_ttp`, `predict_at_threshold_ttp_simp`, `calc_threshold_benefit` and `fraction_time_below_threshold` existed only inside
`SI_Identifiability_Analysis.ipynb`, so readers working from `utils/myUtils.py` could not reproduce eTable S8. They are now in `utils/myUtils.py`,
together with `fraction_time_on_treatment` and `at_score_pole`. Existing functions are unchanged.

## 3. Silent patient exclusions replaced by stated rules

The Zhang analysis dropped patients **1004 and 1009** through a bare `except` (`int(NaN)` raises for them). Both have clinical TTP = 0 under the
1.2x-first-PSA rule because they enrolled near their PSA nadir. The exclusion is defensible but was undocumented, and the published association
depended on it. It is now an explicit, reported filter (`exclude TTP = 0`) with the association shown at every filtering stage
(`analysis/holiday_endpoint.py`).

## 4. Structurally unidentifiable fits are flagged

Two parameters (turnover, K) are fitted. **9 of 17 Zhang patients have fewer than 3 first-cycle PSA measurements**; 5 have exactly one, for which R^2
is undefined and the optimiser cannot move from its start value, and 4 have two, for which R^2 = 1 by construction. The published filter
`RSquared.isna() | RSquared > 0.5` retains all of these. They are now flagged (`flag_unidentifiable`).

## 5. Cohort-specific priors and normalisation documented

| | Bruchovsky | Zhang |
|---|---|---|
| PSA normalisation | by the **first** measurement (so `n0 = 1.0` for every patient) | supplied `relPSA_Indi` = PSA / **per-patient maximum** (`n0` = 0.046-0.738) |
| `fR` | 1e-5 | **1e-2** |
| `cost` | 0.1 | **0.01** |
| First-cycle window | `CycleId == 1` plus one row | to the first off -> on drug switch (see item 11) |

The Zhang priors differ from the Bruchovsky ones by 10x and 1000x and were not reported.

## 6. `n_crit` removed from the AT Score; constant kappa

The published Bruchovsky AT Score used `kappa = n_crit / K` with `n_crit` computed from PSA **up to clinical TTP**, i.e. from the outcome window, so it is
not a first-cycle quantity. The Zhang AT Score used a hard-coded `kappa = 0.2` with the `n_crit/K` line commented out, so the two cohorts used different
definitions. Both now use a constant kappa = 0.2. For Zhang (N = 13), `kappa = n_crit/K` would give Spearman +0.23 (P = .46) with the published fits,
against +0.76 (P = .0024) with the constant kappa that the published Zhang numbers already used.

## 7. Evaluation endpoints

The AT Score predicts the *relative* TTP gain conferred by treatment breaks, so the quantity it speaks to most directly is how much drug-free time a
patient receives. The cohorts are not interchangeable: Bruchovsky used **fixed absolute** PSA thresholds (4 and 10 ug/L on raw PSA), so holiday length
is set by each patient's own regrowth rate, which turnover captures; Zhang used **AT50 with patient-relative** thresholds, which standardises holiday
length, and is the only cohort that received a true adaptive protocol.

| Cohort | Endpoint | rho | P | N |
|---|---|---|---|---|
| Bruchovsky | off-treatment fraction (whole record) | +0.697 | < .001 | 40 |
| Bruchovsky | off-treatment fraction, cycle 2 onward (fit sees cycle 1 only) | +0.367 | .020 | 40 |
| Zhang (published fits) | clinical TTP, manuscript patients | +0.764 | .0024 | 13 |
| Zhang (published fits) | off-treatment fraction, after the first cycle | +0.424 | .149 | 13 |
| Zhang (published fits) | off-treatment fraction, whole record | +0.236 | .437 | 13 |

`SI_Holiday_Fraction_Bruchovsky.ipynb` adds partial correlations (holding the holiday fraction within the fitted cycle and follow-up fixed), a
rank regression against the benchmark PSA metrics, and a selection check. `SI_Holiday_Fraction_Zhang.ipynb` reproduces the manuscript Zhang statistics
(Spearman 0.76, P = .002; Cox HR 0.41, 95% CI 0.16-1.07, P = .068; OS log-rank P = .037 for the AT score and .029 for eTTP) from the published fit table
`paper_figures/patient_fits/Zhang_first_cycle_published.csv` and then computes the holiday correlations on the same patients. On the corrected fits the
Zhang holiday endpoint is not assessable (N = 4 after the identifiability filter).

## 8. Data availability

Patient-level files are not distributed: `.gitignore` excludes `patients/Bruchovsky_et_al/*/` and the Zhang patient files and outcomes workbook are not
tracked, so the deposit tracks **zero** patient records. The scripts and notebooks that read them (`first_cycle_fitting.py`, `holiday_endpoint.py`,
`zhang_holiday_published_fits.py`, the two `SI_Holiday_Fraction_*` notebooks) need the data placed under `patients/` locally.
For eTable S8, `make_etableS8_inputs.py` writes `paper_figures/patient_fits/eTableS8_inputs.csv` containing the three derived quantities that
`predict_clinical_ttp` needs (`PSA_raw(t0)`, `N*` and the time-below-threshold fraction), so the table is reproducible from the deposit alone.
The tracked fit tables (`first_cycle_params_all_corrected.csv`, `Zhang_first_cycle_corrected.csv`, `Zhang_first_cycle_published.csv`) are the
fitted-parameter outputs. The Zhang OS / rPFS statistics additionally need the outcomes workbook and are skipped when it is absent.

## 9. `calc_delta_benefit` is non-monotone in turnover

`calc_delta_benefit` rises toward +infinity and changes sign where the resistant net growth rate `rR(1-kappa) - dR` passes through zero, i.e. at
`d/rS = (1-cost)(1-kappa)`: 0.72 for the Bruchovsky priors and 0.792 for the Zhang priors, inside the fitted bound [0.01, 1]. Three Bruchovsky and four
Zhang fits land at or above it and receive large negative scores. `at_score_pole()` computes the bound and such fits are flagged and excluded.

## 10. Optimiser start value

`params.add('turnover', min=0.01, max=1.)` supplied no `value`, so lmfit initialised the parameter **on its lower bound**. An explicit mid-range start
(0.3) is used in `first_cycle_fitting.py`. This only matters for fits the data do not constrain (the single-point Zhang fits, whose turnover is then
simply the start value).

## 11. Zhang first-cycle window (documented property of the published analysis)

13 of 17 Zhang files begin with pre-treatment, off-drug rows. `TruncateDataframe` cuts at the first off -> on switch, which for those patients is the
start of abiraterone, so for them the fitted "first cycle" is the pre-treatment segment (1-8 rows; 9 patients end up with fewer than 3 points). The
published window is retained unchanged, so the Zhang results here are those of the published analysis; this limitation should be stated alongside them.

## 12. Removed material

The cohort-level analysis of the AT Score against clinical TTP in the Bruchovsky cohort (`paper_figures/SI_Right_Censoring.ipynb`) is removed: the
Bruchovsky cohort is evaluated against the holiday fraction (item 7). The 5-patient case study and eTable S8 machinery are unchanged.

## What is *not* changed

The optimiser (`method='least_squares'`, TRF, not Levenberg-Marquardt as the supplement states), the solver tolerances, `max_nfev = 200`, the
raw-residual objective, and the Lotka-Volterra model itself are unchanged, so the corrected fits remain directly comparable to the published ones.
