"""
Primary endpoint analysis for the AT Score: the observed drug-holiday fraction.

The AT Score predicts the RELATIVE increase in TTP that treatment breaks confer,
so the quantity it speaks to most directly is how much drug-free time a patient
receives, not the absolute TTP. Bruchovsky (fixed-threshold intermittent ADT) is
evaluated against the holiday fraction; Zhang (the only cohort that received a
true AT50 protocol) additionally against clinical TTP, with the TTP = 0
exclusion applied as a stated rule.

Key results reproduced by this script (see CORRECTIONS.md):
  Bruchovsky, N=40: AT Score vs off-treatment fraction  rho = +0.697, P < .001
                    out-of-window (cycle 2 onward)      rho = +0.367, P = .020
  Zhang:            off-treatment fraction NOT ASSESSABLE -- 5 of 13 progressors
                    have a single first-cycle PSA point, so turnover is pinned at
                    its lower bound and 5 patients share an identical AT Score.
                    (For the Zhang holiday correlation on the published fits see
                    analysis/zhang_holiday_published_fits.py.)

Usage:  python analysis/holiday_endpoint.py        # from the repository root
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from scipy.stats import spearmanr
from myUtils import (LoadPatientData, GetPatientIDs, test_progression, calc_delta_benefit,
                     predict_ct_ttp, fraction_time_on_treatment, at_score_pole)
sys.path.insert(0, os.path.join(ROOT, "patients"))
from first_cycle_fitting import load_zhang, COHORTS

KAPPA = 0.2        # constant; see CORRECTIONS.md item 6 (n_crit removed)
MIN_POINTS = 3


def load(pid, cohort):
    cfg = COHORTS[cohort]
    d = os.path.join(ROOT, cfg["data"])
    return LoadPatientData(pid, d) if cohort == "Bruchovsky" else load_zhang(pid, d)


def clinical_ttp(pid, cohort):
    d = load(pid, cohort)
    d = d[d.PSA.notna()]
    return d[d.PSA < 1.2 * d.PSA.iloc[0]].Time.iloc[-1]


def progression_event(pid, cohort):
    d = load(pid, cohort)
    d = d[d.PSA.notna()]
    return int(d.PSA.max() > 1.2 * d.PSA.iloc[0])


def holiday_after_first_cycle(pid):
    """Out-of-window control for Bruchovsky: holiday fraction computed from
    cycle 2 onward, while the fit sees only cycle 1."""
    d = LoadPatientData(pid, os.path.join(ROOT, COHORTS["Bruchovsky"]["data"]))
    d = d.dropna(subset=["DrugConcentration"]).reset_index()
    idx = d[d.CycleId == 1].index
    if len(idx) == 0:
        return np.nan
    d2 = d[d.index > idx[-1]]
    return 1 - fraction_time_on_treatment(d2) if len(d2) >= 2 else np.nan


def build(cohort, fits_csv):
    fits = pd.read_csv(os.path.join(ROOT, "paper_figures", "patient_fits", fits_csv))
    pole = at_score_pole(cost=COHORTS[cohort]["cost"], kappa=KAPPA)
    rows = []
    for _, r in fits.iterrows():
        p = r.to_dict(); pid = int(p["PatientId"])
        if not test_progression(p):                      continue
        if p["turnover"] >= pole:                        continue
        if p["n_first_cycle_points"] < MIN_POINTS:       continue
        if not (np.isnan(p["RSquared"]) or p["RSquared"] > 0.5): continue
        try:
            if predict_ct_ttp(**p) <= 0:                 continue
            at = calc_delta_benefit(kappa=KAPPA, **p)
            if not np.isfinite(at):                      continue
        except Exception:
            continue
        d = load(pid, cohort)
        row = dict(PatientId=pid, AT_Score=at, turnover=p["turnover"],
                   n_first_cycle_points=int(p["n_first_cycle_points"]),
                   holiday_fraction=1 - fraction_time_on_treatment(d),
                   holiday_fraction_after_cycle1=(holiday_after_first_cycle(pid)
                                                  if cohort == "Bruchovsky" else np.nan))
        if cohort == "Zhang":      # TTP is the Zhang endpoint (the only true AT50 cohort); TTP = 0 is excluded by a stated rule
            T = clinical_ttp(pid, cohort)
            if T <= 0:                                   continue
            row.update(Clinical_TTP=T, Event=progression_event(pid, cohort))
        rows.append(row)
    return pd.DataFrame(rows)


def report(df, cohort):
    print(f"\n=== {cohort}  (N = {len(df)}) ===")
    if len(df) < 8:
        print("  NOT ASSESSABLE at this sample size after identifiability filtering.")
        return
    for col, label in [("holiday_fraction", "off-treatment fraction"),
                       ("holiday_fraction_after_cycle1", "off-treatment fraction, cycle 2 onward"),
                       ("Clinical_TTP", "clinical TTP")]:
        if col not in df.columns:
            continue
        s = df[[ "AT_Score", col]].dropna()
        if len(s) < 8:
            continue
        rho, p = spearmanr(s.AT_Score, s[col])
        print(f"  AT Score vs {label:38s} rho = {rho:+.3f}   P = {p:.4f}   (n = {len(s)})")


def zhang_ttp_analysis():
    """TTP endpoint for the Zhang cohort -- the only cohort that received a true
    AT50 protocol. Exclusions are applied as STATED RULES, replacing the bare
    `except` that silently dropped patients 1004 and 1009 (CORRECTIONS.md item 3)."""
    z = pd.read_csv(os.path.join(ROOT, "paper_figures", "patient_fits",
                                 "Zhang_first_cycle_corrected.csv"))
    pole = at_score_pole(cost=COHORTS["Zhang"]["cost"], kappa=KAPPA)
    rows = []
    for _, r in z.iterrows():
        p = r.to_dict(); pid = int(p["PatientId"])
        if not test_progression(p):
            continue
        try:
            if predict_ct_ttp(**p) <= 0:
                continue
            at = calc_delta_benefit(kappa=KAPPA, **p)
            if not np.isfinite(at):
                continue
        except Exception:
            continue
        rows.append(dict(PatientId=pid, AT_Score=at, turnover=p["turnover"],
                         n_first_cycle_points=int(p["n_first_cycle_points"]),
                         above_pole=bool(p["turnover"] >= pole),
                         Clinical_TTP=clinical_ttp(pid, "Zhang")))
    df = pd.DataFrame(rows)
    print("\n=== Zhang: TTP endpoint (only true AT50 cohort) ===")
    stages = [("all progressors", df),
              ("exclude TTP = 0 (enrolled at nadir)", df[df.Clinical_TTP > 0]),
              ("+ exclude turnover above AT-score pole",
               df[(df.Clinical_TTP > 0) & (~df.above_pole)]),
              ("+ exclude < 3 first-cycle points", 
               df[(df.Clinical_TTP > 0) & (~df.above_pole) & (df.n_first_cycle_points >= 3)])]
    for label, sub in stages:
        if len(sub) < 5:
            print(f"  {label:42s} N = {len(sub):2d}   too few to assess")
            continue
        rho, pv = spearmanr(sub.AT_Score, sub.Clinical_TTP)
        print(f"  {label:42s} N = {len(sub):2d}   rho = {rho:+.3f}   P = {pv:.4f}")
    print("  NOTE: the >=3-point stage leaves N = 4, so the Zhang TTP association")
    print("        cannot be shown to be independent of the unidentifiable 1- and")
    print("        2-point fits. State this as a limitation.")
    df.to_csv(os.path.join(ROOT, "analysis", "zhang_ttp_endpoint.csv"), index=False)
    return df


if __name__ == "__main__":
    out = {}
    for cohort, csv in [("Bruchovsky", "first_cycle_params_all_corrected.csv"),
                        ("Zhang", "Zhang_first_cycle_corrected.csv")]:
        df = build(cohort, csv); out[cohort] = df
        report(df, cohort)
        df.to_csv(os.path.join(ROOT, "analysis", f"holiday_endpoint_{cohort.lower()}.csv"),
                  index=False)
    z = pd.read_csv(os.path.join(ROOT, "paper_figures", "patient_fits",
                                 "Zhang_first_cycle_corrected.csv"))
    zhang_ttp_analysis()
    print(f"\nZhang identifiability: {int((z.n_first_cycle_points < MIN_POINTS).sum())} of {len(z)} "
          f"patients have fewer than {MIN_POINTS} first-cycle PSA measurements "
          f"(2 free parameters) -- see CORRECTIONS.md item 4.")
