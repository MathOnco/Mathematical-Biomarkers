"""
Drug-holiday fraction, first-cycle length and time-to-progression for the three cohorts
(Bruchovsky, Zhang, castrate-sensitive), defined identically.

Definitions (fixed before any association was computed):
  first cycle         first DRUG-ON row -> the row at which treatment RESTARTS (start of cycle 2).
                      (13 of 17 Zhang files begin with pre-treatment, off-drug rows: the cycle starts at the first
                      drug-on row, not at the first record.  In the other cohorts these coincide.)
                      Bruchovsky: first row with CycleId == 2.  Castrate-sensitive: first row with Cycle == 2.
                      Zhang: first off -> on transition of the drug column (the files have no cycle label).
  cycle-1 off fraction  time with no drug / first-cycle length, step-holding the drug state of the preceding row.
  whole-record off fraction   the same over the entire record (Bruchovsky, Zhang only; the castrate-sensitive
                      workbook contains the first cycle only, so this is not available there).
  pre-TTP off fraction   the same over [0, TTP] (Bruchovsky, Zhang only).
  TTP / event         Bruchovsky and Zhang: the definitions used in SI_Right_Censoring / SI_Zhang_Right_Censoring
                      (TTP = last time the PSA is still below 1.2 x the first PSA; event = max PSA > 1.2 x first PSA).
                      Castrate-sensitive: TTE and status from progression_outcomes.tsv (time from the first ADT dose).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("utils", "patients", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from myUtils import LoadPatientData, GetPatientIDs
from first_cycle_fitting import load_zhang
import castrate_sensitive as cs

BRUCH_DIR = os.path.join(ROOT, "patients", "Bruchovsky_et_al", "Complete_data")
ZHANG_DIR = os.path.join(ROOT, "patients", "Zhang_et_al", "at_patients", "data")


def off_fraction(time, drug, t0, t1):
    """Fraction of [t0, t1] with no drug, step-holding each row's drug state until the next row.
    Only the span actually covered by the record is counted."""
    t = np.asarray(time, float); g = np.asarray(drug, float)
    o = np.argsort(t, kind="stable"); t, g = t[o], g[o]
    if len(t) < 2:
        return np.nan
    dt = np.clip(np.minimum(t[1:], t1) - np.maximum(t[:-1], t0), 0, None)
    tot = dt.sum()
    return float(dt[g[:-1] == 0].sum() / tot) if tot > 0 else np.nan


def first_on_time(time, drug):
    """Time of the first drug-on row (start of the first treated cycle)."""
    t = np.asarray(time, float); g = np.asarray(drug, float)
    return float(t[g == 1][0]) if (g == 1).any() else np.nan


def _restart_from_drug(time, drug):
    """Time of the first on-row that follows an off period AFTER treatment has started (the restart of treatment)."""
    t = np.asarray(time, float); g = np.asarray(drug, float)
    started = seen_off = False
    for ti, gi in zip(t, g):
        if gi == 1 and not started:
            started = True
        elif started and gi == 0:
            seen_off = True
        elif started and seen_off and gi == 1:
            return float(ti)
    return np.nan


def load_record(dataset, pid, drug="union"):
    """Standardised record: Time (days from first record / first ADT dose), PSA (relative), Drug (0/1), plus cycle info."""
    if dataset == "Bruchovsky":
        d = LoadPatientData(pid, BRUCH_DIR)
        d = d.assign(Drug=d.DrugConcentration.ffill().fillna(0).astype(float))
        restart_label = d.loc[d.CycleId == 2, "Time"]
        restart = float(restart_label.iloc[0]) if len(restart_label) else np.nan
        return d[["Time", "PSA", "Drug"]].reset_index(drop=True), restart, _restart_from_drug(d.Time, d.Drug)
    if dataset == "Zhang":
        d = load_zhang(pid, ZHANG_DIR)
        d = d.assign(Drug=d.DrugConcentration.ffill().fillna(0).astype(float))
        r = _restart_from_drug(d.Time, d.Drug)
        return d[["Time", "PSA", "Drug"]].reset_index(drop=True), r, r
    raise ValueError(dataset)


def clinical_ttp_event(rec):
    """TTP and event exactly as in the SI notebooks for Bruchovsky and Zhang."""
    d = rec[rec.PSA.notna()]
    ttp = float(d[d.PSA < 1.2 * d.PSA.iloc[0]].Time.iloc[-1]) if (d.PSA < 1.2 * d.PSA.iloc[0]).any() else 0.0
    ev = int(d.PSA.max() > 1.2 * d.PSA.iloc[0])
    return ttp, ev


def holiday_table(dataset):
    rows = []
    ids = GetPatientIDs(BRUCH_DIR if dataset == "Bruchovsky" else ZHANG_DIR)
    for pid in ids:
        rec, restart, restart_drug = load_record(dataset, pid)
        ttp, ev = clinical_ttp_event(rec)
        t_end = float(rec.Time.max()); t_on = first_on_time(rec.Time, rec.Drug)
        rows.append(dict(dataset=dataset, PatientId=pid, TTP=ttp, Event=ev,
                         t_first_on=t_on, cycle1_end=restart, cycle1_length=restart - t_on, restart_from_drug=restart_drug,
                         starts_off_drug=int(rec.Drug.iloc[0] == 0),
                         off_cycle1=off_fraction(rec.Time, rec.Drug, t_on, restart) if np.isfinite(restart) else np.nan,
                         off_whole=off_fraction(rec.Time, rec.Drug, t_on, t_end),
                         off_preTTP=off_fraction(rec.Time, rec.Drug, t_on, ttp) if ttp > t_on else np.nan,
                         record_length=t_end))
    return pd.DataFrame(rows)


def cs_table(outcomes_df, tabs, drug):
    """Castrate-sensitive: cycle-1 length and off fraction (first cycle only is available)."""
    rows = []
    for pid, t in tabs.items():
        rec = cs.build_record(t, origin="treatment", drug=drug)
        L = float(rec.loc[rec.CycleId == 2, "Time"].iloc[0])
        rows.append(dict(dataset=f"Castrate-sensitive ({drug})", PatientId=pid,
                         t_first_on=0.0, cycle1_end=L, cycle1_length=L, starts_off_drug=0,
                          off_cycle1=off_fraction(rec.Time, rec.DrugConcentration, 0, L),
                         TTP=float(outcomes_df.loc[pid, "TTE"]), Event=int(outcomes_df.loc[pid, "Event"])))
    return pd.DataFrame(rows)
