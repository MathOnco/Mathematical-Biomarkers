"""
Drug-holiday fraction and first-cycle definitions for the Bruchovsky and Zhang cohorts, defined identically.

Definitions (fixed before any association was computed):
  first cycle           first DRUG-ON row -> the row at which treatment RESTARTS (start of cycle 2).
                        Bruchovsky: first row with CycleId == 2.
                        Zhang (no cycle label in the files): first off -> on transition after treatment has started.
                        13 of 17 Zhang files begin with pre-treatment, off-drug rows, so the cycle starts at the first drug-on
                        row, not at the first record (in Bruchovsky the two coincide).
  off fraction          time with no drug / time in the window, step-holding the drug state of the preceding row.
  off_cycle1            off fraction over the first cycle.
  off_whole             off fraction over the whole record, from the first drug-on row.
  off_after_C1          off fraction from the restart of treatment (end of the first cycle) to the end of the record.
  off_from_first_record off fraction over the whole record counting any pre-treatment lead-in as 'off'.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("utils", "patients", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from myUtils import LoadPatientData, GetPatientIDs
from first_cycle_fitting import load_zhang

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


def load_record(dataset, pid):
    """Standardised record (Time, PSA, Drug) plus the restart time from the dataset's own cycle label and from the drug column."""
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


def holiday_table(dataset):
    rows = []
    ids = GetPatientIDs(BRUCH_DIR if dataset == "Bruchovsky" else ZHANG_DIR)
    for pid in ids:
        rec, restart, restart_drug = load_record(dataset, pid)
        t_end = float(rec.Time.max()); t_on = first_on_time(rec.Time, rec.Drug)
        rows.append(dict(dataset=dataset, PatientId=pid, t_first_on=t_on, cycle1_end=restart, cycle1_length=restart - t_on,
                         restart_from_drug=restart_drug, starts_off_drug=int(rec.Drug.iloc[0] == 0),
                         off_cycle1=off_fraction(rec.Time, rec.Drug, t_on, restart) if np.isfinite(restart) else np.nan,
                         off_whole=off_fraction(rec.Time, rec.Drug, t_on, t_end),
                         off_after_C1=off_fraction(rec.Time, rec.Drug, restart, t_end) if np.isfinite(restart) else np.nan,
                         off_from_first_record=off_fraction(rec.Time, rec.Drug, 0, t_end),
                         record_length=t_end))
    return pd.DataFrame(rows)
