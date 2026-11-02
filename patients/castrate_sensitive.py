"""
Loader and fitting wrapper for the castrate-sensitive cohort (Zhang et al.).

Inputs (patients/Castrate_Sensitive/):
  MCC 19367_C1.xlsx        one tab per patient id (first treatment cycle, on/off) + a Summary tab
  progression_outcomes.tsv pat, TTE (days), status ('progressed' / 'not')

The Lotka-Volterra model, solver, optimiser settings and parameter block are IMPORTED
unchanged from patients/first_cycle_fitting.py. Only data preparation lives here.

Data-preparation decisions (all outcome-blind; see the notebook for the evidence):

  D1  Header row. Row 0 of each tab is a label row (`Enza`/`Apa`/blank); the real header
      is the row whose first cell is 'Days'.
  D2  NaN-PSA rows are KEPT. They carry drug on/off switches (e.g. patient 102's Abi stops on
      a row with no PSA). The shared residual() returns NaN there and lmfit's
      nan_policy='omit' discards it, so the schedule is exact and the fit uses PSA rows only.
      (Dropping them, as the Bruchovsky/Zhang code does, would hold drug on too long here.)
  D3  Drug state. 'ADT' = ADT column. 'union' = max(ADT, Abi, ARSI). Union is used when the
      record shows PSA falling while ADT = 0 but another agent is on.
  D4  Time origin. 'treatment' (default): t = 0 at the first ADT dose, matching the other
      cohorts where t = 0 is treatment initiation. If PSA at t = 0 is missing, the most recent
      PRE-treatment PSA is carried to t = 0 as the baseline; other Cycle-0 rows are dropped.
      'first_psa': the literal reading of the brief - every row from the first PSA, shifted so
      that the first PSA is at t = 0, pre-treatment rows retained with drug = 0.
  D5  Normalisation. PSA / baseline PSA (first measurement), so n0 = 1.0, as in Bruchovsky and
      in the recommended harmonised protocol.
  D6  Window. All Cycle 1 rows plus the first Cycle 2 row (treatment resumption), mirroring the
      Bruchovsky `CycleId == 1` + one extra row.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "utils"))
sys.path.insert(0, HERE)

from lmfit import minimize
from fittingUtils import residual
from odeModels import LotkaVolterraModel
from first_cycle_fitting import build_params, SOLVER_KWS, OPT_KWS, AT_SCORE_POLE

CS_DIR = os.path.join(ROOT, "patients", "Castrate_Sensitive")
XLSX = os.path.join(CS_DIR, "MCC 19367_C1.xlsx")
TSV = os.path.join(CS_DIR, "progression_outcomes.tsv")

PRIORS = {"Zhang-style": dict(fR=1e-2, cost=0.01),
          "Bruchovsky-style": dict(fR=1e-5, cost=0.1)}


# ------------------------------------------------------------------ loading
def load_tabs(path=XLSX):
    xl = pd.ExcelFile(path)
    tabs = {}
    for s in xl.sheet_names:
        if s == "Summary":
            continue
        raw = xl.parse(s, header=None)
        h = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Days"][0]          # [D1]
        d = raw.iloc[h + 1:].copy()
        d.columns = [str(c).strip() for c in raw.iloc[h]]
        for c in ["Days", "Cycle", "PSA", "Testosterone", "ADT", "Abi", "ARSI"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["Days"]).sort_values("Days").reset_index(drop=True)
        d["label_agent"] = str(raw.iloc[0, -1]) if str(raw.iloc[0, -1]) != "nan" else ""
        tabs[int(s)] = d
    return tabs


def load_summary(path=XLSX):
    s = pd.read_excel(path, sheet_name="Summary").rename(columns={"Unnamed: 0": "pat"})
    return s.set_index("pat")


def load_outcomes(path=TSV):
    o = pd.read_csv(path, sep="\t").rename(columns={"pat": "PatientId"})
    o["Event"] = (o["status"].str.strip().str.lower() == "progressed").astype(int)
    return o.set_index("PatientId")


# ------------------------------------------------------------------ record building
def drug_state(d, drug):
    if drug == "ADT":
        return d["ADT"].fillna(0).clip(0, 1)
    if drug == "union":                                                              # [D3]
        return d[["ADT", "Abi", "ARSI"]].fillna(0).max(axis=1).clip(0, 1)
    raise ValueError(drug)


def build_record(tab, origin="treatment", drug="union"):
    """Return a record with Time, PSA_raw, PSA (normalised), DrugConcentration, CycleId and
    a flag for whether the baseline PSA was carried from a pre-treatment measurement."""
    d = tab.copy()
    d["DrugConcentration"] = drug_state(d, drug)
    flag = False
    if origin == "treatment":                                                        # [D4]
        t0 = d.loc[(d.Cycle >= 1) & (d.ADT == 1), "Days"].min()
        pre = d[(d.Cycle == 0) & d.PSA.notna() & (d.Days <= t0)]
        d = d[d.Cycle >= 1].copy()
        d["Time"] = d["Days"] - t0
        row0 = d.index[d.Time == 0]
        if len(row0) and pd.isna(d.loc[row0[0], "PSA"]) and len(pre):
            d.loc[row0[0], "PSA"] = pre.iloc[-1]["PSA"]
            flag = True
    elif origin == "first_psa":
        d = d[d.Cycle >= 0].copy()
        first = d.loc[d.PSA.notna(), "Days"].min()
        d = d[d.Days >= first].copy()
        d["Time"] = d["Days"] - first
    else:
        raise ValueError(origin)
    d = d[d.Cycle <= 2].copy()                                                       # [D6]
    c2 = d.index[d.Cycle == 2]
    if len(c2):                                                                       # keep only the FIRST cycle-2 row
        d = d[(d.Cycle < 2) | (d.index == c2[0])]
    d = d.sort_values("Time").reset_index(drop=True)
    base = d.loc[d.PSA.notna(), "PSA"].iloc[0]
    d["PSA_raw"] = d["PSA"]
    d["PSA"] = d["PSA_raw"] / base                                                    # [D5]
    d["CycleId"] = d["Cycle"]
    d["baseline_from_pretreatment"] = flag
    return d[["Time", "PSA_raw", "PSA", "DrugConcentration", "CycleId", "ADT", "Abi", "ARSI",
              "baseline_from_pretreatment"]]


# ------------------------------------------------------------------ fitting
def r2_on_psa_rows(fit_params, rec):
    """Linear-space R^2 on rows that carry a PSA value (TSS about their mean)."""
    m = LotkaVolterraModel(); m.SetParams(**fit_params)
    from myUtils import ExtractTreatmentFromDf
    from scipy.interpolate import interp1d
    m.Simulate(treatmentScheduleList=ExtractTreatmentFromDf(rec), **SOLVER_KWS)
    f = interp1d(m.resultsDf.Time, m.resultsDf.TumourSize, fill_value="extrapolate")
    ok = rec.PSA.notna()
    obs = rec.loc[ok, "PSA"].values; pred = f(rec.loc[ok, "Time"].values)
    tss = np.sum((obs - obs.mean()) ** 2)
    return 1 - np.sum((obs - pred) ** 2) / tss if tss > 0 else np.nan


def fit_record(rec, priors="Zhang-style"):
    pr = PRIORS[priors]
    params = build_params(pr["fR"], pr["cost"], rec.loc[rec.PSA.notna(), "PSA"].iloc[0])
    model = LotkaVolterraModel(); model.SetParams(**params)
    fit = minimize(residual, params, args=(rec, model, "PSA", SOLVER_KWS), **OPT_KWS)
    v = fit.params.valuesdict()
    v["RSquared"] = r2_on_psa_rows(v, rec)
    v["n_psa_points"] = int(rec.PSA.notna().sum())
    v["success"] = bool(fit.success)
    return v


# ------------------------------------------------------------------ outcome reconciliation
def reconcile_outcomes(outcomes, summary):
    """Compare progression_outcomes.tsv with the workbook's own Summary tab ('PSA progression (days)').

    Returns (as_supplied, reconciled, audit). `as_supplied` is the TSV unchanged. `reconciled` adopts the
    Summary tab wherever it records a PSA progression that the TSV does not match (different time, or
    coded as not progressed). Both are analysed; neither is silently preferred.
    """
    a = outcomes[["TTE", "Event"]].copy()
    r = a.copy(); rows = []
    for pid in a.index:
        sp = summary.loc[pid, "PSA progression (days)"] if pid in summary.index else np.nan
        rp = summary.loc[pid, "rPFS (days)"] if pid in summary.index else np.nan
        if pd.notna(sp):
            mism_t = float(a.loc[pid, "TTE"]) != float(sp); mism_s = int(a.loc[pid, "Event"]) != 1
            if mism_t or mism_s:
                r.loc[pid, "TTE"] = float(sp); r.loc[pid, "Event"] = 1
            rows.append(dict(PatientId=pid, tsv_TTE=a.loc[pid, "TTE"], tsv_Event=int(a.loc[pid, "Event"]),
                             summary_PSA_progression=sp, summary_rPFS=rp,
                             TTE_mismatch=mism_t, status_mismatch=mism_s,
                             tsv_equals_rPFS=bool(pd.notna(rp) and float(a.loc[pid, "TTE"]) == float(rp))))
    return a, r, pd.DataFrame(rows).set_index("PatientId")
