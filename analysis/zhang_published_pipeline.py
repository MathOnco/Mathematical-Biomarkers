"""
The Zhang (mCRPC) statistics of the manuscript, as computed in SI_Zhang_Right_Censoring.ipynb
(cells 17-18, 29-33, 38-44), written as a function of the fits table.

Pipeline (verbatim logic of the notebook):
  fits -> keep Progression == True -> AT score (kappa = 0.2, hard-coded in the working tree) and eTTP
  ("Clinical_Preds" = predict_at_threshold_ttp(n0 * N*)) inside try/except (patients whose clinical TTP cannot be computed drop out)
  -> keep R^2 NaN or > 0.5 -> Spearman(AT score, clinical TTP); univariable Cox on the z-scored score (HR per +1 SD, event = PSA ever > 1.2 x first PSA)
  -> merge with the outcomes workbook (Sheet2) and median-split log-rank tests for OS and rPFS.
  -> (added) Spearman correlation of the AT score with the fraction of the record spent off treatment (`holiday_correlations`).
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for sub in ("utils", "patients"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lmfit import minimize
from myUtils import (calc_average_overshoot, calc_delta_benefit, test_progression, TruncateDataframe, GetPatientIDs)
from fittingUtils import residual, compute_r_sq
from odeModels import LotkaVolterraModel
import first_cycle_fitting as fcf

ZDATA = os.path.join(ROOT, "patients", "Zhang_et_al", "at_patients", "data")
PUBLISHED_FITS = os.environ.get("ZHANG_PUBLISHED_FITS", os.path.join(ROOT, "paper_figures", "patient_fits", "Zhang_first_cycle_published.csv"))
OUTCOMES_XLSX = os.path.join(ROOT, "patients", "Zhang_et_al", "Adaptive abi_Subject outcomes with OS update 050326.xlsx")


def load(pid):
    """PSA column = relPSA_Indi (PSA / per-patient maximum over the whole record), as in the published analysis."""
    d = fcf.load_zhang(pid, ZDATA)
    d = d.assign(DrugConcentration=d.DrugConcentration.ffill().fillna(0))
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- refit with the published settings (verification only)
def window(pid):
    """First-cycle window as published: rows up to the first off -> on drug switch, counted from the first record; time zeroed at the first row."""
    d = load(pid).dropna(subset=["PSA"]).reset_index(drop=True)
    w = TruncateDataframe(d, drug_name="DrugConcentration").copy()
    w["Time"] = w.Time - w.Time.iloc[0]
    return w


def refit_published(fR=1e-2, cost=0.01):
    """Refit all Zhang patients with the published settings (turnover starts on its lower bound, 0.01). Reproduces the published
    fit table (AT_Benefit_Prediction/paper_figures/patient_fits/Zhang_first_cycle.csv) to ~1e-16."""
    rows = []
    for pid in GetPatientIDs(ZDATA):
        w = window(pid)
        params = fcf.build_params(fR, cost, w.PSA.iloc[0]); params["turnover"].value = 0.01
        model = LotkaVolterraModel(); model.SetParams(**params)
        fit = minimize(residual, params, args=(w, model, "PSA", fcf.SOLVER_KWS), **fcf.OPT_KWS)
        v = fit.params.valuesdict()
        rows.append({"PatientId": pid, "n_points": len(w), "RSquared": compute_r_sq(fit, w, feature="PSA"), **v,
                     "Progression": bool(test_progression(v))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- notebook helpers (cells 14-16 / 29)
def _psa_data(pid):
    return load(pid).dropna(subset=["PSA"])

def get_ttp(pid):
    d = _psa_data(pid); d = d[d.PSA < 1.2 * d.PSA.iloc[0]]; return d.Time.iloc[-1]

def biochemical_progression(pid):
    d = _psa_data(pid); return bool(d.PSA.max() > 1.2 * d.PSA.iloc[0])

def get_N_star(pid, **kw):
    ttp = get_ttp(pid); d = _psa_data(pid); d = d[d.Time <= ttp]
    N = d.PSA.mean(); interval = d.Time.diff().mean()
    return N + calc_average_overshoot(int(interval), N, **kw)

def predict_ct_ttp(n_lim=None, **p):
    if n_lim is None: n_lim = p["n0"] * 1.2
    f = p["rR"] - p["dR"]
    return np.log((p["rR"] - f * (p["K"] / p["R0"])) / (p["rR"] - f * (p["K"] / n_lim))) / f

def predict_at_threshold_ttp(n_crit, **p):
    return predict_ct_ttp(**p) * (1 + calc_delta_benefit(kappa=n_crit / p["K"], **p))


def build_table(fits):
    """Published filters: model must predict progression (test_progression) and R^2 is NaN or > 0.5."""
    rows = []
    keep = fits.PatientId[fits.Progression == True].values
    for pid in [x for x in GetPatientIDs(ZDATA) if x in keep]:
        p = fits.loc[fits.PatientId == pid].iloc[0].to_dict()
        try:
            ct = int(predict_ct_ttp(**p)); at = calc_delta_benefit(kappa=0.2, **p)
            clin = int(predict_at_threshold_ttp(p["n0"] * get_N_star(pid, **p), **p)); ttp = int(get_ttp(pid))
            rows.append([pid, p["RSquared"], ct, at, clin, ttp, p["turnover"], p.get("n_points", np.nan)])
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["PatientId", "RSquared", "CT_Perfect", "AT_Score", "Clinical_Preds", "Clinical_TTP", "turnover", "n_points"])
    df = df[df.RSquared.isna() | (df.RSquared > 0.5)].reset_index(drop=True)
    df["Event"] = [int(biochemical_progression(x)) for x in df.PatientId]
    return df


# ---------------------------------------------------------------- statistics
def cox_uni(df, col, dur="Clinical_TTP", ev="Event"):
    sub = df[[col, dur, ev]].dropna().copy(); sd = sub[col].std()
    sub[col] = (sub[col] - sub[col].mean()) / sd
    s = CoxPHFitter().fit(sub, duration_col=dur, event_col=ev).summary
    return dict(HR=float(s.loc[col, "exp(coef)"]), lo=float(s.loc[col, "exp(coef) lower 95%"]),
                hi=float(s.loc[col, "exp(coef) upper 95%"]), p=float(s.loc[col, "p"]))

def _parse(row, col, fallback="on study (months)"):
    val = row[col]
    if isinstance(val, (int, float)) and not pd.isna(val): return val, 1
    s = str(val).strip()
    if ">" in s: return float(s.replace(">", "").strip()), 0
    if "*" in s:
        try: return float(s.replace("*", "").strip()), 1
        except ValueError: pass
    if s.replace(".", "", 1).isalpha() or "Not" in s or pd.isna(val): return float(row[fallback]), 0
    return np.nan, 0

def load_outcomes():
    df = pd.read_excel(OUTCOMES_XLSX, sheet_name="Sheet2")
    df[["OS_time", "OS_event"]] = df.apply(lambda r: pd.Series(_parse(r, "OS (months)")), axis=1)
    df[["rPFS_time", "rPFS_event"]] = df.apply(lambda r: pd.Series(_parse(r, "rPFS (months)")), axis=1)
    nan = df["Subject ID"].isna()
    if nan.any(): df = df.iloc[: np.where(nan.to_numpy())[0][0]].copy()
    return df.dropna(axis=1, how="all").rename(columns={"Subject ID": "PatientId"})

def logrank_median(df, col, t, e):
    sub = df.dropna(subset=[col, t, e]); hi = sub[col] >= sub[col].median()
    return logrank_test(sub[hi][t], sub[~hi][t], event_observed_A=sub[hi][e], event_observed_B=sub[~hi][e]).p_value

def standard_metrics(pid):
    """Baseline PSA, nadir, time to nadir, doubling time on the notebook's first-cycle definition (relative PSA = relPSA_Indi, as in the notebook; raw PSA gives a nadir log-rank P of .35)."""
    d = load(pid).dropna(subset=["PSA"]).sort_values("Time").reset_index(drop=True)
    drug = d.DrugConcentration.to_numpy() > 0; seen_off = False; end = len(d) - 1
    for i in range(1, len(d)):
        if not drug[i]: seen_off = True
        if seen_off and drug[i]: end = i - 1; break
    c = d.iloc[: end + 1].groupby("Time", as_index=False).mean(numeric_only=True)
    t0, p0 = float(c.Time.iloc[0]), float(c.PSA.iloc[0]); k = int(np.nanargmin(c.PSA.to_numpy())); tn = float(c.Time.iloc[k])
    out = dict(PatientId=pid, Baseline_Tumor_Burden=p0, PSA_Nadir=float(c.PSA.iloc[k]), Time_to_Nadir=tn - t0, PSA_Doubling_Time_days=np.nan)
    reg = c[c.Time >= tn]; off = reg[reg.DrugConcentration <= 0]
    if len(off) >= 2: reg = off
    if len(reg) >= 2:
        x, y = reg.Time.to_numpy(float), reg.PSA.to_numpy(float); pos = y > 0
        if pos.sum() >= 2 and not np.allclose(x[pos], x[pos][0]):
            sl = np.polyfit(x[pos], np.log(y[pos]), 1)[0]
            if sl > 0: out["PSA_Doubling_Time_days"] = np.log(2) / sl
    return out

def holiday_correlations(T):
    """Spearman correlation of the AT score with the off-treatment fraction, for the patients in table T (published window).
    Endpoints follow analysis/holiday_fraction_utils.py: 'whole' = whole record from the first drug-on row; 'from_first_record' counts the
    pre-treatment lead-in of 13/17 patients as off; 'after_C1' = from the restart of treatment (end of the first cycle) to the end of the record."""
    import holiday_fraction_utils as hf
    H = hf.holiday_table("Zhang").set_index("PatientId")
    D = T.set_index("PatientId")[["AT_Score"]].join(H[["off_whole", "off_from_first_record", "off_after_C1"]])
    out = {}
    for key, col in (("whole", "off_whole"), ("from_first_record", "off_from_first_record"), ("after_C1", "off_after_C1")):
        sub = D[["AT_Score", col]].dropna(); r, p = spearmanr(sub.AT_Score, sub[col]); out[key] = dict(N=len(sub), rho=float(r), p=float(p))
    return out

def evaluate(fits, outcomes=None):
    """Manuscript statistics for a Zhang fits table. The OS / rPFS log-rank tests need the patient-level outcomes workbook, which is
    not distributed with the repository (patient data are git-ignored); when it is absent they are skipped (entries are None)."""
    if outcomes is None and os.path.exists(OUTCOMES_XLSX):
        outcomes = load_outcomes()
    T = build_table(fits)
    rho, p = spearmanr(T.AT_Score, T.Clinical_TTP)
    rho2, p2 = spearmanr(T.Clinical_Preds, T.Clinical_TTP)
    holiday = holiday_correlations(T)
    res = dict(holiday=holiday, N=len(T), events=int(T.Event.sum()), rho_AT=rho, p_rho_AT=p, rho_eTTP=rho2, p_rho_eTTP=p2,
               cox_AT=cox_uni(T, "AT_Score"), cox_eTTP=cox_uni(T, "Clinical_Preds"),
               N_OS=None, OS_events=None, OS_logrank=None, rPFS_logrank=None, OS_cox_AT=None, OS_cox_eTTP=None)
    M = None
    if outcomes is not None:
        m = pd.DataFrame([standard_metrics(x) for x in T.PatientId])
        M = outcomes.merge(T.merge(m, on="PatientId"), on="PatientId")
        res.update(N_OS=len(M), OS_events=int(M.OS_event.sum()),
                   OS_logrank={c: logrank_median(M, c, "OS_time", "OS_event") for c in ["AT_Score", "Clinical_Preds", "Baseline_Tumor_Burden", "PSA_Nadir", "Time_to_Nadir", "PSA_Doubling_Time_days"]},
                   rPFS_logrank={c: logrank_median(M, c, "rPFS_time", "rPFS_event") for c in ["AT_Score", "Clinical_Preds"]})
        try:
            res["OS_cox_AT"] = cox_uni(M, "AT_Score", "OS_time", "OS_event"); res["OS_cox_eTTP"] = cox_uni(M, "Clinical_Preds", "OS_time", "OS_event")
        except Exception as e:
            res["OS_cox_AT"] = res["OS_cox_eTTP"] = str(e)[:60]
    return res, T, M
