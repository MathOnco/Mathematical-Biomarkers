"""
Canonical first-cycle fitting protocol for both clinical cohorts.

This replaces the fitting code that previously existed only inside the figure
notebooks (SI_Right_Censoring.ipynb cells 9-11, SI_Zhang_Right_Censoring.ipynb
cells 7-9, Fig2_Probing_Cycle.ipynb cells 24-26). `patients/patient_fitting.py`
is retained for the full-history "virtual patient twin" fits only.

Differences from the protocol used for the published tables, each flagged inline:
  [C1] The fitted carrying capacity K is RETAINED. The published Bruchovsky
       tables overwrote it with the constant 2.5 after fitting (`trunc_df['K'] = 2.5`),
       which differed from the fitted value for 21 of 72 patients. The override is
       recorded in a separate column rather than applied.
  [C2] `turnover` is given an explicit mid-range start value. Supplying only
       `min=` made lmfit initialise it on its lower bound (0.01).
  [C3] Fits whose turnover lands at or above the pole of calc_delta_benefit
       (dR -> rR(1-kappa)) are flagged; the AT score is not an ordered quantity there.
  [C4] Patients whose first cycle contains fewer than 3 PSA measurements are
       flagged as structurally unidentifiable (2 free parameters). The published
       Zhang analysis retained 1- and 2-point fits, for which R^2 is NaN or
       exactly 1 by construction.

Usage:  python patients/first_cycle_fitting.py            # from the repository root
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "utils"))

from lmfit import minimize, Parameters
from myUtils import LoadPatientData, TruncateDataframe, GetPatientIDs, test_progression
from fittingUtils import residual, compute_r_sq
from odeModels import LotkaVolterraModel

# ----------------------------------------------------------------- settings
SOLVER_KWS = {"method": "RK45", "absErr": 1.0e-10, "relErr": 1.0e-10,
              "suppressOutputB": False}
OPT_KWS = {"method": "least_squares", "xtol": 1e-8, "ftol": 1e-8,
           "max_nfev": 200, "nan_policy": "omit", "verbose": 0}

TURNOVER_START = 0.3          # [C2] explicit start; was implicitly min = 0.01
AT_SCORE_POLE  = 0.7196       # [C3] for cost = 0.1, K = 2.5, fR = 1e-5, n0 = 1
MIN_POINTS     = 3            # [C4] 2 free parameters

COHORTS = {
    "Bruchovsky": dict(data="patients/Bruchovsky_et_al/Complete_data",
                       loader="bruchovsky", fR=1e-5, cost=0.1, window="cycle1_plus1"),
    "Zhang":      dict(data="patients/Zhang_et_al/at_patients/data",
                       loader="zhang",      fR=1e-2, cost=0.01, window="to_next_on"),
}


def load_zhang(patient_id, data_dir):
    """Zhang records ship a per-patient-max normalised column (relPSA_Indi).
    Bruchovsky records are normalised by the FIRST measurement inside
    myUtils.LoadPatientData. The two cohorts are therefore on different scales;
    see CORRECTIONS.md item 5."""
    df = pd.read_csv(os.path.join(data_dir, "patient%.3d.csv" % patient_id))
    df = df.rename(columns={"Days": "Time", "Abi": "DrugConcentration",
                            "PSA": "PSA_raw", "relPSA_Indi": "PSA"})
    return df.sort_values("Time")


def first_cycle(patient_id, cfg, data_dir):
    if cfg["loader"] == "bruchovsky":
        d = LoadPatientData(patient_id, data_dir).dropna(subset=["PSA"]).reset_index()
        last = d[d["CycleId"] == 1].index[-1]
        return d[: last + 2]                     # cycle 1 plus first point of cycle 2
    d = load_zhang(patient_id, data_dir).dropna(subset=["PSA"]).reset_index()
    return TruncateDataframe(d, drug_name="DrugConcentration")   # to first off->on switch


def build_params(fR, cost, n0):
    p = Parameters()
    p.add("turnover", value=TURNOVER_START, min=0.01, max=1.0, vary=True)   # [C2]
    p.add("K", value=3, min=2.5, max=5, vary=True)
    p.add("fR", value=fR, vary=False)
    p.add("cost", value=cost, vary=False)
    p.add("rS", value=0.027, vary=False)
    p.add("n0", value=n0, vary=False)
    p.add("rR", expr="(1-cost)*rS", vary=False)
    p.add("dS", expr="turnover*rS", vary=False)
    p.add("dR", expr="dS", vary=False)
    p.add("dD", value=1.5, vary=False)
    p.add("theta", value=1, vary=False)
    p.add("D", value=1, vary=False)
    p.add("DMax", value=1, vary=False)
    p.add("S0", expr="n0*(1-fR)", vary=False)
    p.add("R0", expr="n0*fR", vary=False)
    return p


def fit_cohort(name, cfg):
    data_dir = os.path.join(ROOT, cfg["data"])
    rows = []
    for pid in GetPatientIDs(data_dir):
        d = first_cycle(pid, cfg, data_dir)
        params = build_params(cfg["fR"], cfg["cost"], d.PSA.iloc[0])
        model = LotkaVolterraModel()
        model.SetParams(**params)
        fit = minimize(residual, params, args=(d, model, "PSA", SOLVER_KWS), **OPT_KWS)
        v = fit.params.valuesdict()
        rows.append({"PatientId": pid, "n_first_cycle_points": len(d),
                     "RSquared": compute_r_sq(fit, d, feature="PSA"),
                     "AIC": fit.aic, "Message": fit.message, **v})
    df = pd.DataFrame(rows).sort_values("PatientId").reset_index(drop=True)
    df["Cost"] = 1 - df["rR"] / df["rS"]
    df["Turnover"] = df["dS"] / df["rS"]
    df["K_published_override"] = 2.5 if name == "Bruchovsky" else np.nan   # [C1]
    df["flag_above_at_score_pole"] = df["turnover"] >= AT_SCORE_POLE       # [C3]
    df["flag_unidentifiable"] = df["n_first_cycle_points"] < MIN_POINTS    # [C4]
    df["Progression"] = df.apply(lambda r: test_progression(r.to_dict()), axis=1)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "paper_figures", "patient_fits")
    os.makedirs(out_dir, exist_ok=True)
    for name, cfg in COHORTS.items():
        df = fit_cohort(name, cfg)
        fname = {"Bruchovsky": "first_cycle_params_all_corrected.csv",
                 "Zhang": "Zhang_first_cycle_corrected.csv"}[name]
        df.to_csv(os.path.join(out_dir, fname), index=False)
        print(f"{name}: n={len(df)}  progressors={int(df.Progression.sum())}  "
              f"above pole={int(df.flag_above_at_score_pole.sum())}  "
              f"unidentifiable={int(df.flag_unidentifiable.sum())}  -> {fname}")
