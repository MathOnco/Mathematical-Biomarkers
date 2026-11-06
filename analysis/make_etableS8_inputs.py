"""
Deposit the derived per-patient quantities needed to reproduce eTable S8 without
the raw clinical records, which .gitignore excludes (CORRECTIONS.md item 8).

predict_clinical_ttp() reads PSA_raw(t0), N* and the time-below-threshold
fraction from each patient's record. Those three numbers are written here so an
external reader can reproduce the table from the deposit alone.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
from myUtils import (LoadPatientData, fraction_time_below_threshold,
                     predict_at_threshold_ttp_simp, predict_ct_ttp)

DATA = os.path.join(ROOT, "patients", "Bruchovsky_et_al", "Complete_data")
PATIENTS = [12, 20, 25, 78, 85]

rows = []
fits = pd.read_csv(os.path.join(ROOT, "paper_figures", "patient_fits",
                                "first_cycle_params.csv"))
for pid in PATIENTS:
    p = fits[fits.PatientId == pid].iloc[0].to_dict()
    d = LoadPatientData(pid, DATA)
    d = d[d.PSA.notna()]
    ttp = d[d.PSA < 1.2 * d.PSA.iloc[0]].Time.iloc[-1]
    d = d[d.Time <= ttp]
    thr = 2.5 / d.PSA_raw.iloc[0]
    N_star = d[d.PSA >= thr].PSA.mean()
    frac = fraction_time_below_threshold(d, thr)
    ct = predict_ct_ttp(**p)
    at = predict_at_threshold_ttp_simp(p["n0"] * N_star, **p)
    rows.append(dict(PatientId=pid, PSA_raw_t0_ng_per_mL=d.PSA_raw.iloc[0],
                     clinical_PSA_threshold_ng_per_mL=2.5,
                     threshold_normalised=thr, N_star=N_star,
                     fraction_time_below_threshold=frac,
                     predict_ct_ttp_days=ct, at_threshold_ttp_simp_days=at,
                     eTTP_days=ct + (1 - frac) * (at - ct),
                     turnover=p["turnover"], K=p["K"], n0=p["n0"], fR=p["fR"],
                     cost=p["cost"], rS=p["rS"], dD=p["dD"], R0=p["R0"]))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "paper_figures", "patient_fits",
                        "eTableS8_inputs.csv"), index=False)
print(out[["PatientId", "N_star", "fraction_time_below_threshold",
           "predict_ct_ttp_days", "at_threshold_ttp_simp_days", "eTTP_days"]].round(3)
      .to_string(index=False))
