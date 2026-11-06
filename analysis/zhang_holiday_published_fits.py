"""
Does the AT score predict the fraction of follow-up spent off treatment in the Zhang (mCRPC) cohort?
Uses the PUBLISHED fits and the published window (no refitting): AT score = calc_delta_benefit(kappa = 0.2) from
paper_figures/patient_fits/Zhang_first_cycle_published.csv (the fit table behind the manuscript's Zhang statistics; override with env ZHANG_PUBLISHED_FITS).
Endpoints (holiday_fraction_utils.py definitions): off fraction of the whole record from the first drug-on row; from the first record
(counts the pre-treatment lead-in of 13/17 patients as 'off'); and after the first cycle (restart to end of record).
Run from the repository root:  python analysis/zhang_holiday_published_fits.py
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for sub in ("utils", "patients", "analysis"): sys.path.insert(0, os.path.join(ROOT, sub))
import zhang_published_pipeline as zp, holiday_fraction_utils as hf
from scipy.stats import spearmanr
from myUtils import calc_delta_benefit

PUB = zp.PUBLISHED_FITS
POLE = 0.792

def boot_ci(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float); y = np.asarray(y, float); out = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) and np.ptp(y[i]): out.append(spearmanr(x[i], y[i])[0])
    return np.percentile(out, [2.5, 97.5])

def perm_p(x, y, n=20000, seed=0):
    rng = np.random.default_rng(seed); r0 = spearmanr(x, y)[0]; y = np.asarray(y, float)
    return (sum(abs(spearmanr(x, rng.permutation(y))[0]) >= abs(r0) - 1e-12 for _ in range(n)) + 1) / (n + 1)

def build():
    pub = pd.read_csv(PUB); f = pub.set_index("PatientId")
    H = hf.holiday_table("Zhang").set_index("PatientId").rename(columns={"off_whole": "off_from_first_drug_on"})
    pub13 = set(zp.build_table(pub).PatientId)
    rows = []
    for pid in H.index:
        p = f.loc[pid].to_dict()
        try: at = calc_delta_benefit(kappa=0.2, **p)
        except Exception: at = np.nan
        rows.append(dict(PatientId=pid, AT_Score=at, turnover=p["turnover"], RSquared=p["RSquared"], Progression=bool(p["Progression"]),
                         n_pts=len(zp.window(pid)), in_published_N13=pid in pub13))
    return pd.DataFrame(rows).set_index("PatientId").join(H[["off_from_first_drug_on", "off_from_first_record", "off_after_C1", "starts_off_drug", "cycle1_length", "record_length"]])

def main(out=HERE):
    A = build(); ok = A.RSquared.isna() | (A.RSquared > 0.5)
    sets = {"A. Published filters (N=13 of the manuscript)": A[A.in_published_N13],
            "B. Progression + R2 filters, no TTP requirement": A[A.Progression & ok],
            "C. Published N=13 minus the four one-point fits": A[A.in_published_N13 & (A.n_pts >= 2)],
            "D. B minus fits above the AT-score pole": A[A.Progression & ok & (A.turnover < POLE)],
            "E. All 17 patients, no filters": A}
    ends = [("off_from_first_drug_on", "whole record, from first drug-on row"),
            ("off_from_first_record", "whole record, from first record (pre-treatment rows count as off)"),
            ("off_after_C1", "after the first cycle (restart to end)")]
    rows = []
    for sn, d in sets.items():
        for col, lab in ends:
            s = d[["AT_Score", col]].dropna()
            if len(s) < 6 or s.AT_Score.nunique() < 3: continue
            r, p = spearmanr(s.AT_Score, s[col]); lo, hi = boot_ci(s.AT_Score, s[col])
            rows.append(dict(patient_set=sn, endpoint=lab, N=len(s), rho=r, P_spearman=p, P_permutation=perm_p(s.AT_Score.values, s[col].values), CI_lo=lo, CI_hi=hi))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(out, "zhang_holiday_fraction_published_fits.csv"), index=False)
    A.reset_index().to_csv(os.path.join(out, "zhang_holiday_fraction_per_patient.csv"), index=False)
    return R, A

if __name__ == "__main__":
    R, A = main(); pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 70); print(R.round(3).to_string(index=False))
