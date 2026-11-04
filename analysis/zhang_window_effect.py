"""
How the fit window changes the published Zhang (mCRPC) statistics.

Scenario 1 reproduces the manuscript numbers from the published fits (AT_Benefit_Prediction working tree,
paper_figures/patient_fits/Zhang_first_cycle.csv) through the notebook pipeline in zhang_published_pipeline.py.
All other scenarios change ONLY the fit window / filters:  published window = rows up to the first off->on switch counted from the first
record (for 13/17 patients: the pre-treatment, off-drug rows);  treated window = first drug-on row -> last off row before the restart.

Run from the repository root:  python analysis/zhang_window_effect.py   (needs ../AT_Benefit_Prediction for the published CSV)
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import zhang_published_pipeline as zp
from scipy.stats import spearmanr, norm

PUB_CSV = os.environ.get("ZHANG_PUBLISHED_FITS",
          os.path.join(os.path.dirname(ROOT), "AT_Benefit_Prediction", "paper_figures", "patient_fits", "Zhang_first_cycle.csv"))
POLE = 0.792   # (1 - cost)(1 - kappa) for cost = 0.01, kappa = 0.2

def fisher_ci(rho, n, a=0.05):
    z = np.arctanh(rho); se = 1 / np.sqrt(n - 3); q = norm.ppf(1 - a / 2); return np.tanh(z - q * se), np.tanh(z + q * se)

def row(label, res):
    ca, ce, lr = res["cox_AT"], res["cox_eTTP"], res["OS_logrank"]
    lo, hi = fisher_ci(res["rho_AT"], res["N"])
    return dict(scenario=label, N=res["N"], rho_AT=res["rho_AT"], rho_AT_ci_lo=lo, rho_AT_ci_hi=hi, P_rho_AT=res["p_rho_AT"],
                HR_AT=ca["HR"], HR_AT_lo=ca["lo"], HR_AT_hi=ca["hi"], P_cox_AT=ca["p"],
                rho_eTTP=res["rho_eTTP"], P_rho_eTTP=res["p_rho_eTTP"], HR_eTTP=ce["HR"], HR_eTTP_lo=ce["lo"], HR_eTTP_hi=ce["hi"], P_cox_eTTP=ce["p"],
                N_OS=res["N_OS"], OS_events=res["OS_events"], P_OS_AT=lr["AT_Score"], P_OS_eTTP=lr["Clinical_Preds"],
                P_OS_baselinePSA=lr["Baseline_Tumor_Burden"], P_OS_nadir=lr["PSA_Nadir"], P_OS_timeToNadir=lr["Time_to_Nadir"], P_OS_PSAdt=lr["PSA_Doubling_Time_days"])

def main(out=os.path.join(HERE)):
    pub = pd.read_csv(PUB_CSV)
    fits_pub_refit = zp.fit_window("published"); fits_tw = zp.fit_window("treated"); fits_tw_first = zp.fit_window("treated", norm="first", turnover_start=0.3)
    rows, tabs = [], {}
    def add(label, fits, norm="published", mode="published", subset=None, **kw):
        f = fits if subset is None else fits[fits.PatientId.isin(subset)]
        res, T, M = zp.evaluate(f, norm=norm, mode_for_metrics=mode, **kw); rows.append(row(label, res)); tabs[label] = T; return T
    T1 = add("1. Published fits, published window, published filters", pub)
    T2 = add("2. Treated-cycle window (published settings), published filters", fits_tw, mode="treated")
    common = sorted(set(T1.PatientId) & set(T2.PatientId))
    add("3. Published fits, the %d patients common to 1 and 2" % len(common), pub, subset=common)
    add("4. Treated-cycle fits, the same %d patients" % len(common), fits_tw, mode="treated", subset=common)
    below = fits_tw.PatientId[fits_tw.turnover < POLE]
    add("5. Treated-cycle fits, fits above the AT-score pole removed", fits_tw, mode="treated", subset=below)
    add("6. Published fits, the four one-point fits removed", pub, subset=[p for p in pub.PatientId if p not in (1005, 1010, 1017, 1018)])
    add("7. Published fits, no model filters (all with TTP > 0)", pub, require_progression=False, require_r2=False)
    add("8. Treated-cycle fits, no model filters (all with TTP > 0)", fits_tw, mode="treated", require_progression=False, require_r2=False)
    add("9. Treated window + first-measurement normalisation, published filters", fits_tw_first, norm="first", mode="treated")
    S = pd.DataFrame(rows); S.to_csv(os.path.join(out, "zhang_window_effect_scenarios.csv"), index=False)
    # per-patient comparison
    a = pub.set_index("PatientId"); b = fits_tw.set_index("PatientId"); pr = fits_pub_refit.set_index("PatientId")
    P = pd.DataFrame({"n_points_published_window": pr.n_points, "n_points_treated_window": b.n_points,
                      "model_progresses_pub": a.Progression, "model_progresses_treated": b.Progression, "R2_pub": a.RSquared, "R2_treated": b.RSquared,
                      "turnover_pub": a.turnover, "turnover_treated": b.turnover, "K_pub": a.K, "K_treated": b.K})
    t1 = T1.set_index("PatientId"); t2 = T2.set_index("PatientId")
    P["AT_score_pub"] = t1.AT_Score; P["AT_score_treated"] = t2.AT_Score; P["eTTP_pub"] = t1.Clinical_Preds; P["eTTP_treated"] = t2.Clinical_Preds
    P["in_published_N13"] = P.index.isin(t1.index); P["in_treated_set"] = P.index.isin(t2.index)
    P["clinical_TTP"] = pd.concat([t1.Clinical_TTP, t2.Clinical_TTP]).groupby(level=0).first()
    P.reset_index().to_csv(os.path.join(out, "zhang_window_effect_patients.csv"), index=False)
    return S, P, T1, T2

if __name__ == "__main__":
    S, P, T1, T2 = main()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(S.round(3).to_string(index=False))
