"""
(a) Castrate-sensitive cohort: re-test AT score / eTTP after subtracting the first-cycle length from the
    progression times (time from the END of the first cycle to progression).
(b) TTP vs drug-holiday fraction (first cycle / whole record / up to TTP) in Bruchovsky, Zhang, castrate-sensitive.

Definitions of first cycle, holiday fraction and TTP: see holiday_fraction_utils.py.
Assumption (cannot be verified from the files): progression_outcomes.tsv times share the origin of the sheet's
`Days` column (first ADT dose).  All TTE exceed the first-cycle length (smallest gap 169 d), so no one is dropped.
"""
import itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr
from lifelines import CoxPHFitter
import castrate_sensitive as cs, castrate_sensitive_stats as st
import holiday_fraction_utils as hf


# ---------------------------------------------------------------- (a) landmark re-analysis
def fit_all_variants(tabs):
    out = {}
    for origin, drug, priors in itertools.product(["treatment", "first_psa"], ["ADT", "union"], ["Zhang-style", "Bruchovsky-style"]):
        rows = []
        for pid, t in tabs.items():
            rec = cs.build_record(t, origin=origin, drug=drug)
            rows.append(dict(PatientId=pid, origin=origin, drug=drug, priors=priors, **cs.fit_record(rec, priors)))
        out[(origin, drug, priors)] = pd.DataFrame(rows)
    return out


def shift_outcomes(outs, L):
    o = outs.copy(); o["TTE"] = o["TTE"] - L.reindex(o.index); return o


def landmark_primary(fitv, outs_by_version, L, perm_n=5000):
    prim = st.score_fits(fitv[("treatment", "ADT", "Zhang-style")], cost=0.01).set_index("PatientId")
    rows = []
    for ov, outs in outs_by_version.items():
        for tv, o in (("raw", outs), ("minus cycle-1 length", shift_outcomes(outs, L))):
            for sname in "ABC":
                for col, lab in (("AT_score", "AT score"), ("eTTP", "eTTP")):
                    rows.append(dict(outcomes=ov, time=tv, set=sname, biomarker=lab, **st.analyse(prim, o, col, sname, perm_n=perm_n)))
    return pd.DataFrame(rows)


def landmark_grid(fitv, outs_by_version, L, perm_n=3000):
    rows = []
    for key, fits in fitv.items():
        scd = st.score_fits(fits, cost=0.01 if key[2] == "Zhang-style" else 0.1).set_index("PatientId")
        for ov, outs in outs_by_version.items():
            for tv, o in (("raw", outs), ("adjusted", shift_outcomes(outs, L))):
                for sname in "ABC":
                    for col, lab in (("AT_score", "AT score"), ("eTTP", "eTTP")):
                        rows.append(dict(origin=key[0], drug=key[1], priors=key[2], outcomes=ov, time=tv, set=sname,
                                         biomarker=lab, **st.analyse(scd, o, col, sname, perm_n=perm_n)))
    G = pd.DataFrame(rows)
    # 'degenerate' = quasi-separated Cox fit (CI ratio >1e3, or |HR| outside 0.05-20)
    G["degenerate"] = (G.hi / G.lo > 1e3) | (G.HR < 0.05) | (G.HR > 20)
    return G


def grid_summary(G):
    est = G[G.HR.notna()]
    return pd.Series(dict(cells=len(G), estimable=len(est), HR_below_1=int((est.HR < 1).sum()),
                          wald_p_below_05=int((est.p_wald < .05).sum()), min_wald_p=est.p_wald.min(),
                          permC_onesided_p_below_05=int((est.p_C_one < .05).sum()),
                          permC_onesided_p_below_05_nondegenerate=int(((est.p_C_one < .05) & ~est.degenerate).sum()),
                          setC_permC_p_below_05=int(((est.p_C_one < .05) & (est.set == "C")).sum()),
                          setC_cells=int((est.set == "C").sum())))


def schoenfeld_events(hr, alpha=0.05, power=0.80):
    """Events needed to detect a hazard ratio per +1 SD of a standardised covariate (Schoenfeld)."""
    from scipy.stats import norm
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 / np.log(hr) ** 2


# ---------------------------------------------------------------- (b) holiday fraction vs TTP
def assoc(frac, T, E, perm_n=10000):
    frac = np.asarray(frac, float); T = np.asarray(T, float); E = np.asarray(E, int)
    ok = np.isfinite(frac) & np.isfinite(T) & (T > 0); frac, T, E = frac[ok], T[ok], E[ok]
    out = dict(N=len(T), events=int(E.sum()))
    if len(T) < 6 or E.sum() < 1 or np.nanstd(frac) == 0:
        return out
    rho, rp = spearmanr(frac, T); cx = st.cox_uni(frac, T, E); c = st.perm_c(frac, T, E, n=perm_n, seed=0)
    out.update(rho=rho, p_rho=rp, HR=cx["HR"], lo=cx["lo"], hi=cx["hi"], p_wald=cx["p_wald"], C=c["C"], p_C_two=c["p_two"])
    return out


def holiday_associations(B, Z, CS_tables, outs_by_version, subB):
    rows = []
    VAR = {"off_cycle1": "first cycle", "off_whole": "whole record", "off_preTTP": "up to TTP"}
    TT = {"TTP": False, "TTP - end of cycle 1": True}

    def go(df, label, ds, variants):
        for var in variants:
            for tl, adjflag in TT.items():
                T = df["TTP"].values - (df["cycle1_end"].values if adjflag else 0)
                rows.append(dict(dataset=ds, subset=label, fraction=VAR[var], outcome=tl, **assoc(df[var].values, T, df.Event.values)))
    v3 = ["off_cycle1", "off_whole", "off_preTTP"]
    go(B, "all 72", "Bruchovsky", v3)
    go(B[B.PatientId.isin(subB)], "40 analysed in the AT-score study", "Bruchovsky", v3)
    go(Z[Z.TTP > 0], "all with TTP>0 (15)", "Zhang", v3)
    for dname, tab in CS_tables.items():
        for ov, outs in outs_by_version.items():
            d = tab.copy(); d["TTP"] = outs.TTE.reindex(d.PatientId).values; d["Event"] = outs.Event.reindex(d.PatientId).values
            go(d, f"all 16; drug={dname}; outcomes={ov}", "Castrate-sensitive", ["off_cycle1"])
    return pd.DataFrame(rows)


def partial_spearman(x, y, z):
    rx, ry, rz = map(rankdata, (x, y, z))
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    return pearsonr(res(rx, rz), res(ry, rz))


def coupling_table(B, Z, CS_tables, outs_by_version, subB):
    sets = [("Bruchovsky all 72", B), ("Bruchovsky 40 analysed", B[B.PatientId.isin(subB)]), ("Zhang all TTP>0 (15)", Z[Z.TTP > 0])]
    for dname, tab in CS_tables.items():
        for ov, outs in outs_by_version.items():
            d = tab.copy(); d["TTP"] = outs.TTE.reindex(d.PatientId).values; d["Event"] = outs.Event.reindex(d.PatientId).values
            sets.append((f"Castrate-sensitive {dname} / {ov}", d))
    rows = []
    for name, d in sets:
        d = d[d.TTP > 0].dropna(subset=["off_cycle1", "cycle1_length"])
        f, Lc, T, E = d.off_cycle1.values, d.cycle1_length.values, d.TTP.values, d.Event.values.astype(int)
        pr, pp = partial_spearman(f, T, Lc)
        z = lambda v: (v - v.mean()) / v.std(ddof=1)
        m = CoxPHFitter().fit(pd.DataFrame(dict(frac=z(f), L=z(Lc), T=T, E=E)), "T", "E").summary
        rows.append(dict(set=name, N=len(d), events=int(E.sum()), rho_off_vs_L=spearmanr(f, Lc)[0], rho_L_vs_TTP=spearmanr(Lc, T)[0],
                         rho_off_vs_TTP=spearmanr(f, T)[0], partial_rho_off_vs_TTP_given_L=pr, partial_p=pp,
                         cox_adj_HR_off=m.loc["frac", "exp(coef)"], cox_adj_lo=m.loc["frac", "exp(coef) lower 95%"],
                         cox_adj_hi=m.loc["frac", "exp(coef) upper 95%"], cox_adj_p=m.loc["frac", "p"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Zhang fit-window sensitivity
def zhang_window(pid, mode, data_dir=hf.ZHANG_DIR):
    """Fit window for a Zhang patient, PSA normalised by the first row of the window, t = 0 at window start.
    mode='published': TruncateDataframe from the first record (what the notebooks and corrected branch do; for the 13/17
                      files that start off drug this is the PRE-TREATMENT segment, 1-8 rows).
    mode='treated'  : first drug-on row -> last off row before the restart (the first treated cycle)."""
    from first_cycle_fitting import load_zhang
    from myUtils import TruncateDataframe
    d = load_zhang(pid, data_dir).dropna(subset=["PSA_raw"]).reset_index(drop=True)
    if mode == "treated":
        t_on = d.loc[d.DrugConcentration == 1, "Time"].min(); d = d[d.Time >= t_on].reset_index(drop=True)
    w = TruncateDataframe(d, drug_name="DrugConcentration").copy()
    w["PSA"] = w.PSA_raw / w.PSA_raw.iloc[0]; w["Time"] = w.Time - w.Time.iloc[0]
    return w


def zhang_window_fits(mode, fR, cost, Z, kappa=0.2, data_dir=hf.ZHANG_DIR):
    from lmfit import minimize
    import first_cycle_fitting as fcf
    from fittingUtils import residual, compute_r_sq
    from odeModels import LotkaVolterraModel
    from myUtils import calc_delta_benefit, GetPatientIDs
    pole = (1 - cost) * (1 - kappa); rows = []
    for pid in GetPatientIDs(data_dir):
        w = zhang_window(pid, mode, data_dir); params = fcf.build_params(fR, cost, w.PSA.iloc[0])
        model = LotkaVolterraModel(); model.SetParams(**params)
        fit = minimize(residual, params, args=(w, model, "PSA", fcf.SOLVER_KWS), **fcf.OPT_KWS); v = fit.params.valuesdict()
        try: at = calc_delta_benefit(kappa=kappa, **v)
        except Exception: at = np.nan
        rows.append(dict(window=mode, fR=fR, cost=cost, PatientId=pid, n_pts=len(w), turnover=v["turnover"], K=v["K"], AT_score=at,
                         above_pole=v["turnover"] >= pole,
                         R2=compute_r_sq(fit, w, feature="PSA") if len(w) > 2 else np.nan))
    return pd.DataFrame(rows).merge(Z[["PatientId", "TTP", "off_whole", "off_cycle1", "t_first_on"]], on="PatientId")


def zhang_window_sensitivity(Z):
    fits, out = [], []
    for mode in ("published", "treated"):
        for pl, (fR, cost) in {"Zhang-style (fR 1e-2, cost 0.01)": (1e-2, .01), "Bruchovsky-style (fR 1e-5, cost 0.1)": (1e-5, .1)}.items():
            d = zhang_window_fits(mode, fR, cost, Z); d["priors"] = pl; fits.append(d)
            d = d[d.TTP > 0].dropna(subset=["AT_score"])
            for lab, s in (("all with TTP>0", d), ("below AT-score pole", d[~d.above_pole])):
                for end, col in (("TTP", "TTP"), ("whole-record off fraction", "off_whole"), ("first-cycle off fraction (same window as fit)", "off_cycle1")):
                    r = spearmanr(s.AT_score, s[col]); out.append(dict(window=mode, priors=pl, subset=lab, N=len(s), endpoint=end, rho=r[0], p=r[1]))
    return pd.concat(fits, ignore_index=True), pd.DataFrame(out)
