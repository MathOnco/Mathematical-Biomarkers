"""
Scoring and small-sample statistics for the castrate-sensitive cohort analysis.

Biomarkers (first-cycle fits only; no outcome-window quantities):
  AT score = calc_delta_benefit(kappa = 0.2, **params)         constant kappa (CORRECTIONS.md item 6)
  eTTP     = predict_atx_ttp(tau = 30, **params)               repo function; AT50-like, with overshoot correction

The published eTTP (predict_clinical_ttp) needs PSA up to clinical TTP and the raw record, which this
cohort does not provide (first cycle only), so it cannot be computed here.

Analysis sets (declared before any association was computed):
  A  convention-matched   model predicts progression AND turnover below the AT-score pole AND R2 > 0.5
  B  AT-score domain      turnover below the pole AND R2 > 0.5 (the AT score does not need CT progression)
  C  full cohort          all fitted patients. Where the model predicts the resistant population is held
                          indefinitely (turnover >= pole, or no CT progression for eTTP) the biomarker is
                          +infinity, which is what the formulae give: the doubling time diverges.

Direction convention throughout: a HIGHER score is predicted to give a LONGER time to progression,
so a protective association has HR < 1 and Harrell's C > 0.5.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from myUtils import (calc_delta_benefit, predict_ct_ttp, predict_atx_ttp, test_progression,
                     calc_critical_treatment_threshold, calc_average_overshoot)

KAPPA = 0.2
TAU = 30.0


def pole(cost, kappa=KAPPA):
    return (1.0 - cost) * (1.0 - kappa)


def score_fits(fits, cost):
    """Add AT score / eTTP and the flags that define analysis sets A/B/C."""
    out = fits.copy()
    P = pole(cost)
    at, ettp, prog, above, kap_eff_above = [], [], [], [], []
    for _, r in out.iterrows():
        p = r.to_dict()
        pr = bool(test_progression(p)); ab = bool(p["turnover"] >= P)
        prog.append(pr); above.append(ab)
        # ---- AT score
        if ab:
            at.append(np.inf)
        else:
            try:
                v = calc_delta_benefit(kappa=KAPPA, **p); at.append(v if np.isfinite(v) else np.nan)
            except Exception:
                at.append(np.nan)
        # ---- eTTP (tau = 30)
        if not pr:
            ettp.append(np.inf); kap_eff_above.append(False); continue
        try:
            n_crit = calc_critical_treatment_threshold(tau=TAU, **p)
            ov = calc_average_overshoot(TAU, n_crit, **p)
            ave_n = (n_crit + 1.2 * p["n0"]) / 2 - ov
            kap_eff = ave_n / p["K"]
            if p["turnover"] >= (1 - cost) * (1 - kap_eff):
                ettp.append(np.inf); kap_eff_above.append(True)
            else:
                v = predict_atx_ttp(tau=TAU, **p)
                ettp.append(v if (np.isfinite(v) and v > 0) else np.nan); kap_eff_above.append(False)
        except Exception:
            ettp.append(np.nan); kap_eff_above.append(False)
    out["model_progresses"] = prog
    out["above_pole"] = above
    out["AT_score"] = at
    out["eTTP"] = ettp
    out["R2_ok"] = (out["RSquared"] > 0.5) | out["RSquared"].isna()
    out["set_A"] = out.model_progresses & ~out.above_pole & out.R2_ok & np.isfinite(out.AT_score)
    out["set_B"] = ~out.above_pole & out.R2_ok & np.isfinite(out.AT_score)
    out["set_C"] = True
    return out


# ------------------------------------------------------------------ statistics
def cindex(score, T, E):
    """Harrell's C with 'higher score -> longer time'. Ties in score count 0.5."""
    s = np.asarray(score, float); T = np.asarray(T, float); E = np.asarray(E, int)
    n = len(s); conc = 0.0; comp = 0
    for i in range(n):
        if not E[i]:
            continue
        for j in range(n):
            if T[j] > T[i]:
                comp += 1
                conc += 1.0 if s[j] > s[i] else (0.5 if s[j] == s[i] else 0.0)
    return (conc / comp if comp else np.nan), comp


def _rank_inf(x):
    """Replace +inf by a value above the finite maximum so ranking/Cox are defined."""
    x = np.asarray(x, float).copy()
    fin = np.isfinite(x)
    if (~fin & ~np.isnan(x)).any():
        x[np.isposinf(x)] = (np.nanmax(x[fin]) if fin.any() else 0) + 1.0
    return x


def perm_c(score, T, E, n=10000, seed=0):
    """Permutation p-values for Harrell's C (one-sided for the expected direction, and two-sided)."""
    rng = np.random.default_rng(seed)
    s = _rank_inf(score); obs, comp = cindex(s, T, E)
    if comp == 0 or np.isnan(obs):
        return dict(C=np.nan, n_comp=comp, p_one=np.nan, p_two=np.nan)
    ge = 0; ext = 0
    for _ in range(n):
        c, _ = cindex(rng.permutation(s), T, E)
        ge += c >= obs - 1e-12; ext += abs(c - 0.5) >= abs(obs - 0.5) - 1e-12
    return dict(C=obs, n_comp=comp, p_one=(ge + 1) / (n + 1), p_two=(ext + 1) / (n + 1))


def cox_uni(score, T, E, use_rank=False):
    """Univariable Cox on a standardised score. HR is per +1 SD; HR < 1 = protective (higher -> longer)."""
    from lifelines import CoxPHFitter
    s = np.asarray(score, float)
    if use_rank:
        from scipy.stats import rankdata
        s = rankdata(_rank_inf(s))
    else:
        s = _rank_inf(s)
    if np.nanstd(s) == 0 or int(np.sum(E)) < 2:
        return dict(HR=np.nan, lo=np.nan, hi=np.nan, p_wald=np.nan, p_lr=np.nan, note="degenerate")
    z = (s - np.mean(s)) / np.std(s, ddof=1)
    df = pd.DataFrame({"z": z, "T": T, "E": E})
    for pen in (0.0, 0.05):
        try:
            cph = CoxPHFitter(penalizer=pen).fit(df, duration_col="T", event_col="E")
            sm = cph.summary.loc["z"]
            lr = cph.log_likelihood_ratio_test().p_value
            return dict(HR=sm["exp(coef)"], lo=sm["exp(coef) lower 95%"], hi=sm["exp(coef) upper 95%"],
                        p_wald=sm["p"], p_lr=lr, note="" if pen == 0 else "penalised (0.05)")
        except Exception as e:
            err = str(e)[:40]
    return dict(HR=np.nan, lo=np.nan, hi=np.nan, p_wald=np.nan, p_lr=np.nan, note="fit failed: " + err)


def spearman_with_time(score, T):
    from scipy.stats import spearmanr
    s = _rank_inf(score); ok = np.isfinite(s)
    r, p = spearmanr(s[ok], np.asarray(T)[ok]); return r, p


def auc_nonprogressor(score, E, n=10000, seed=0):
    """P(score higher in non-progressors than progressors); exact-ish permutation p (two-sided)."""
    s = _rank_inf(score); E = np.asarray(E, int)
    a = s[E == 0]; b = s[E == 1]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    def auc(x, y):
        return (np.sum(x[:, None] > y[None, :]) + 0.5 * np.sum(x[:, None] == y[None, :])) / (len(x) * len(y))
    obs = auc(a, b); rng = np.random.default_rng(seed); ext = 0
    for _ in range(n):
        pe = rng.permutation(E); u = s[pe == 0]; v = s[pe == 1]
        ext += abs(auc(u, v) - 0.5) >= abs(obs - 0.5) - 1e-12
    return obs, (ext + 1) / (n + 1)


def logrank_median_split(score, T, E):
    from lifelines.statistics import logrank_test
    s = _rank_inf(score); med = np.median(s)
    hi = s > med; lo = ~hi
    if hi.sum() < 1 or lo.sum() < 1 or int(np.sum(E)) < 1:
        return dict(p=np.nan, n_hi=int(hi.sum()), n_lo=int(lo.sum()), ev_hi=int(np.sum(np.asarray(E)[hi])),
                    ev_lo=int(np.sum(np.asarray(E)[lo])))
    r = logrank_test(np.asarray(T)[hi], np.asarray(T)[lo], np.asarray(E)[hi], np.asarray(E)[lo])
    return dict(p=r.p_value, n_hi=int(hi.sum()), n_lo=int(lo.sum()),
                ev_hi=int(np.sum(np.asarray(E)[hi])), ev_lo=int(np.sum(np.asarray(E)[lo])))


# ------------------------------------------------------------------ empirical comparators
def first_cycle_metrics(rec):
    """Phenomenological first-cycle PSA metrics, defined as in the other cohorts (raw PSA)."""
    c1 = rec[(rec.CycleId <= 1) & rec.PSA_raw.notna()].sort_values("Time")
    out = dict(Baseline_PSA=float(c1.PSA_raw.iloc[0]), PSA_Nadir=np.nan, Time_to_Nadir=np.nan,
               PSA_Doubling_Time_days=np.nan, PSA_Velocity_per_day=np.nan)
    i = int(np.argmin(c1.PSA_raw.values))
    out["PSA_Nadir"] = float(c1.PSA_raw.iloc[i]); out["Time_to_Nadir"] = float(c1.Time.iloc[i] - c1.Time.iloc[0])
    reg = c1.iloc[i:]
    off = reg[reg.DrugConcentration <= 0]
    if len(off) >= 2:
        reg = off
    if len(reg) >= 2:
        x = reg.Time.values.astype(float); y = reg.PSA_raw.values.astype(float)
        if np.ptp(x) > 0:
            out["PSA_Velocity_per_day"] = float(np.polyfit(x, y, 1)[0])
            pos = y > 0
            if pos.sum() >= 2 and np.ptp(x[pos]) > 0:
                sl = np.polyfit(x[pos], np.log(y[pos]), 1)[0]
                if sl > 0:
                    out["PSA_Doubling_Time_days"] = float(np.log(2) / sl)
    return out


# ------------------------------------------------------------------ one analysis cell
def analyse(scored, outs, score_col, setname, perm_n=5000, seed=0):
    """Cox + concordance + rank statistics for one (fit, outcome version, set, biomarker) cell.
    Sets A/B use the standardised score; set C (contains +inf) uses the rank-normalised score."""
    sub = scored[scored["set_" + setname]].join(outs, how="inner")
    sub = sub[~sub[score_col].isna()]
    s = sub[score_col].values; T = sub["TTE"].values; E = sub["Event"].values.astype(int)
    if int(E.sum()) < 1 or len(sub) < 6:
        return dict(N=len(sub), events=int(E.sum()))
    c = perm_c(s, T, E, n=perm_n, seed=seed); cx = cox_uni(s, T, E, use_rank=(setname == "C"))
    rho, rp = spearman_with_time(s, T); auc, ap = auc_nonprogressor(s, E, n=perm_n, seed=seed)
    return dict(N=len(sub), events=int(E.sum()), HR=cx["HR"], lo=cx["lo"], hi=cx["hi"], p_wald=cx["p_wald"],
                p_lr=cx["p_lr"], C=c["C"], n_comp=c["n_comp"], p_C_one=c["p_one"], p_C_two=c["p_two"],
                rho_T=rho, p_rho=rp, AUC_nonprog=auc, p_auc=ap, note=cx["note"])
