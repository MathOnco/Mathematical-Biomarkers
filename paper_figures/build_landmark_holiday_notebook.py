"""Builds and executes SI_Landmark_and_Holiday_Fraction.ipynb in-process (no Jupyter kernel needed).
Run from paper_figures/:  python build_landmark_holiday_notebook.py"""
import io, os, sys, contextlib, traceback, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

C = []   # (kind, content)  content = str, or callable(ns)->str for markdown
def md(x): C.append(("md", x))
def code(x): C.append(("code", x))

md("""# Landmark re-analysis (castrate-sensitive) and drug-holiday fraction vs TTP (all cohorts)

Branch `corrected-fitting-protocol`. Code: `analysis/holiday_fraction_utils.py`, `analysis/landmark_holiday_analysis.py`
(fits and statistics reuse `patients/castrate_sensitive.py`, `analysis/castrate_sensitive_stats.py`,
`patients/first_cycle_fitting.py`). Patient data are not tracked in the public repository.

**Definitions, fixed before any association was computed**
* **First cycle** = first drug-on row -> the row at which treatment restarts (start of cycle 2). Bruchovsky: first `CycleId == 2` row;
  castrate-sensitive: first `Cycle == 2` row; Zhang (no cycle label): first off -> on transition after treatment has started.
  13 of 17 Zhang files begin with pre-treatment, off-drug rows, so for them the cycle starts at the first drug-on row, not the first record.
* **Off (holiday) fraction** = time with no drug / time in window, step-holding the drug state of the preceding row. Windows: first cycle;
  whole record (Bruchovsky, Zhang only); up to TTP (Bruchovsky, Zhang only). The castrate-sensitive workbook contains the first cycle only, so
  for that cohort the 'full treatment history' fraction is not available and equals the first-cycle fraction at most.
* **TTP / event**: Bruchovsky, Zhang as in the SI right-censoring notebooks (TTP = last time PSA < 1.2 x first PSA; event = max PSA > 1.2 x first PSA).
  Castrate-sensitive: TTE and status from `progression_outcomes.tsv` (TSV) and the workbook-reconciled version (see `SI_Castrate_Sensitive_Cohort.ipynb`).
* **Drug state for the castrate-sensitive holiday fraction**: 'union' (any of ADT / abiraterone / ARSI) is the physically meaningful holiday;
  'ADT-only' is the state used for fitting. Both are reported.
* **Assumption that cannot be checked from the files**: the TSV/Summary times share their origin with the sheet's `Days` (first ADT dose).
* All tests are exploratory and not corrected for multiplicity.""")

code('''import os, sys, warnings, numpy as np, pandas as pd, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ROOT = os.path.abspath(".."); os.chdir(ROOT)
for sub in ("utils", "patients", "analysis"): sys.path.insert(0, os.path.join(ROOT, sub))
import castrate_sensitive as cs, castrate_sensitive_stats as st
import holiday_fraction_utils as hf, landmark_holiday_analysis as lh
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
tabs = cs.load_tabs(); summ = cs.load_summary(); outc = cs.load_outcomes()
as_supplied, reconciled, audit = cs.reconcile_outcomes(outc, summ)
OUTS = {"TSV": as_supplied, "Reconciled": reconciled}''')

md("## 1. Castrate-sensitive cohort: time from the end of the first cycle to progression")
code('''B = hf.holiday_table("Bruchovsky"); Z = hf.holiday_table("Zhang")
CSu = hf.cs_table(as_supplied, tabs, "union"); CSa = hf.cs_table(as_supplied, tabs, "ADT")
L = CSu.set_index("PatientId").cycle1_length
tab = pd.DataFrame({"first_cycle_length_d": L, "TTE_TSV": as_supplied.TTE, "event_TSV": as_supplied.Event,
                    "TTE_reconciled": reconciled.TTE, "event_reconciled": reconciled.Event})
tab["TTE_TSV_minus_L"] = tab.TTE_TSV - tab.first_cycle_length_d; tab["TTE_reconciled_minus_L"] = tab.TTE_reconciled - tab.first_cycle_length_d
print(tab.astype(int).to_string()); print("\\nsmallest TTE - L:", int(tab[["TTE_TSV_minus_L","TTE_reconciled_minus_L"]].min().min()), "d (nobody is excluded by the shift)")
FITV = lh.fit_all_variants(tabs)   # 8 outcome-blind fit variants (origin x drug state x priors), ~45 s''')
code('''prim = lh.landmark_primary(FITV, OUTS, L)
cols = ["outcomes","time","set","biomarker","N","events","HR","lo","hi","p_wald","C","p_C_one","p_C_two","rho_T","p_rho"]
print(prim[cols].round(3).to_string(index=False))''')
code('''G = lh.landmark_grid(FITV, OUTS, L)
GS = G.groupby("time").apply(lh.grid_summary).T; print(GS.round(3).to_string())
m = G[G.HR.notna()].pivot_table(index=["origin","drug","priors","outcomes","set","biomarker"], columns="time", values=["C","HR","p_C_one"])
dC = (m["C"]["adjusted"] - m["C"]["raw"]).abs().max(); dH = np.log(m["HR"]["adjusted"] / m["HR"]["raw"]).abs().max()
cross = m[(m["p_C_one"]["raw"] < .05) != (m["p_C_one"]["adjusted"] < .05)]["p_C_one"]
print(f"max |change in C| = {dC:.3f}; max |change in log HR| = {dH:.3f}; cells crossing one-sided P=.05 on shifting: {len(cross)}"); print(cross.round(3))
print("events for 80% power at HR 0.5 / 0.7 per SD:", round(lh.schoenfeld_events(.5),1), round(lh.schoenfeld_events(.7),1))
prim.to_csv("analysis/castrate_sensitive_landmark_primary.csv", index=False); G.to_csv("analysis/castrate_sensitive_landmark_grid.csv", index=False)''')
md(lambda ns: (lambda P, G, GS: f"""**Result.** Subtracting the first-cycle length does not change the conclusion. In the pre-declared primary specification (set C, all 16 patients,
5 events with the TSV outcomes) the AT-score hazard ratio per +1 SD moves from {P.query("outcomes=='TSV' and time=='raw' and set=='C' and biomarker=='AT score'").HR.iloc[0]:.2f} to
{P.query("outcomes=='TSV' and time!='raw' and set=='C' and biomarker=='AT score'").HR.iloc[0]:.2f}
(P = {P.query("outcomes=='TSV' and time=='raw' and set=='C' and biomarker=='AT score'").p_wald.iloc[0]:.2f} -> {P.query("outcomes=='TSV' and time!='raw' and set=='C' and biomarker=='AT score'").p_wald.iloc[0]:.2f}); the eTTP hazard ratio from
{P.query("outcomes=='TSV' and time=='raw' and set=='C' and biomarker=='eTTP'").HR.iloc[0]:.2f} to {P.query("outcomes=='TSV' and time!='raw' and set=='C' and biomarker=='eTTP'").HR.iloc[0]:.2f}.
Across the 96-cell specification grid ({int(GS.loc['estimable','adjusted'])} estimable) no Wald P falls below .05 either before (minimum {GS.loc['min_wald_p','raw']:.3f}) or after
(minimum {GS.loc['min_wald_p','adjusted']:.3f}) the shift; the largest change in Harrell's C in any cell is {ns['dC']:.3f}.
One-sided permutation P < .05 occurs in {int(GS.loc['permC_onesided_p_below_05','raw'])} (raw) and {int(GS.loc['permC_onesided_p_below_05','adjusted'])} (shifted) cells, but almost all are the quasi-separated 3-event
sets A/B (HR ~ 0.03, CI spanning 0-20); in the full cohort (set C) it is {int(GS.loc['setC_permC_p_below_05','raw'])} -> {int(GS.loc['setC_permC_p_below_05','adjusted'])} of {int(GS.loc['setC_cells','raw'])} cells, about what chance gives.
Why the shift changes so little: first-cycle length is unrelated to the AT score (see section 2) and only weakly related to TTE, so removing it re-ranks few event times.
With 5-6 events this cohort cannot show significance for any plausible effect size: ~{ns['lh'].schoenfeld_events(.5):.0f} events would be needed for 80% power at HR 0.5 per SD.""")(ns['prim'], ns['G'], ns['GS']))

md("## 2. Is the first-cycle length coupled to the AT score or the holiday fraction?")
code('''prim_scored = st.score_fits(FITV[("treatment","ADT","Zhang-style")], cost=0.01).set_index("PatientId")
at_rank = pd.Series(st._rank_inf(prim_scored.AT_score.values), index=prim_scored.index)
for lab, s in [("TTE (TSV)", as_supplied.TTE), ("fitted turnover", prim_scored.turnover), ("AT score", at_rank),
               ("first-cycle off fraction, union", CSu.set_index("PatientId").off_cycle1), ("first-cycle off fraction, ADT-only", CSa.set_index("PatientId").off_cycle1)]:
    r, p = spearmanr(L, s.reindex(L.index)); print(f"  L vs {lab:36s} rho={r:+.3f}  P={p:.3f}")''')

md("## 3. TTP vs drug-holiday fraction in the three cohorts")
code('''subB = set(pd.read_csv("analysis/holiday_endpoint_bruchovsky.csv").PatientId)      # the 40 analysed in the AT-score study
CST = {"union": CSu, "ADT-only": CSa}
print("Bruchovsky: first drug-on at t=0 in", int((B.t_first_on == 0).sum()), "/72; CycleId restart == drug-transition restart in", int(np.isclose(B.cycle1_end, B.restart_from_drug).sum()), "/72")
print("Zhang: files starting off drug:", int(Z.starts_off_drug.sum()), "/", len(Z), "; TTP = 0 (excluded):", sorted(Z.PatientId[Z.TTP <= 0]))
print("median off fraction  Bruchovsky first cycle / whole: %.3f / %.3f ;  Zhang: %.3f / %.3f ; castrate-sensitive first cycle (union / ADT-only): %.3f / %.3f" % (
      B.off_cycle1.median(), B.off_whole.median(), Z.off_cycle1.median(), Z.off_whole.median(), CSu.off_cycle1.median(), CSa.off_cycle1.median()))
HA = lh.holiday_associations(B, Z, CST, OUTS, subB)
print(HA.round(3).to_string(index=False))
HA.to_csv("analysis/holiday_fraction_associations.csv", index=False)
pd.concat([B.assign(cohort="Bruchovsky"), Z.assign(cohort="Zhang"), CSu.assign(cohort="Castrate-sensitive (union)"), CSa.assign(cohort="Castrate-sensitive (ADT-only)")]).to_csv("analysis/holiday_fraction_per_patient.csv", index=False)''')
code('''CP = lh.coupling_table(B, Z, CST, OUTS, subB); print(CP.round(3).to_string(index=False)); CP.to_csv("analysis/holiday_fraction_coupling.csv", index=False)''')

md("## 4. Sensitivity: where the Zhang AT-score fits are made (published window vs first treated cycle)")
code('''ZFITS, ZSENS = lh.zhang_window_sensitivity(Z)
print("patients with <3 points in the fit window:", ZFITS.groupby(["window","priors"]).n_pts.apply(lambda s: int((s < 3).sum())).to_dict())
print(ZSENS.pivot_table(index=["priors","subset","endpoint"], columns="window", values=["N","rho","p"]).round(3).to_string())
ZFITS.to_csv("analysis/zhang_window_fits.csv", index=False); ZSENS.to_csv("analysis/zhang_window_sensitivity.csv", index=False)''')

md("## 5. Figure")
code('''exec(open("paper_figures/landmark_holiday_figure.py").read())''')
md("![landmark and holiday fraction](../analysis/landmark_holiday_fraction.png)")

def _summary(ns):
    HA, CP, ZS = ns["HA"], ns["CP"], ns["ZSENS"]
    def h(ds, sub, frac, out="TTP"):
        r = HA[(HA.dataset == ds) & (HA.subset.str.contains(sub, regex=False)) & (HA.fraction == frac) & (HA.outcome == out)].iloc[0]
        return r
    def fmt(r): return f"rho = {r.rho:+.2f} (P = {r.p_rho:.2f}), Cox HR {r.HR:.2f} per SD ({r.lo:.2f}-{r.hi:.2f}), C = {r.C:.2f}, N = {int(r.N)}, events = {int(r.events)}"
    b1, b2 = h("Bruchovsky", "all 72", "first cycle"), h("Bruchovsky", "all 72", "whole record")
    z1, z2 = h("Zhang", "all with", "first cycle"), h("Zhang", "all with", "whole record")
    c1 = h("Castrate-sensitive", "union; outcomes=TSV", "first cycle"); c2 = h("Castrate-sensitive", "ADT-only; outcomes=TSV", "first cycle")
    cp = CP.set_index("set")
    def zs(win, sub, end, pri="Zhang"):
        r = ZS[(ZS.window == win) & (ZS.priors.str.startswith(pri)) & (ZS.subset == sub) & (ZS.endpoint == end)].iloc[0]; return f"rho = {r.rho:+.2f} (P = {r.p:.3f}, N = {int(r.N)})"
    return f"""## Summary

**Holiday fraction vs TTP (first cycle and whole record; TTP as defined above).**
* Bruchovsky (72 patients): first-cycle off fraction {fmt(b1)}; whole record {fmt(b2)}.
* Zhang (15 patients with TTP > 0): first cycle {fmt(z1)}; whole record {fmt(z2)}.
* Castrate-sensitive (16; first cycle only): union {fmt(c1)}; ADT-only {fmt(c2)} (TSV outcomes; reconciled outcomes in the table above).
* None of the Cox or concordance tests reaches P < .05 for the raw TTP. The only nominal Spearman P < .05 are the castrate-sensitive ADT-only cells, and these are explained by cycle length:
  the first-cycle off fraction is almost a monotone function of first-cycle length there (rho = {cp.loc['Castrate-sensitive union / TSV','rho_off_vs_L']:+.2f} union, {cp.loc['Castrate-sensitive ADT-only / TSV','rho_off_vs_L']:+.2f} ADT-only), and the partial correlation given cycle length is
  {cp.loc['Castrate-sensitive union / TSV','partial_rho_off_vs_TTP_given_L']:+.2f} (union) and {cp.loc['Castrate-sensitive ADT-only / TSV','partial_rho_off_vs_TTP_given_L']:+.2f} (ADT-only), P = {cp.loc['Castrate-sensitive union / TSV','partial_p']:.2f} and {cp.loc['Castrate-sensitive ADT-only / TSV','partial_p']:.2f}. The same is true in Bruchovsky (rho off vs L = {cp.loc['Bruchovsky all 72','rho_off_vs_L']:+.2f}).
* Subtracting the end-of-cycle time from TTP creates an artefactual negative association in Bruchovsky (rho {h('Bruchovsky','all 72','first cycle','TTP - end of cycle 1').rho:+.2f}) because the off fraction rises with cycle length; do not interpret those rows as biology.
  In Zhang the whole-record fraction against TTP minus end of cycle 1 gives rho = {h('Zhang','all with','whole record','TTP - end of cycle 1').rho:+.2f} (P = {h('Zhang','all with','whole record','TTP - end of cycle 1').p_rho:.3f}, N = {int(h('Zhang','all with','whole record','TTP - end of cycle 1').N)} because three patients have TTP before the cycle ends);
  one nominal result among 26 tests, in a cohort whose first-record PSA is pre-treatment for 13/17 patients, so it is not treated as evidence.
* Whole-record and up-to-TTP fractions share their window with TTP (more follow-up -> more cycles -> a more stable fraction), so a positive association there is partly built in.

**Zhang fit window (new finding).** 13 of 17 Zhang files begin with pre-treatment, off-drug rows. `TruncateDataframe` cuts at the first off -> on switch, which for those patients is the start of abiraterone, so the 'first cycle' that was
fitted is the pre-treatment segment ({int((ns['ZFITS'].query("window=='published' and priors.str.startswith('Zhang')").n_pts < 3).sum())} patients with < 3 points, i.e. the 'sparsity' seen earlier). Re-fitting on the first treated cycle (all 17 have >= 3 points; Zhang-style priors, first-measurement normalisation, kappa = 0.2) gives:
* AT score vs TTP: published truncation window (otherwise harmonised settings) {zs('published','all with TTP>0','TTP')} -> treated cycle {zs('treated','all with TTP>0','TTP')}; below the pole {zs('treated','below AT-score pole','TTP')}.
* AT score vs whole-record off fraction: published truncation window {zs('published','all with TTP>0','whole-record off fraction')} -> treated cycle {zs('treated','all with TTP>0','whole-record off fraction')}; below the pole {zs('treated','below AT-score pole','whole-record off fraction')}.
* With Bruchovsky-style priors the treated-window AT score vs whole-record off fraction (below the pole) is {zs('treated','below AT-score pole','whole-record off fraction','Bruchovsky')}; AT score vs TTP is not significant under either prior.

Interpretation: on a correct treated-cycle window the Zhang TTP association with the AT score is not supported, whereas the AT score does track the fraction of the history spent off drug in both Bruchovsky and Zhang.
This is a post hoc check on 15 patients (one main specification plus one prior sensitivity), prompted by the discovery of the window problem, so it is a hypothesis to confirm, not a result. The Bruchovsky holiday-fraction association comes from the earlier analysis in this branch (`analysis/holiday_endpoint.py`). The first-cycle off fraction column in `zhang_window_sensitivity.csv` shares the fit window and is therefore not an independent endpoint."""

md(_summary)

HERE = os.path.dirname(os.path.abspath(__file__))

def build(path=os.path.join(HERE, "SI_Landmark_and_Holiday_Fraction.ipynb")):
    ns = {"__name__": "__main__"}; nb = new_notebook(); nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    n = 0
    for kind, content in C:
        if kind == "md":
            txt = content(ns) if callable(content) else content
            nb.cells.append(new_markdown_cell(txt))
        else:
            n += 1; buf = io.StringIO(); cell = new_code_cell(content)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try: exec(compile(content, f"<cell {n}>", "exec"), ns)
                except Exception: buf.write(traceback.format_exc())
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue())]; cell.execution_count = n
            nb.cells.append(cell)
            if "Traceback" in buf.getvalue(): print("ERROR in cell", n); print(buf.getvalue()); break
    nbformat.write(nb, path); print("wrote", path)

if __name__ == "__main__":
    build()
