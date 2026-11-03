# executed inside SI_Landmark_and_Holiday_Fraction.ipynb (uses G, B, Z, CSu, CP, ZSENS from the notebook namespace)
fcol = {"A": "#4C72B0", "B": "#55A868", "C": "#C44E52"}
fig, axs = plt.subplots(1, 4, figsize=(21, 4.9))
ax = axs[0]; est = G[G.HR.notna()]
mm = est.pivot_table(index=["origin", "drug", "priors", "outcomes", "set", "biomarker"], columns="time", values="C").reset_index()
for s in "ABC":
    d = mm[mm.set == s]; ax.scatter(d.raw, d.adjusted, s=24, alpha=.75, color=fcol[s], label=f"set {s}")
ax.plot([.3, 1], [.3, 1], color="grey", lw=1, ls="--"); ax.axhline(.5, color="lightgrey", lw=.8); ax.axvline(.5, color="lightgrey", lw=.8)
ax.set_xlabel("Harrell C, raw time to progression"); ax.set_ylabel("Harrell C, time from end of cycle 1")
ax.set_title("A  Castrate-sensitive: 92 specifications", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax = axs[1]
for nm, d, c in (("Bruchovsky", B, "#4C72B0"), ("Zhang", Z, "#DD8452"), ("Castrate-sens. (union)", CSu, "#C44E52")):
    r = spearmanr(d.cycle1_length, d.off_cycle1)[0]; ax.scatter(d.cycle1_length, d.off_cycle1, s=22, alpha=.8, color=c, label=f"{nm}  rho={r:+.2f}")
ax.set_xscale("log"); ax.set_xlabel("First-cycle length (days, log)"); ax.set_ylabel("Fraction of first cycle off drug")
ax.set_title("B  Off fraction vs cycle length", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axs[2]; cp = CP.copy(); lab = cp.set.str.replace("Castrate-sensitive ", "CS ").str.replace(" / Reconciled", " / recon."); y = np.arange(len(cp))[::-1]
ax.scatter(cp.rho_off_vs_TTP, y, color="#4C72B0", label="Spearman rho with TTP", zorder=3)
ax.scatter(cp.partial_rho_off_vs_TTP_given_L, y, color="#C44E52", marker="s", label="partial rho given cycle length", zorder=3)
for yi, a, b in zip(y, cp.rho_off_vs_TTP, cp.partial_rho_off_vs_TTP_given_L): ax.plot([a, b], [yi, yi], color="lightgrey", zorder=1)
ax.axvline(0, color="grey", lw=.8); ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8); ax.set_xlim(-.15, .7); ax.set_ylim(-1.6, len(cp) - .4)
ax.set_xlabel("Correlation of first-cycle off fraction with TTP"); ax.set_title("C  Holiday fraction vs TTP", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axs[3]
sub = ZSENS[(ZSENS.priors.str.startswith("Zhang")) & (ZSENS.subset == "all with TTP>0") & (ZSENS.endpoint.isin(["TTP", "whole-record off fraction"]))]
xs = np.arange(2); wd = .35
for i, (mode, c) in enumerate((("published", "#9a9a9a"), ("treated", "#C44E52"))):
    v = [sub[(sub.window == mode) & (sub.endpoint == e)].rho.iloc[0] for e in ("TTP", "whole-record off fraction")]
    pv = [sub[(sub.window == mode) & (sub.endpoint == e)].p.iloc[0] for e in ("TTP", "whole-record off fraction")]
    ax.bar(xs + (i - .5) * wd, v, wd, color=c, label={"published": "published window (pre-treatment rows in 13/17)", "treated": "first treated cycle"}[mode])
    for xx, vv, pp in zip(xs + (i - .5) * wd, v, pv): ax.text(xx, vv + .02, f"P={pp:.2f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(["TTP", "whole-record\noff fraction"]); ax.axhline(0, color="grey", lw=.8); ax.set_ylim(-.05, .75)
ax.set_ylabel("Spearman rho with AT score (Zhang, N=15)"); ax.set_title("D  Zhang: effect of the fit window", loc="left", fontsize=10); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("analysis/landmark_holiday_fraction.png", dpi=200); fig.savefig("analysis/landmark_holiday_fraction.pdf"); plt.close(fig)
