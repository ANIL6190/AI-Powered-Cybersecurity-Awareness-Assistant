"""
Phase 1 `eda` stage: reproducible exploratory analysis of the raw and prepared data.

Writes figures to reports/phase1/eda/ and a machine-readable summary to
reports/phase1/eda_summary.json. notebooks/01_eda.ipynb presents these outputs
with the written narrative.
"""

import argparse
import os
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import ensure_dir, get_logger, load_config, read_json, write_json  # noqa: E402
from src.utils.urls import canonicalize_url, extract_host  # noqa: E402

logger = get_logger("eda")

COLORS = {"legitimate": "#2b7bba", "phishing": "#d1495b"}


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def profile_frame(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }


def format_artefacts(urls: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """Rates of collection-format features per class, measured on the RAW URL string."""
    u = urls.astype(str)
    f = pd.DataFrame({
        "label": labels,
        "starts_https": u.str.startswith("https://"),
        "starts_http": u.str.startswith("http://"),
        "has_scheme": u.str.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://"),
        "has_www": u.str.contains(r"^(?:[a-zA-Z]+://)?www\d*\.", regex=True),
        "trailing_slash": u.str.endswith("/"),
        "has_path": u.str.replace(r"^[a-zA-Z]+://", "", regex=True).str.rstrip("/").str.contains(r"[/?#]", regex=True),
    })
    return f.groupby("label").mean().round(4)


def run(config_path: str = "params.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    raw_cfg, out = cfg["raw_data"], cfg["output"]
    eda_dir = ensure_dir(os.path.join(out["reports_dir"], "eda"))
    cutoff = pd.Timestamp(cfg["temporal"]["cutoff_date"])
    summary: Dict[str, Any] = {}

    # ---------------------------------------------------------------- raw data
    phi_raw = pd.read_csv(raw_cfg["phiusiil_path"], encoding="utf-8-sig", low_memory=False)
    pvn_raw = pd.read_csv(raw_cfg["phishvn_path"], dtype=str, keep_default_na=False)
    summary["phiusiil_raw"] = profile_frame(phi_raw[["URL", "Domain", "TLD", "label"]])
    summary["phiusiil_raw"]["n_columns_total"] = int(phi_raw.shape[1])
    summary["phishvn_raw"] = profile_frame(pvn_raw.replace("", np.nan))
    summary["phishvn_raw"]["n_columns_total"] = int(pvn_raw.shape[1])

    phi_lab = phi_raw["label"].map({1: "legitimate", 0: "phishing"})
    pvn_lab = pvn_raw["label"].map({"phishing": "phishing", "benign": "legitimate"})
    summary["phiusiil_class_counts"] = phi_lab.value_counts().to_dict()
    summary["phishvn_source_tier_label_counts"] = (
        pvn_raw.assign(label=pvn_lab).groupby(["source", "tier", "label"]).size()
        .reset_index(name="n").to_dict(orient="records"))
    summary["phiusiil_duplicate_urls"] = int(phi_raw["URL"].duplicated().sum())
    summary["phishvn_duplicate_urls"] = int(pvn_raw["url"].duplicated().sum())

    # Figure 1: class balance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    phi_lab.value_counts().reindex(["legitimate", "phishing"]).plot.bar(
        ax=axes[0], color=[COLORS["legitimate"], COLORS["phishing"]])
    axes[0].set_title("PhiUSIIL — class counts")
    tab = pvn_raw.assign(label=pvn_lab).groupby(["source", "label"]).size().unstack(fill_value=0)
    tab.reindex(columns=["legitimate", "phishing"], fill_value=0).plot.barh(
        ax=axes[1], stacked=True, color=[COLORS["legitimate"], COLORS["phishing"]])
    axes[1].set_title("PhishVN — rows by source and label (all tiers)")
    _save(fig, os.path.join(eda_dir, "01_class_balance.png"))

    # Figure 2 + table: collection-format artefacts (the shortcut audit)
    art_phi = format_artefacts(phi_raw["URL"], phi_lab)
    art_pvn = format_artefacts(pvn_raw["url"], pvn_lab)
    summary["format_artefacts_raw_phiusiil"] = art_phi.to_dict(orient="index")
    summary["format_artefacts_raw_phishvn"] = art_pvn.to_dict(orient="index")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, art) in zip(axes, [("PhiUSIIL (raw URL)", art_phi), ("PhishVN (raw URL)", art_pvn)]):
        art.T.reindex(columns=["legitimate", "phishing"]).plot.bar(
            ax=ax, color=[COLORS["legitimate"], COLORS["phishing"]])
        ax.set_title(name)
        ax.set_ylabel("share of rows")
        ax.set_ylim(0, 1.05)
    _save(fig, os.path.join(eda_dir, "02_format_artefacts.png"))

    # ------------------------------------------------------------ prepared data
    splits = {s: pd.read_csv(os.path.join(out["processed_dir"], f"{s}.csv"), keep_default_na=False,
                             dtype={"event_date": str})
              for s in ("train", "val", "test", "shift_test", "temporal_test")}
    pool = pd.concat([splits[s] for s in ("train", "val", "test")], ignore_index=True)
    pool["label_name"] = pool["label"].map({1: "phishing", 0: "legitimate"})

    summary["prepared_rows"] = {s: int(len(d)) for s, d in splits.items()}
    summary["prepared_phishing_rate"] = {s: round(float(d["label"].mean()), 4) for s, d in splits.items()}
    summary["has_path_rate_after_canonicalisation"] = pool.groupby("label_name")["has_path"].mean().round(4).to_dict()

    # Figure 3: URL length distributions (model input)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    shift = splits["shift_test"].assign(label_name=lambda d: d["label"].map({1: "phishing", 0: "legitimate"}))
    for ax, (name, d) in zip(axes, [("PhiUSIIL pool", pool), ("PhishVN shift_test", shift)]):
        for lab in ("legitimate", "phishing"):
            lens = d.loc[d["label_name"] == lab, "model_text"].str.len().clip(upper=150)
            ax.hist(lens, bins=60, alpha=0.6, label=lab, color=COLORS[lab], density=True)
        ax.set_title(f"{name} — canonical URL length (clipped at 150)")
        ax.legend()
    _save(fig, os.path.join(eda_dir, "03_url_length.png"))
    summary["model_text_length"] = {
        name: d.groupby("label_name")["model_text"].apply(lambda s: s.str.len().describe().round(2).to_dict()).unstack().to_dict(orient="index")
        for name, d in [("phiusiil_pool", pool), ("phishvn_shift", shift)]
    }

    # Figure 4: top TLDs by class
    def tld(s: pd.Series) -> pd.Series:
        return s.map(lambda u: extract_host(u).rsplit(".", 1)[-1] if "." in extract_host(u) else "(none)")

    pool["tld"] = tld(pool["model_text"])
    shift["tld"] = tld(shift["model_text"])
    top_tld = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, d) in zip(axes, [("PhiUSIIL pool", pool), ("PhishVN shift_test", shift)]):
        t = d.groupby(["tld", "label_name"]).size().unstack(fill_value=0)
        t = t.loc[t.sum(axis=1).sort_values(ascending=False).index[:15]]
        t.reindex(columns=["legitimate", "phishing"], fill_value=0).plot.bar(
            ax=ax, stacked=True, color=[COLORS["legitimate"], COLORS["phishing"]])
        ax.set_title(f"{name} — top 15 TLDs")
        top_tld[name] = t.to_dict(orient="index")
    _save(fig, os.path.join(eda_dir, "04_top_tlds.png"))
    summary["top_tlds"] = top_tld
    vn = shift["tld"] == "vn"
    summary["phishvn_shift_p_legitimate_given_vn_tld"] = round(float((shift.loc[vn, "label"] == 0).mean()), 4) if vn.any() else None

    # Figure 5: registrable-domain group sizes
    g = pool.groupby("registrable_domain").agg(n=("label", "size"), phishing_rate=("label", "mean"))
    summary["domain_groups"] = {
        "n_groups": int(len(g)),
        "singleton_share": round(float((g["n"] == 1).mean()), 4),
        "max_group_size": int(g["n"].max()),
        "groups_over_100_rows": int((g["n"] > 100).sum()),
        "largest_groups": g.sort_values("n", ascending=False).head(10)["n"].to_dict(),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(g["n"]), bins=40, color="#555")
    ax.set_xlabel("log10(rows per registrable domain)")
    ax.set_ylabel("number of domains")
    ax.set_title("PhiUSIIL pool — rows per registrable domain")
    _save(fig, os.path.join(eda_dir, "05_domain_group_sizes.png"))

    # Figure 6: PhishVN timeline of source-attested dates vs cutoff
    attested = cfg["phishvn"]["attested_date_sources"]
    dated = pvn_raw[pvn_raw["source"].isin(attested) & pvn_raw["tier"].isin(cfg["phishvn"]["tiers"])].copy()
    dated["date"] = pd.to_datetime(dated[cfg["phishvn"]["timestamp_column"]], format=cfg["phishvn"]["timestamp_format"], errors="coerce")
    dated = dated[dated["date"].notna()]
    monthly = dated.groupby([dated["date"].dt.to_period("M"), "source"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    monthly.index = monthly.index.to_timestamp()
    for src in monthly.columns:
        ax.plot(monthly.index, monthly[src], label=src, lw=1.3)
    ax.axvline(cutoff, color="black", ls="--", lw=1)
    ax.text(cutoff, ax.get_ylim()[1] * 0.92, f" split date {cutoff.date()}", fontsize=9)
    ax.set_title("PhishVN — source-attested event dates per month (gold/silver)")
    ax.set_ylabel("rows")
    ax.legend()
    _save(fig, os.path.join(eda_dir, "06_phishvn_timeline.png"))
    summary["phishvn_attested_dates"] = {
        src: {"min": str(d["date"].min().date()), "max": str(d["date"].max().date()), "n": int(len(d)),
              "after_cutoff": int((d["date"] > cutoff).sum())}
        for src, d in dated.groupby("source")
    }
    other_dated = pvn_raw[~pvn_raw["source"].isin(attested) & (pvn_raw[cfg["phishvn"]["timestamp_column"]].str.strip() != "")]
    summary["phishvn_dates_not_used"] = other_dated.groupby(["source", "tier"]).size().reset_index(name="n").to_dict(orient="records")

    # Figure 7: PhishVN abuse types of the verified phishing rows (what the positive label means)
    abuse_path = os.path.join(raw_cfg["phishvn_dir"], "abuse_type.csv")
    if os.path.exists(abuse_path):
        abuse = pd.read_csv(abuse_path, dtype=str, keep_default_na=False)
        verified_phish = pvn_raw[(pvn_raw["label"] == "phishing") & pvn_raw["tier"].isin(cfg["phishvn"]["tiers"])]
        type_col = [c for c in abuse.columns if c != "url"][0] if "abuse_type" not in abuse.columns else "abuse_type"
        joined = verified_phish.merge(abuse[["url", type_col]], on="url", how="left")
        counts = joined[type_col].replace("", "unknown").fillna("unmatched").value_counts()
        summary["phishvn_verified_phishing_abuse_types"] = counts.to_dict()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        counts.plot.barh(ax=ax, color=COLORS["phishing"])
        ax.set_title("PhishVN verified phishing — abuse_type")
        _save(fig, os.path.join(eda_dir, "07_phishvn_abuse_types.png"))

    # Figure 8: scenario distribution of PhishVN verified phishing
    scen = pvn_raw[(pvn_raw["label"] == "phishing") & pvn_raw["tier"].isin(cfg["phishvn"]["tiers"])]["scenario"].value_counts()
    summary["phishvn_verified_phishing_scenarios"] = scen.to_dict()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    scen.plot.barh(ax=ax, color=COLORS["phishing"])
    ax.set_title("PhishVN verified phishing — impersonation scenario")
    _save(fig, os.path.join(eda_dir, "08_phishvn_scenarios.png"))

    # Pipeline bookkeeping from the prepare and validate stages
    prep_path = os.path.join(out["reports_dir"], "prepare_summary.json")
    if os.path.exists(prep_path):
        prep = read_json(prep_path)
        summary["cleaning"] = {k: prep[k] for k in ("phiusiil", "phishvn") if k in prep}

    # Canonicalisation examples
    examples = phi_raw.groupby(phi_lab)["URL"].apply(lambda s: s.sample(3, random_state=cfg["base"]["random_seed"]).tolist()).to_dict()
    summary["canonicalisation_examples"] = {
        lab: [{"raw": u, "canonical": canonicalize_url(u)} for u in urls] for lab, urls in examples.items()
    }

    write_json(summary, os.path.join(out["reports_dir"], "eda_summary.json"))
    logger.info("EDA written to %s", eda_dir)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 EDA stage")
    parser.add_argument("--config", default="params.yaml")
    run(parser.parse_args().config)
