# =====================================================================================
# IAA AI-Assisted Hackathon 2026 - Week 2: Emerging Market
# -------------------------------------------------------------------------------------
# Predicts the 2025 expected ultimate claim cost (risk premium) for every row in
# data/emerging/test_{motor,buildings,contents}.parquet.
#
# Data:   data/emerging    - new market, policy years 2023-2024 (+ 2025 test)
#         data/developping - Part 1 developed market (used as a source of prior knowledge)
#
# What changed vs Part 1 and how this script handles it
#   1. ~20x less data (9k claims vs 163k)       -> much smaller trees, CV by policy,
#                                                  transfer of Part 1 relativities (step 5)
#   2. 2025 book is ~76% new business (broker    -> new_business / tenure kept as features,
#      portfolio transfer) vs 14% in training      NB-only validation score reported
#   3. Different schema (age bands, gender, no   -> unified age_band; developed prior model
#      credit band / telematics / mileage)         uses only features both markets share
#   4. Different costs (BI ~half developed)       -> levels always re-estimated on emerging
#                                                  data; only *shapes* are borrowed
#   5. Only 2 policy years -> no own development -> reporting pattern carried over from the
#      history                                      developed market (checked vs emerging delays)
#   6. Severity trend from 1 year-on-year step   -> credibility blend with developed trend
#   7. Data issues (exposure in days, 1000x SI,   -> cleaned + logged in the QA table
#      BRK/STN codes, pre-inception claims,
#      messy occupation text, leaky contacts)
#
# Outputs (outputs_week2/): submission_week2.csv, validation tables, QA log, reference
# policy technical premiums, run manifest. Set IAA_SMOKE=1 for a fast dry run.
# =====================================================================================
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
T0 = time.time()

# ----------------------------------- config -----------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("IAA_DATA", ROOT / "data"))
EM_DIR, DV_DIR = DATA / "emerging", DATA / "developping"
OUT = Path(os.environ.get("IAA_OUT", ROOT / "outputs_week2"))
OUT.mkdir(exist_ok=True)

SMOKE = os.environ.get("IAA_SMOKE") == "1"
SEED = 42
SEEDS = [42] if SMOKE else [42, 7, 2025, 11, 99]
POWER = 1.5
LINES = ["motor", "buildings", "contents"]
VALUATION = pd.Timestamp("2024-12-31")
TARGET_YEAR = 2025
NEAREST_DAYS = 31
CAP_Q = 0.995                    # large-loss cap: line-level quantile (peril cells are too thin)
MATURE_OPEN_SHARE = 0.5
TREND_TAU = 0.03                 # prior sd of the emerging trend around the developed trend
TREND_BOUNDS = (-0.02, np.log(1.12))
N_FOLDS = 3 if SMOKE else 5
DEV_PRIOR_ROWS = 60_000 if SMOKE else None   # subsample developed rows per line in smoke mode
LEVEL_BAND = (0.80, 1.30)        # 2025 cost/exposure vs 2024 experience@2025: outside -> escalate
WGRID = np.round(np.arange(0, 1.01, 0.1), 2)
NB_CRED_K = 300                  # claims for 50% credibility in the new-business / renewal calibration
NB_CAL_BOUNDS = (0.80, 1.25)
CAL_CAP_Q = 0.975                # tighter cap used only to measure segment A/E robustly

LEAKY = ["customer_service_contacts"]          # Part 1 finding, re-checked below
# guardrail escalations that a human reviewed and decided to keep (reason is printed and logged)
REVIEWED_OVERRIDES = {
    "prior_log": "developed-market risk score built only from vetted inputs (customer_service_contacts "
                 "excluded); any genuine risk score separates claim / claim-free rows, so the leak "
                 "heuristic does not apply to model outputs (reviewed 2026-10-27, motor PSI 0.029)"}
NOISE = ["previous_insurer", "campaign_code", "email_domain", "branch_id", "vehicle_colour",
         "marketing_consent", "document_preference", "marketing_segment", "surname"]
AGE_BANDS = ["17-24", "25-34", "35-49", "50-64", "65+"]
CONSTRUCTION = {"BRK": "brick", "STN": "stone", "WOD": "timber", "MIX": "non-standard"}
WVARS = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
         "wildfire_index", "rainfall_mm", "mean_temperature_c"]
ORD = {"energy_rating": list("ABCDEFG"), "credit_band": list("ABCDE"),
       "urban_density": ["rural", "suburban", "urban", "dense_urban"],
       "security_level": ["standard", "locks", "alarm_and_locks"]}

# small-data LightGBM settings (Part 1 used min_data_in_leaf=3000 on ~1M rows/line)
EM_PARAMS = dict(objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.02,
                 num_leaves=6, max_depth=4, min_data_in_leaf=(30 if SMOKE else 400),
                 feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=50,
                 cat_smooth=100, min_data_per_group=200, max_cat_threshold=8,
                 seed=SEED, verbose=-1, num_threads=0)
# transfer model: boosts the emerging residual on top of the developed prior -> even more cautious
TRANSFER_PARAMS = {**EM_PARAMS, "learning_rate": 0.01, "num_leaves": 4, "max_depth": 3}
DEV_PARAMS = dict(objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.05,
                  num_leaves=15, min_data_in_leaf=(50 if SMOKE else 2000), feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=50, cat_smooth=200,
                  min_data_per_group=500, max_cat_threshold=8, seed=SEED, verbose=-1)


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def d2(y, p, w=None):
    return d2_tweedie_score(y, np.maximum(p, 1e-6), power=POWER, sample_weight=w)


def banner(title):
    print("\n" + "=" * 90 + f"\n>>> {title}\n" + "=" * 90)


QA = []          # (market, split, line, issue, rows)


# =====================================================================================
# STEP 1  load & clean (same cleaning for both markets so features are comparable)
# =====================================================================================
def read(d, name):
    """Reads <name>.parquet, tolerating browser-download suffixes such as 'addresses (1).parquet'."""
    p = d / f"{name}.parquet"
    if not p.exists():
        hits = sorted(d.glob(f"{name}*.parquet"))
        assert hits, f"{name}.parquet not found in {d}"
        p = hits[0]
    return pd.read_parquet(p)


def full_year_end(s):
    return s.dt.strftime("%m-%d").isin(["12-30", "12-31"])


def clean_policies(df, line, split, market):
    df = df.copy()
    df["line"] = line
    df["period_start"] = pd.to_datetime(df.period_start)
    df["period_end"] = pd.to_datetime(df.period_end)
    df["year"] = df.period_start.dt.year
    days = (((df.period_end - df.period_start).dt.days + 1) / 365).clip(upper=1)
    bad = df.exposure > 1.0001
    df.loc[bad, "exposure"] = days[bad]
    df["exposure"] = df.exposure.clip(lower=1 / 365)
    QA.append((market, split, line, "exposure > 1 (days / 370-day rows) -> recomputed from dates", int(bad.sum())))
    if "sum_insured" in df:
        big = df.sum_insured > 100 * df.sum_insured.median()
        df.loc[big, "sum_insured"] /= 1000
        QA.append((market, split, line, "sum_insured ~1000x too large -> /1000", int(big.sum())))
    if "construction_type" in df:
        coded = df.construction_type.isin(CONSTRUCTION.keys())
        df["construction_type"] = df.construction_type.replace(CONSTRUCTION).str.lower()
        QA.append((market, split, line, "construction_type BRK/STN/WOD/MIX codes -> names", int(coded.sum())))
    if "smoke_alarms" in df:
        df["smoke_alarms"] = df.smoke_alarms.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    # unified ordinal age band (emerging only reports bands; developed reports exact ages)
    age_col = next((c for c in ["driver_age", "policyholder_age"] if c in df), None)
    band_col = next((c for c in ["driver_age_band", "policyholder_age_band"] if c in df), None)
    if band_col:
        df["age_band"] = df[band_col].map({b: i for i, b in enumerate(AGE_BANDS)}).astype(float)
        df = df.drop(columns=band_col)
    elif age_col:
        df["age_band"] = pd.cut(df[age_col], [0, 24, 34, 49, 64, 200], labels=False).astype(float)
    return df


def link_claims(cl, pol, market):
    cl = cl.copy()
    cl["loss_date"] = pd.to_datetime(cl.loss_date)
    cl["report_date"] = pd.to_datetime(cl.report_date)
    cl["loss_year"] = cl.loss_date.dt.year
    cl["delay"] = (cl.report_date - cl.loss_date).dt.days
    QA.append((market, "train", "all", "repudiated claims (excluded from target)", int((cl.status == "repudiated").sum())))
    cl["net"] = np.where(cl.status == "repudiated", 0.0, (cl.incurred_to_date - cl.recoveries).clip(lower=0))
    cl = cl[cl.net > 0]
    pcols = ["row_id", "policy_id", "period_start", "period_end"]
    m = cl.merge(pol[pcols], on="policy_id")
    exact = m[(m.loss_date >= m.period_start) & (m.loss_date <= m.period_end)]
    assert not exact.claim_id.duplicated().any(), "claim matched to more than one policy row"
    near = cl[~cl.claim_id.isin(exact.claim_id)].merge(pol[pcols], on="policy_id")
    near["gap"] = np.where(near.loss_date < near.period_start, (near.period_start - near.loss_date).dt.days,
                           (near.loss_date - near.period_end).dt.days)
    near = near.sort_values("gap").drop_duplicates("claim_id")
    near = near[near.gap <= NEAREST_DAYS].drop(columns="gap")
    out = pd.concat([exact, near], ignore_index=True).drop(columns=["period_start", "period_end"])
    dropped = cl[~cl.claim_id.isin(out.claim_id)]
    QA.append((market, "train", "all", f"claims re-linked to nearest policy row (<= {NEAREST_DAYS}d)", len(near)))
    QA.append((market, "train", "all",
               f"claims outside any policy period dropped (loss years {sorted(dropped.loss_year.unique().tolist())})",
               len(dropped)))
    return out


def load_market(d, market):
    log(f"loading {market} market from {d}")
    cust, addr, wthr = read(d, "customers"), read(d, "addresses"), read(d, "weather")
    pol = pd.concat([clean_policies(read(d, f"policies_{l}"), l, "train", market) for l in LINES], ignore_index=True)
    tst = pd.concat([clean_policies(read(d, f"test_{l}"), l, "test", market) for l in LINES], ignore_index=True)
    if "occupation" in cust:
        n_raw = cust.occupation.nunique()
        cust["occupation"] = cust.occupation.str.strip().str.lower()
        QA.append((market, "all", "all", f"occupation text normalised ({n_raw} -> {cust.occupation.nunique()} levels)", n_raw))
    cl = link_claims(read(d, "claims"), pol, market)
    return dict(cust=cust, addr=addr, wthr=wthr, pol=pol, tst=tst, cl=cl, sample=read(d, "sample_submission"))


EM = load_market(EM_DIR, "emerging")
DV = load_market(DV_DIR, "developed")
expenses, reference = read(EM_DIR, "expenses"), read(EM_DIR, "reference_policies")
log(f"emerging: {len(EM['pol']):,} train rows, {len(EM['tst']):,} test rows, {len(EM['cl']):,} linked claims | "
    f"developed: {len(DV['pol']):,} train rows, {len(DV['cl']):,} linked claims")

# timing guardrail (Part 1 bug: 2024 rows end 12-30). Train-2024 and test must look alike *within*
# new business / renewal: the raw totals differ only because the 2025 book is 76% new business.
for l in LINES:
    a, t = EM["pol"][(EM["pol"].line == l) & (EM["pol"].year == 2024)], EM["tst"][EM["tst"].line == l]
    for nb in (False, True):
        aa, tt = a[a.new_business == nb], t[t.new_business == nb]
        ea, et = (~full_year_end(aa.period_end)).mean(), (~full_year_end(tt.period_end)).mean()
        assert abs(ea - et) < 0.03, f"{l} nb={nb}: ends-early share differs train-2024 {ea:.3f} vs test {et:.3f}"


# =====================================================================================
# STEP 2  what changed? (market comparison, written to outputs for the report)
# =====================================================================================
def experience(mk, years):
    p, c = mk["pol"], mk["cl"]
    g = c.groupby("row_id").agg(n=("net", "size"), cost=("net", "sum"))
    p = p[p.year.isin(years)].merge(g, left_on="row_id", right_index=True, how="left").fillna({"n": 0, "cost": 0})
    return p


banner("STEP 2 - WHAT CHANGED: emerging vs developed (same policy year 2023, raw incurred)")
cmp = []
for mk_name, mk in [("developed", DV), ("emerging", EM)]:
    e = experience(mk, [2023])
    for l, g in e.groupby("line"):
        cmp.append((l, mk_name, g.exposure.sum(), g.n.sum() / g.exposure.sum(), g.cost.sum() / max(g.n.sum(), 1),
                    g.cost.sum() / g.exposure.sum(), g.cost.sum() / g.written_premium.sum(),
                    g.new_business.mean()))
cmp = pd.DataFrame(cmp, columns=["line", "market", "exposure", "freq", "severity", "pure_prem", "loss_ratio", "nb_share"])
print(cmp.round(3).to_string(index=False))
sev_cmp = (pd.concat([DV["cl"].assign(m="developed"), EM["cl"].assign(m="emerging")])
           .query("loss_year == 2023").groupby(["line", "peril", "m"]).net.median().unstack().round(0))
sev_cmp["em/dev"] = (sev_cmp.emerging / sev_cmp.developed).round(2)
print("\nmedian claim size by peril (2023 losses):\n" + sev_cmp.to_string())
mix = pd.DataFrame({
    "train_2024_nb_share": EM["pol"][EM["pol"].year == 2024].groupby("line").new_business.mean(),
    "test_2025_nb_share": EM["tst"].groupby("line").new_business.mean(),
    "train_2024_age17_24": EM["pol"][EM["pol"].year == 2024].groupby("line").age_band.apply(lambda s: (s == 0).mean()),
    "test_2025_age17_24": EM["tst"].groupby("line").age_band.apply(lambda s: (s == 0).mean())})
print("\nportfolio mix shift (broker book transfer):\n" + mix.round(3).to_string())
wcmp = pd.concat([EM["wthr"].groupby("year")[WVARS].mean().assign(m="emerging"),
                  DV["wthr"].groupby("year")[WVARS].mean().assign(m="developed")])
print("\nclimate by year:\n" + wcmp.round(2).to_string())
nbr = []
for mk_name, mk in [("developed", DV), ("emerging", EM)]:
    e = experience(mk, sorted(mk["pol"].year.unique()))
    for l, g in e.groupby("line"):
        pp = g.groupby("new_business").apply(lambda d: d.cost.sum() / d.exposure.sum())
        nbr.append((l, mk_name, pp[True] / pp[False]))
print("\nnew-business / renewal raw pure-premium ratio:\n" +
      pd.DataFrame(nbr, columns=["line", "market", "nb_relativity"]).pivot(index="line", columns="market").round(2).to_string())
cmp.to_csv(OUT / "market_comparison.csv", index=False)


# =====================================================================================
# STEP 3  development (IBNR) & trend - carried over from developed where emerging is thin
# =====================================================================================
def completeness_table(cl_dev):
    """Share of ultimate claims reported by VALUATION, per line x peril x loss year, using the
    developed reporting-delay distribution (2021-22 losses: long, credible history)."""
    rows = []
    for (line, peril), g in cl_dev[cl_dev.loss_year.between(2021, 2022)].groupby(["line", "peril"]):
        d = np.sort(g.delay.values)
        for ly in range(2021, VALUATION.year + 1):
            days = (VALUATION - pd.date_range(f"{ly}-01-01", f"{ly}-12-31")).days.values
            rows.append((line, peril, ly, np.searchsorted(d, days, side="right").mean() / len(d)))
    return pd.DataFrame(rows, columns=["line", "peril", "loss_year", "completeness"])


COMP = completeness_table(DV["cl"])
banner("STEP 3a - REPORTING DELAY CHECK before carrying the developed pattern over")
dl = (pd.concat([DV["cl"][DV["cl"].loss_year.between(2021, 2022)].assign(m="developed"),
                 EM["cl"][EM["cl"].loss_year == 2023].assign(m="emerging")])
      .groupby(["line", "peril", "m"]).delay.quantile(0.9).unstack())
# emerging 2023 delays are truncated at 2 years, so compare a quantile both can see
dl["ratio_p90"] = dl.emerging / dl.developed
dl["carry_over_ok"] = dl.ratio_p90.between(0.6, 1.5)
print(dl.round(2).to_string())
if not dl.carry_over_ok.all():
    print(">>> ESCALATE: reporting delay differs materially for", dl.index[~dl.carry_over_ok].tolist(),
          "- developed IBNR pattern may not apply")


def severity_trends(cl, years, min_n=30):
    """Per line: claim-weighted mean of per-peril log-severity steps between consecutive years,
    using only mature cells, with its standard error. Returns {line: (slope, se, n_cells)}."""
    cell = (cl[cl.loss_year.isin(years)].groupby(["line", "peril", "loss_year"])
            .agg(n=("net_cap", "size"), sev=("net_cap", "mean"), sd=("net_cap", "std"),
                 open_share=("status", lambda s: (s == "open").mean())).reset_index())
    out = {}
    for line, g in cell.groupby("line"):
        steps, wts, ses = [], [], []
        for peril, gp in g.groupby("peril"):
            gp = gp.set_index("loss_year")
            for y0, y1 in zip(years[:-1], years[1:]):
                if y0 in gp.index and y1 in gp.index and min(gp.n[y0], gp.n[y1]) >= min_n \
                        and gp.open_share[y1] <= MATURE_OPEN_SHARE:
                    se2 = sum((gp.sd[y] / gp.sev[y]) ** 2 / gp.n[y] for y in (y0, y1))
                    steps.append(np.log(gp.sev[y1] / gp.sev[y0])); wts.append(1 / se2); ses.append(se2)
        if steps:
            w = np.array(wts)
            out[line] = (float(np.average(steps, weights=w)), float(np.sqrt(1 / w.sum())), len(steps))
        else:
            out[line] = (np.nan, np.inf, 0)
    return out


def apply_caps(cl, years):
    q = cl[cl.loss_year.isin(years)].groupby("line").net
    caps = pd.concat([q.quantile(CAP_Q).rename("cap"), q.quantile(CAL_CAP_Q).rename("cap_cal")], axis=1)
    cl = cl.drop(columns=["cap", "cap_cal"], errors="ignore").merge(caps, left_on="line", right_index=True)
    cl["net_cap"] = np.minimum(cl.net, cl.cap)
    cl["net_xs"] = cl.net - cl.net_cap
    return cl


DV["cl"] = apply_caps(DV["cl"], range(2021, 2025))
EM["cl"] = apply_caps(EM["cl"], [2023, 2024])
TR_DEV = severity_trends(DV["cl"], [2021, 2022, 2023])
TR_EM = severity_trends(EM["cl"], [2023, 2024])
TRENDS = {}
banner("STEP 3b - SEVERITY TREND: emerging (1 step) credibility-blended with developed (2 steps)")
for l in LINES:
    sd, _, _ = TR_DEV[l]
    se, se_se, k = TR_EM[l]
    z = 0.0 if not np.isfinite(se) else TREND_TAU ** 2 / (TREND_TAU ** 2 + se_se ** 2)
    TRENDS[l] = float(np.clip(z * (se if np.isfinite(se) else 0) + (1 - z) * sd, *TREND_BOUNDS))
    print(f"  {l:9s} developed {np.exp(sd) - 1:+.2%} | emerging {np.exp(se) - 1:+.2%} (se {se_se:.3f}, "
          f"{k} peril cells) | Z={z:.2f} -> used {np.exp(TRENDS[l]) - 1:+.2%} p.a.")
QA.append(("emerging", "assumption", "all", "severity trend p.a. " +
           ", ".join(f"{l}={np.exp(v) - 1:.1%}" for l, v in TRENDS.items()), 0))


def build_targets(rows, cl, trends, to_year):
    """Row-level target on an ultimate, to_year cost basis:
    y_cap = sum(capped net / completeness * trend), y_xs = same for the part above the cap."""
    c = cl.merge(COMP, on=["line", "peril", "loss_year"], how="left")
    c["completeness"] = c.completeness.fillna(1.0).clip(lower=0.3)
    c["ibnr"] = 1 / c.completeness
    c["trend"] = np.exp(c.line.map(trends).values * (to_year - c.loss_year.values))
    c["y_cap"] = c.net_cap * c.ibnr * c.trend
    c["y_xs"] = c.net_xs * c.ibnr * c.trend
    c["y_cal"] = np.minimum(c.net, c.cap_cal) * c.ibnr * c.trend
    agg = c.groupby("row_id").agg(y_cap=("y_cap", "sum"), y_xs=("y_xs", "sum"), y_raw=("net", "sum"),
                                  y_cal=("y_cal", "sum"),
                                  n_claims=("net", "size"), n_dev=("ibnr", "sum"))
    out = rows.merge(agg, left_on="row_id", right_index=True, how="left")
    tcols = ["y_cap", "y_xs", "y_raw", "y_cal", "n_claims", "n_dev"]
    out[tcols] = out[tcols].fillna(0.0)
    out["y_tot"] = out.y_cap + out.y_xs
    return out


TRENDS_DV = {l: float(np.clip(TR_DEV[l][0], *TREND_BOUNDS)) for l in LINES}
em_train = build_targets(EM["pol"], EM["cl"], TRENDS, TARGET_YEAR)
dv_train = build_targets(DV["pol"], DV["cl"], TRENDS_DV, TARGET_YEAR)
ibnr_used = (EM["cl"].merge(COMP, on=["line", "peril", "loss_year"]).groupby(["line", "loss_year"])
             .apply(lambda g: g.net.sum() / (g.net * g.completeness).sum()).unstack().round(3))
print("\nIBNR uplift applied to emerging incurred (ultimate / reported):\n" + ibnr_used.to_string())

banner("STEP 3c - EMERGING EXPERIENCE at 2025 cost level (developed for IBNR, trended)")
lv = em_train.groupby(["line", "year"]).agg(exposure=("exposure", "sum"), n_dev=("n_dev", "sum"),
                                            y_tot=("y_tot", "sum"), prem=("written_premium", "sum"))
lv["freq"] = lv.n_dev / lv.exposure
lv["pp_2025"] = lv.y_tot / lv.exposure
lv["LR_2025"] = lv.y_tot / lv.prem
print(lv.round(3).to_string())


# =====================================================================================
# STEP 4  features
# =====================================================================================
def history(panel):
    """Prior-year (developed) claim counts by policy / customer. Strictly years < row year."""
    p = panel[["policy_id", "customer_id", "year", "exposure", "n_dev"]].copy()
    feats = []
    for key, pre in [("policy_id", "pol"), ("customer_id", "cust")]:
        g = p.groupby([key, "year"]).agg(n=("n_dev", "sum"), e=("exposure", "sum")).reset_index()
        g = g.sort_values([key, "year"])
        g[f"{pre}_hist_n"] = g.groupby(key).n.cumsum() - g.n
        g[f"{pre}_hist_e"] = g.groupby(key).e.cumsum() - g.e
        g.loc[g[f"{pre}_hist_e"] <= 0, [f"{pre}_hist_n", f"{pre}_hist_e"]] = np.nan
        feats.append((key, g.drop(columns=["n", "e"])))
    held = p.groupby(["customer_id", "year"]).size().rename("cust_rows").reset_index()
    return feats, held


def make_features(df, mk, hist=None):
    n0 = len(df)
    cust = mk["cust"].drop(columns=[c for c in NOISE if c in mk["cust"]])
    if "date_of_birth" in cust:
        cust = cust.drop(columns="date_of_birth")
    w = mk["wthr"]
    w_mean = w.groupby("address_id")[WVARS].mean().add_suffix("_avg")
    f = (df.merge(cust, on="customer_id", how="left").merge(mk["addr"], on="address_id", how="left")
           .merge(w[["address_id", "year"] + WVARS], on=["address_id", "year"], how="left")
           .merge(w_mean, left_on="address_id", right_index=True, how="left"))
    for v in WVARS:
        f[v] = f[v].fillna(f[v + "_avg"])
        f[v + "_anom"] = f[v] - f[v + "_avg"]
    if hist is not None:
        feats, held = hist
        for key, g in feats:
            f = f.merge(g, on=[key, "year"], how="left")
        f = f.merge(held, on=["customer_id", "year"], how="left")
        for pre in ["pol", "cust"]:
            f[f"{pre}_hist_freq"] = (f[f"{pre}_hist_n"] + 0.1) / (f[f"{pre}_hist_e"] + 1)
    assert len(f) == n0, "feature merge changed row count"
    for col, levels in ORD.items():
        if col in f:
            f[col] = f[col].map({v: i for i, v in enumerate(levels)})
    if "gender" in f:
        f["gender"] = (f.gender == "F").astype(float)
    f["ends_early"] = (~full_year_end(f.period_end)).astype(int)
    f["starts_late"] = (f.period_start.dt.strftime("%m-%d") != "01-01").astype(int)
    f["customer_tenure"] = f.year - f.customer_since_year
    f["property_age"] = f.year - f.year_built
    f["total_excess"] = f.voluntary_excess + f.compulsory_excess
    # peril interactions on the address's long-run climate (stable across years and markets)
    f["clay_dry_trees"] = (f.soil_type == "clay") * f.dry_summer_index_avg * f.trees_within_5m
    f["elev_over_river"] = f.elevation_m / (f.distance_to_river_km.clip(lower=0) + 0.1)
    f["frost_x_age"] = f.freezing_days_avg * f.property_age
    f["hail_x_flat_roof"] = f.hail_days_avg * (f.roof_type == "flat")
    f["wildfire_x_timber"] = f.wildfire_index_avg * (f.get("construction_type", "") == "timber")
    if "sum_insured" in f:
        # sums insured are in each market's own currency scale -> use relativity to market/line median
        f["log_si_rel"] = np.log(f.sum_insured) - np.log(f.groupby("line").sum_insured.transform("median"))
        f["excess_to_si"] = f.total_excess / f.sum_insured
        f["si_per_bedroom_rel"] = np.exp(f.log_si_rel) / f.bedrooms.clip(lower=1)
    if "ncb_years" in f:
        f["ncb_ratio"] = f.ncb_years / f.years_licensed.clip(lower=1)
        f["excess_per_vvg"] = f.total_excess / f.vehicle_value_group.clip(lower=1)
    f["log_premium_rel"] = np.log(f.written_premium) - np.log(f.groupby(["line", "year"]).written_premium.transform("median"))
    return f


# features that mean different things across markets / pricing regimes -> never in the developed prior
NOT_SHARED = {"region", "make_model", "log_premium_rel", "gender", "credit_band", "driver_age", "policyholder_age",
              "annual_mileage", "telematics_opt_in", "telematics_score", "occupation",
              "pol_hist_n", "pol_hist_e", "pol_hist_freq", "cust_hist_n", "cust_hist_e", "cust_hist_freq", "cust_rows"}
BASE_DROP = set(LEAKY + NOISE + ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end",
                                 "renewal_decision", "y_cap", "y_xs", "y_raw", "y_cal", "y_tot", "n_claims", "n_dev", "line",
                                 "year", "exposure", "written_premium", "customer_since_year", "year_built",
                                 "sum_insured", "make_model"])
# the developed prior keeps year-specific weather (4 years: effects are learnable) but not the
# address climate averages; emerging models do the opposite - with only 2 policy years any
# year-specific weather value acts as a disguised "year" flag (PSI ~7 for 2025 anomalies)
YEAR_WEATHER = set(WVARS) | {v + "_anom" for v in WVARS}
DEV_DROP = BASE_DROP | NOT_SHARED | {v + "_avg" for v in WVARS} | {"prior_log"}
EM_DROP = BASE_DROP | YEAR_WEATHER


def cat_align(tr, ap, feats, cats=None):
    """Turns text/bool columns into pandas categoricals with identical categories on both frames."""
    cats = dict(cats or {})
    tr, ap = tr.copy(), ap.copy()
    for c in feats:
        if tr[c].dtype == object or tr[c].dtype == bool or str(tr[c].dtype) in ("category", "string", "str"):
            if c not in cats:
                cats[c] = pd.Categorical(pd.concat([tr[c], ap[c]]).astype(str)).categories
            tr[c] = pd.Categorical(tr[c].astype(str), categories=cats[c])
            ap[c] = pd.Categorical(ap[c].astype(str), categories=cats[c])
    return tr, ap, cats


# =====================================================================================
# STEP 5  developed-market prior (Part 1 knowledge, shared features only)
# =====================================================================================
def fit_tweedie(X, y, expo, params, init=None, n_rounds=None, groups=None, max_rounds=3000, weight=None):
    """LightGBM Tweedie with log(exposure) (+ optional prior) offset. Early-stops on a 15% policy
    holdout when n_rounds is None. Returns (booster, rounds, offset_base, balance)."""
    off = np.log(expo) + (0.0 if init is None else init)
    base = np.log(y.sum() / np.exp(off).sum())
    off = off + base
    if n_rounds is None:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(X, groups=groups))
        dtr = lgb.Dataset(X.iloc[a], y[a], init_score=off[a], weight=None if weight is None else weight[a])
        dva = lgb.Dataset(X.iloc[b], y[b], init_score=off[b], weight=None if weight is None else weight[b])
        m = lgb.train(params, dtr, max_rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        n_rounds = max(int(m.best_iteration), 10)
    m = lgb.train(params, lgb.Dataset(X, y, init_score=off, weight=weight), n_rounds)
    bal = y.sum() / np.exp(m.predict(X, raw_score=True) + off).sum()
    return m, n_rounds, base, bal


def pred_tweedie(model, X, expo, init=None):
    m, _, base, bal = model
    return np.exp(m.predict(X, raw_score=True) + np.log(expo) + (0.0 if init is None else init) + base) * bal


banner("STEP 5 - DEVELOPED PRIOR: Part 1 market model on features both markets share")
TCOLS = ["y_cap", "y_xs", "y_raw", "y_cal", "y_tot", "n_claims", "n_dev"]
em_test_rows = EM["tst"].assign(**{c: np.nan for c in TCOLS})
em_hist = history(pd.concat([em_train, em_test_rows]))
EF = make_features(em_train, EM, em_hist)
ET = make_features(em_test_rows, EM, em_hist)
# 20 reference policies from the emerging market (commercial sense-check in STEP 9)
RF = pd.concat([make_features(clean_policies(reference[reference.line == l].drop(columns="line")
                                             .dropna(axis=1, how="all"), l, "reference", "emerging")
                              .assign(**{c: np.nan for c in TCOLS}), EM, em_hist)
                for l in LINES if (reference.line == l).any()], ignore_index=True)
RF["row_id"] = "REF_" + RF.row_id.astype(str)     # reference row_ids collide with training rows

cache_key = hashlib.md5(json.dumps([DEV_PARAMS, CAP_Q, TRENDS_DV, SMOKE, sorted(DEV_DROP), "v3"], sort_keys=True, default=str)
                        .encode()).hexdigest()[:10]
cache = OUT / f"dev_prior_{cache_key}.parquet"
SHARED = {}
if cache.exists():
    prior = pd.read_parquet(cache)
    log(f"developed prior loaded from cache {cache.name}")
else:
    dvf = make_features(dv_train, DV)
    rows = []
    for l in LINES:
        tr = dvf[dvf.line == l]
        if DEV_PRIOR_ROWS:
            tr = tr.sample(min(DEV_PRIOR_ROWS, len(tr)), random_state=SEED)
        ape = pd.concat([EF[EF.line == l], ET[ET.line == l], RF[RF.line == l]])
        feats = [c for c in tr.columns if c in ape.columns and c not in DEV_DROP
                 and tr[c].notna().any() and ape[c].notna().any()]
        tr_, ap_, _ = cat_align(tr, ape, feats)
        mdl = fit_tweedie(tr_[feats], tr_.y_cap.values, tr_.exposure.values, DEV_PARAMS, groups=tr_.policy_id)
        # prior_log = developed log-rate relativity (level is irrelevant: re-based on emerging data later)
        raw = mdl[0].predict(ap_[feats], raw_score=True)
        rows.append(pd.DataFrame({"row_id": ap_.row_id.values, "prior_log": raw - np.average(raw)}))
        imp = pd.Series(mdl[0].feature_importance("gain"), index=feats)
        log(f"  {l:9s} developed prior: {len(tr):,} rows, {len(feats)} shared features, {mdl[1]} rounds | "
            f"top: {(imp / imp.sum()).sort_values(ascending=False).head(6).round(3).to_dict()}")
    prior = pd.concat(rows, ignore_index=True)
    assert prior.row_id.is_unique, "duplicate row_id in developed prior"
    prior.to_parquet(cache, index=False)
    del dvf
EF = EF.merge(prior, on="row_id", how="left")
ET = ET.merge(prior, on="row_id", how="left")
RF = RF.merge(prior, on="row_id", how="left")
assert EF.prior_log.notna().all() and ET.prior_log.notna().all() and RF.prior_log.notna().all(), \
    "developed prior missing for some rows"


# =====================================================================================
# STEP 6  leakage & shift guardrail (train-2024 vs test-2025)
# =====================================================================================
def psi(a, b, bins=10):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if a.nunique() <= 1 or len(b) == 0:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.r_[a.min() - 1, np.sort(a.unique())[:bins], a.max() + 1])
    pa = np.histogram(a, edges)[0] / len(a) + 1e-4
    pb = np.histogram(b.clip(edges[0], edges[-1]), edges)[0] / len(b) + 1e-4
    return float(np.sum((pa - pb) * np.log(pa / pb)))


banner("STEP 6 - SHIFT / LEAK GUARDRAIL (leak = test resembles *claim-free* training rows)")
num_cols = [c for c in EF.select_dtypes("number").columns if c not in EM_DROP or c in LEAKY]
shift = []
for l in LINES:
    a = EF[(EF.line == l) & (EF.year == 2024) & ~EF.new_business]   # like-for-like: renewals vs renewals
    t = ET[(ET.line == l) & ~ET.new_business]
    for c in num_cols:
        if a[c].notna().any() and t[c].notna().any():
            shift.append((l, c, psi(a[c], t[c]), psi(a.loc[a.n_claims == 0, c], t[c])))
shift = pd.DataFrame(shift, columns=["line", "feature", "psi_train_vs_test", "psi_noclaim_vs_test"])
shift["leak_signature"] = (shift.psi_train_vs_test > 0.02) & (shift.psi_noclaim_vs_test < 0.5 * shift.psi_train_vs_test)
print(shift.sort_values("psi_train_vs_test", ascending=False).head(10).round(4).to_string(index=False))
flagged = sorted(shift.loc[shift.leak_signature, "feature"].unique())
print("leak signature:", flagged)
for c in sorted(set(flagged) & set(REVIEWED_OVERRIDES)):
    print(f"   human-review override kept '{c}': {REVIEWED_OVERRIDES[c]}")
    QA.append(("emerging", "governance", "all", f"leak guardrail flagged '{c}' - kept after review", 0))
extra_leaks = [c for c in flagged if c not in LEAKY and c not in REVIEWED_OVERRIDES]
if extra_leaks:
    print(f">>> ESCALATE: new leak-like features auto-dropped, needs human review: {extra_leaks}")
DROP = EM_DROP | set(extra_leaks)
shift.to_csv(OUT / "shift_guardrail.csv", index=False)


# =====================================================================================
# STEP 7  candidate models & validation
#   A  emerging-only GBM                      (ignores Part 1)
#   B  emerging GBM + developed prior feature (model decides how much to trust Part 1)
#   C  transfer: developed prior as offset, GBM learns emerging corrections only
#   P  developed prior recalibrated to emerging (intercept + slope GLM) - pure carry-over benchmark
# =====================================================================================
def target_encode(tr, ap, col, smooth=300):
    prior = tr.n_dev.sum() / tr.exposure.sum()

    def enc(src, dst):
        s = src.groupby(col).agg(n=("n_dev", "sum"), e=("exposure", "sum"))
        return dst[col].map((s.n + smooth * prior) / (s.e + smooth)).astype(float).fillna(prior).values / prior
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(tr, groups=tr.policy_id):
        oof[b] = enc(tr.iloc[a], tr.iloc[b])
    return oof, enc(tr, ap)


def prep(tr, ap):
    tr, ap = tr.copy(), ap.copy()
    if "make_model" in tr and tr.make_model.notna().any():
        tr["make_model_te"], ap["make_model_te"] = target_encode(tr, ap, "make_model")
    feats = [c for c in tr.columns if c not in DROP and tr[c].notna().any()]
    tr, ap, _ = cat_align(tr, ap, feats)
    assert not {"y_cap", "y_xs", "y_tot", "y_raw", "y_cal", "n_claims", "n_dev"} & set(feats), "target leaked into features"
    assert not set(LEAKY) & set(feats), "leaky feature in model"
    return tr, ap, feats


def fit_glm_prior(tr):
    """Tweedie GLM log(mu) = log(expo) + a + b * prior_log, fitted by IRLS-free Newton steps."""
    x, y, e = tr.prior_log.values, tr.y_cap.values, tr.exposure.values
    a, b = np.log(y.sum() / e.sum()), 1.0
    for _ in range(50):
        mu = e * np.exp(a + b * x)
        # Tweedie(p) score & Fisher information for log link
        r = (y - mu) * mu ** (1 - POWER)
        wv = mu ** (2 - POWER)
        g = np.array([r.sum(), (r * x).sum()])
        H = np.array([[wv.sum(), (wv * x).sum()], [(wv * x).sum(), (wv * x * x).sum()]])
        step = np.linalg.solve(H, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return a, b


def run_models(tr, ap, rounds=None, seed=SEED, loading=None):
    """Fits A, B, C, P on tr and predicts ap. Returns predictions (incl. large-loss loading) and rounds."""
    rounds = rounds or {}
    tr, ap, feats = prep(tr, ap)
    feats_nb = [c for c in feats if c != "prior_log"]
    loading = tr.y_xs.sum() / tr.y_cap.sum() if loading is None else loading
    y, e, g = tr.y_cap.values, tr.exposure.values, tr.policy_id
    pa, pr = {}, {}
    pA = {**EM_PARAMS, "seed": seed}
    pC = {**TRANSFER_PARAMS, "seed": seed}
    mA = fit_tweedie(tr[feats_nb], y, e, pA, n_rounds=rounds.get("A"), groups=g)
    pa["A"], pr["A"] = pred_tweedie(mA, ap[feats_nb], ap.exposure.values), mA[1]
    mB = fit_tweedie(tr[feats], y, e, pA, n_rounds=rounds.get("B"), groups=g)
    pa["B"], pr["B"] = pred_tweedie(mB, ap[feats], ap.exposure.values), mB[1]
    a, b = fit_glm_prior(tr)
    init_tr, init_ap = b * tr.prior_log.values, b * ap.prior_log.values
    mC = fit_tweedie(tr[feats_nb], y, e, pC, init=init_tr, n_rounds=rounds.get("C"), groups=g)
    pa["C"], pr["C"] = pred_tweedie(mC, ap[feats_nb], ap.exposure.values, init=init_ap), mC[1]
    pa["P"] = ap.exposure.values * np.exp(a + b * ap.prior_log.values)
    pa = {k: v * (1 + loading) for k, v in pa.items()}
    imp = pd.Series(mB[0].feature_importance("gain"), index=feats)
    return pa, pr, {"glm_slope": b, "loading": loading, "imp": imp / imp.sum()}


def score_table(V, cols, y="y_tot"):
    """Per-line D2 is weighted to the 2025 new-business mix (column w); ALL is shown both ways.
    P/A = predicted / actual (>1 = over-predicting)."""
    out = {}
    for c in cols:
        r = {f"{l}_testmix": d2(g[y], g[c], g.w) for l, g in V.groupby("line")}
        r["ALL_testmix"] = d2(V[y], V[c], V.w)
        r["ALL_unweighted"] = d2(V[y], V[c])
        r["new_business_only"] = d2(V.loc[V.nb, y], V.loc[V.nb, c])
        r["P/A"] = V[c].sum() / V[y].sum()
        r["P/A_new_business"] = V.loc[V.nb, c].sum() / V.loc[V.nb, y].sum()
        out[c] = r
    return pd.DataFrame(out).T


# covariate-shift weights: validation rows re-weighted so new business / renewal match the 2025 book
_nb_tr = EF.groupby("line").new_business.mean()
_nb_te = ET.groupby("line").new_business.mean()
MIXW = {l: {True: _nb_te[l] / _nb_tr[l], False: (1 - _nb_te[l]) / (1 - _nb_tr[l])} for l in LINES}
print("validation weights (new business, renewal):",
      {l: (round(float(v[True]), 2), round(float(v[False]), 2)) for l, v in MIXW.items()})


def mixw(line, nb):
    return np.where(np.asarray(nb, dtype=bool), MIXW[line][True], MIXW[line][False])


MODELS = ["A", "B", "C", "P"]
banner(f"STEP 7a - {N_FOLDS}-FOLD CV grouped by policy, 2023-24 pooled at 2025 cost level")
cv_parts, cv_rounds, cv_info = [], {l: [] for l in LINES}, {}
for l in LINES:
    d = EF[EF.line == l].reset_index(drop=True)
    for k, (a, b) in enumerate(GroupKFold(N_FOLDS).split(d, groups=d.policy_id)):
        tr, va = d.iloc[a], d.iloc[b]
        pa, pr, info = run_models(tr, va)
        cv_rounds[l].append(pr)
        cv_parts.append(pd.DataFrame({"line": l, "row_id": va.row_id.values, "year": va.year.values,
                                      "nb": va.new_business.values.astype(bool), "y_tot": va.y_tot.values,
                                      "y_raw": va.y_raw.values, "exposure": va.exposure.values,
                                      "n_claims": va.n_dev.values, "w": mixw(l, va.new_business.values),
                                      "y_cal": va.y_cal.values,
                                      **{f"pred_{m}": pa[m] for m in MODELS}}))
    cv_info[l] = info
    log(f"  {l:9s} rounds A/B/C: {[(r['A'], r['B'], r['C']) for r in cv_rounds[l]]} | GLM slope on developed prior "
        f"{info['glm_slope']:.2f} | large-loss loading {info['loading']:.2%}")
CV = pd.concat(cv_parts, ignore_index=True)
CV["exp_rate"] = CV.groupby("line").apply(lambda g: pd.Series(g.exposure * g.y_tot.sum() / g.exposure.sum(),
                                                                index=g.index)).reset_index(level=0, drop=True)
cv_tab = score_table(CV, ["exp_rate"] + [f"pred_{m}" for m in MODELS])
print(cv_tab.round(4).to_string())

# blend weights per line on CV (A, B, C, P on a simplex grid; level-neutral)
banner("STEP 7b - BLEND WEIGHTS per line (CV, D2 vs developed actuals)")
WEIGHTS = {}
for l in LINES:
    g = CV[CV.line == l]
    best, best_w = -np.inf, None
    for wa in WGRID:
        for wb in WGRID:
            for wc in WGRID:
                wp = round(1 - wa - wb - wc, 2)
                if wp < -1e-9:
                    continue
                p = wa * g.pred_A + wb * g.pred_B + wc * g.pred_C + wp * g.pred_P
                s = d2(g.y_tot, p, g.w)
                if s > best + 1e-6:          # ties -> keep the earlier (simpler) weights
                    best, best_w = s, (float(wa), float(wb), float(wc), float(max(wp, 0.0)))
    WEIGHTS[l] = dict(zip(MODELS, best_w))
    print(f"  {l:9s} weights A/B/C/P = {best_w} -> CV D2 (2025 mix) {best:.4f} "
          f"(best single: {max(d2(g.y_tot, g[f'pred_{m}'], g.w) for m in MODELS):.4f})")
CV["pred_blend"] = sum(CV.line.map(lambda l, m=m: WEIGHTS[l][m]) * CV[f"pred_{m}"] for m in MODELS)

# new-business / renewal calibration. The model under-uses new_business (8% of training exposure,
# 76% of 2025). Segment A/E is measured out-of-fold on claims capped at the line's 97.5th percentile
# (2023 motor NB pure premium was 2.25x renewals vs 1.15x in 2024 - a few large claims), relative
# to the line's overall A/E so only the NB / renewal split moves, then credibility-weighted.
banner("STEP 7b2 - NEW-BUSINESS CALIBRATION (robust out-of-fold A/E, Z = n / (n + %d claims))" % NB_CRED_K)
NB_CAL = {}


def seg_ae(g, G):
    return (g.y_cal.sum() / g.pred_blend.sum()) / (G.y_cal.sum() / G.pred_blend.sum())


for l in LINES:
    NB_CAL[l] = {}
    G = CV[CV.line == l]
    for nb in (True, False):
        g = G[G.nb == nb]
        ae, n = seg_ae(g, G), g.n_claims.sum()
        by_year = {int(y): round(float(seg_ae(g[g.year == y], G[G.year == y])), 3) for y in sorted(G.year.unique())}
        z = n / (n + NB_CRED_K)
        NB_CAL[l][nb] = float(np.clip(1 + z * (ae - 1), *NB_CAL_BOUNDS))
        print(f"  {l:9s} {'new business' if nb else 'renewal     '}: claims {n:6.0f} A/E {ae:.3f} by year {by_year} "
              f"Z {z:.2f} -> factor {NB_CAL[l][nb]:.3f}")
CV["pred_blend_cal"] = CV.pred_blend * [NB_CAL[l][nb] for l, nb in zip(CV.line, CV.nb)]
cv_tab = score_table(CV, ["exp_rate"] + [f"pred_{m}" for m in MODELS] + ["pred_blend", "pred_blend_cal"])
print("note: pred_blend_cal is in-sample for the 2 calibration factors per line")
print("\n" + cv_tab.round(4).to_string())
cv_tab.to_csv(OUT / "cv_scores.csv")

# temporal back-test: does a model built on 2023 alone price 2024 (level + ranking)?
banner("STEP 7c - TEMPORAL BACK-TEST: train 2023 -> score 2024 (2024 targets developed, not trended)")
bt_parts = []
for l in LINES:
    tr = EF[(EF.line == l) & (EF.year == 2023)]
    va = EF[(EF.line == l) & (EF.year == 2024)].copy()
    # put 2023 targets on a 2024 cost basis and 2024 targets at 2024 cost
    tr = tr.assign(y_cap=tr.y_cap / np.exp(TRENDS[l]), y_xs=tr.y_xs / np.exp(TRENDS[l]))
    va = va.assign(y_tot=va.y_tot / np.exp(TRENDS[l]))
    pa, _, _ = run_models(tr, va)
    blend = sum(WEIGHTS[l][m] * pa[m] for m in MODELS)
    blend_cal = blend * np.where(va.new_business.values.astype(bool), NB_CAL[l][True], NB_CAL[l][False])
    bt_parts.append(pd.DataFrame({"line": l, "nb": va.new_business.values.astype(bool), "y_tot": va.y_tot.values,
                                  "w": mixw(l, va.new_business.values), "pred_blend_cal": blend_cal,
                                  "y_raw": va.y_raw.values, "exposure": va.exposure.values,
                                  **{f"pred_{m}": pa[m] for m in MODELS}, "pred_blend": blend}))
BT = pd.concat(bt_parts, ignore_index=True)
bt_tab = score_table(BT, [f"pred_{m}" for m in MODELS] + ["pred_blend", "pred_blend_cal"])
print(bt_tab.round(4).to_string())
print("note: pred_blend_cal uses NB factors estimated on CV folds that include 2024 -> optimistic")
for c in ["pred_blend", "pred_blend_cal"]:
    print(f"P/A by line {c:15s}", (BT.groupby("line")[c].sum() / BT.groupby("line").y_tot.sum()).round(3).to_dict(),
          "| new business only", (BT[BT.nb].groupby("line")[c].sum() / BT[BT.nb].groupby("line").y_tot.sum()).round(3).to_dict())
bt_tab.to_csv(OUT / "backtest_2024_scores.csv")

# lift check on CV: does the blend rank risk (actual/expected by predicted decile)?
CV["decile"] = CV.groupby("line").pred_blend_cal.transform(lambda s: pd.qcut(s / CV.loc[s.index, "exposure"], 10,
                                                                         labels=False, duplicates="drop"))
lift = CV.groupby(["line", "decile"]).agg(pred=("pred_blend_cal", "sum"), act=("y_tot", "sum"), e=("exposure", "sum"))
lift["pred_pp"], lift["act_pp"] = lift.pred / lift.e, lift.act / lift.e
print("\nlift (CV, cost per exposure by predicted decile):\n" +
      lift[["pred_pp", "act_pp"]].unstack("line").round(0).to_string())
lift.to_csv(OUT / "cv_lift.csv")


# =====================================================================================
# STEP 8  final fit on 2023-24 -> predict 2025
# =====================================================================================
banner("STEP 8 - FINAL FIT 2023-2024 -> 2025")
preds, LEVEL_2024 = [], {}
for l in LINES:
    tr, ap = EF[EF.line == l], ET[ET.line == l]
    rounds = {m: int(np.mean([r[m] for r in cv_rounds[l]]) * 1.1) for m in ["A", "B", "C"]}
    loading = tr.y_xs.sum() / tr.y_cap.sum()
    runs = [run_models(tr, ap, rounds=rounds, seed=s, loading=loading)[0] for s in SEEDS]
    comp = {m: np.mean([r[m] for r in runs], axis=0) for m in MODELS}
    p_uncal = sum(WEIGHTS[l][m] * comp[m] for m in MODELS)
    p = p_uncal * np.where(ap.new_business.values.astype(bool), NB_CAL[l][True], NB_CAL[l][False])
    # level check: 2025 predicted cost/exposure vs 2024 experience at 2025 costs, mix-adjusted view
    # 2023 was a wildfire year for buildings, so compare with the pooled 2-year experience
    hist_pp = tr.y_tot.sum() / tr.exposure.sum()
    pp24 = tr[tr.year == 2024].pipe(lambda d: d.y_tot.sum() / d.exposure.sum())
    ratio = p.sum() / ap.exposure.sum() / hist_pp
    flag = "" if LEVEL_BAND[0] <= ratio <= LEVEL_BAND[1] else "  <-- ESCALATE: outside level band, human review"
    nb_mask = ap.new_business.values.astype(bool)
    nb_rel = (p[nb_mask].sum() / ap.exposure.values[nb_mask].sum()) / \
             (p[~nb_mask].sum() / ap.exposure.values[~nb_mask].sum())
    log(f"  {l:9s} rounds {rounds} | 2025 cost/exp {p.sum() / ap.exposure.sum():8.1f} vs 2023-24@2025 {hist_pp:8.1f} "
        f"(x{ratio:.2f}){flag} | 2024 only {pp24:8.1f} | NB/renewal relativity {nb_rel:.2f}")
    LEVEL_2024[l] = pp24 / hist_pp
    for m in MODELS:
        if WEIGHTS[l][m] > 0:
            print(f"      component {m}: weight {WEIGHTS[l][m]:.1f} level {comp[m].sum() / p.sum():.3f} "
                  f"corr {np.corrcoef(comp[m], p)[0, 1]:.3f}")
    print(f"      new-business calibration moves line level by x{p.sum() / p_uncal.sum():.3f}")
    preds.append(pd.DataFrame({"row_id": ap.row_id.values, "line": l, "prediction": p, "pred_uncal": p_uncal,
                               **{f"pred_{m}": comp[m] for m in MODELS}}))
P = pd.concat(preds, ignore_index=True)

# ------------------------------- submission guardrails --------------------------------
sample = EM["sample"]
sub = sample[["row_id"]].merge(P[["row_id", "prediction"]], on="row_id", how="left")
assert len(sub) == len(sample) and sub.row_id.is_unique, "row count / uniqueness mismatch vs sample submission"
assert set(sub.row_id) == set(EM["tst"].row_id), "row_ids differ from the test files"
assert sub.prediction.notna().all() and np.isfinite(sub.prediction).all(), "missing / non-finite predictions"
sub["prediction"] = sub.prediction.clip(lower=0.01).round(4)
assert (sub.prediction >= 0).all()
sub_path = OUT / "submission_week2.csv"
sub.to_csv(sub_path, index=False)
P.to_parquet(OUT / "predictions_components.parquet", index=False)
log(f"wrote {sub_path} rows={len(sub):,} mean={sub.prediction.mean():.2f} max={sub.prediction.max():.0f}")


# =====================================================================================
# STEP 9  commercial sense-check on the 20 reference policies (expenses.parquet)
# =====================================================================================
banner("STEP 9 - TECHNICAL PREMIUM SENSE-CHECK on reference policies (emerging expense basis)")
ex = expenses.set_index("item").emerging
ref_rows = []
for l in LINES:
    rf = RF[RF.line == l].reset_index(drop=True)
    if rf.empty:
        continue
    tr = EF[EF.line == l]
    rounds = {m: int(np.mean([x[m] for x in cv_rounds[l]]) * 1.1) for m in ["A", "B", "C"]}
    pa, _, _ = run_models(tr, rf.reindex(columns=tr.columns), rounds=rounds, loading=tr.y_xs.sum() / tr.y_cap.sum())
    rp = sum(WEIGHTS[l][m] * pa[m] for m in MODELS)
    rp = rp * np.where(rf.new_business.values.astype(bool), NB_CAL[l][True], NB_CAL[l][False])
    for i in range(len(rf)):
        cr = ex.get(f"commission_{rf.channel.iloc[i]}", np.nan)
        cr = ex["commission_broker"] if pd.isna(cr) else cr       # comparison sites not used in emerging
        risk = rp[i] * (1 + (ex["cat_reinsurance"] if l != "motor" else 0.0))
        tech = (risk + ex["fixed_expense_per_policy"] * rf.exposure.iloc[i]) / \
               (1 - ex["variable_expense"] - cr - ex["premium_tax"] - ex["target_profit_margin"])
        ref_rows.append((rf.row_id.iloc[i].removeprefix("REF_"), l, rf.channel.iloc[i], rf.written_premium.iloc[i], rp[i], tech,
                         tech / rf.written_premium.iloc[i]))
REF = pd.DataFrame(ref_rows, columns=["row_id", "line", "channel", "written_premium", "risk_premium",
                                      "technical_premium", "tech/written"])
print(REF.round(2).to_string(index=False))
print("note: cat reinsurance loaded on property lines only; fixed expense pro-rated by exposure")
print("technical / written premium by line:", REF.groupby("line").apply(
    lambda g: g.technical_premium.sum() / g.written_premium.sum()).round(2).to_dict())
REF.to_csv(OUT / "reference_policies_technical_premium.csv", index=False)


# =====================================================================================
# STEP 10  QA log, uncertainty statement and run manifest (reproducibility)
# =====================================================================================
banner("STEP 10 - DATA-QUALITY LOG")
qa = pd.DataFrame(QA, columns=["market", "split", "line", "issue", "rows"])
print(qa.to_string(index=False))
qa.to_csv(OUT / "qa_log.csv", index=False)

banner("STEP 10b - UNCERTAINTY: line-level 2025 cost under alternative assumptions")
unc = []
for l in LINES:
    p = P[P.line == l]
    base = p.prediction.sum()
    t_dv, t_em = TR_DEV[l][0], TR_EM[l][0]
    for name, f in [("trend = developed only", np.exp(t_dv - TRENDS[l])),
                    ("trend = emerging only", np.exp((t_em if np.isfinite(t_em) else TRENDS[l]) - TRENDS[l])),
                    ("emerging-only model (A)", p.pred_A.sum() / base),
                    ("pure carry-over (P)", p.pred_P.sum() / base),
                    ("CV pred/actual (calibrated)", CV[CV.line == l].pipe(lambda g: g.pred_blend_cal.sum() / g.y_tot.sum())),
                    ("back-test pred/actual 2024", BT[BT.line == l].pipe(lambda g: g.pred_blend_cal.sum() / g.y_tot.sum())),
                    ("level = 2024 experience only", LEVEL_2024[l]),
                    ("no new-business calibration", p.pred_uncal.sum() / base)]:
        unc.append((l, name, f))
unc = pd.DataFrame(unc, columns=["line", "scenario", "factor_vs_submitted"])
print(unc.round(3).pivot(index="scenario", columns="line", values="factor_vs_submitted").to_string())
unc.to_csv(OUT / "uncertainty.csv", index=False)

manifest = dict(
    script=Path(__file__).name, run_seconds=round(time.time() - T0, 1), smoke=SMOKE, seeds=SEEDS,
    data_md5={p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(EM_DIR.glob("*.parquet"))},
    trends_pa={l: round(float(np.exp(v) - 1), 4) for l, v in TRENDS.items()},
    blend_weights=WEIGHTS, nb_calibration={l: {str(k): v for k, v in d.items()} for l, d in NB_CAL.items()},
    cv_d2_testmix=cv_tab["ALL_testmix"].round(5).to_dict(), backtest_d2_testmix=bt_tab["ALL_testmix"].round(5).to_dict(),
    leak_features_dropped=sorted(set(LEAKY) | set(extra_leaks)),
    guardrail_overrides={c: REVIEWED_OVERRIDES[c] for c in flagged if c in REVIEWED_OVERRIDES},
    submission=dict(rows=len(sub), mean=float(sub.prediction.mean()), path=str(sub_path.name)))
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
log("done")
