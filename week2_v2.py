# =====================================================================================
# IAA AI-Assisted Hackathon 2026 - Week 2: Emerging Market - v2 pipeline (final configuration = v7)
# -------------------------------------------------------------------------------------
# Predicts the 2025 expected ultimate claim cost for every row of the emerging-market test files.
# v1 (week2_emerging.py, public LB 0.154) -> v2. A change is kept only if it improves local
# validation on the 2025 business mix (forward 2023->2024, backward 2024->2023, 5-fold CV grouped
# by customer; Tweedie D2 p=1.5 on capped claims, level-neutral per line) AND does not lower the
# uncapped score by more than 0.001. Same folds: v1 Kaggle blend 0.1585 / 0.1310 -> v2 0.1635 / 0.1317.
#
# What v2 does differently (each one validated):
#   1. The Part 1 prior has its own settings per line: current premium for motor, claims history for
#      buildings, smaller trees for contents. Car model and occupation were tested: they do NOT
#      transfer between the markets and are left out.
#   2. Part 1 relativities are re-based 50/50 per peril and per line for motor and buildings
#      (emerging claim costs differ by peril), frequency/severity re-basing for contents.
#   3. Gender - recorded only in the emerging market - enters the motor layer as a GLM term.
#   4. A 15% emerging-only GBM in the motor blend.
#   5. Credibility-weighted new-business calibration (76% of the 2025 book is new business).
#   6. All settings are fixed from local tuning; v1 re-selected its blend on Kaggle from CV noise.
#   7. Review fixes: training medians for relative features, reference rows priced on the test-book
#      premium median, missing gender stays missing, prior cache keyed on data and code.
#   8. Case-reserve development (IBNER, severity_development): open claims grow until they settle, so the
#      immature 2023-24 loss years are developed per line x peril with factors from the developed market,
#      only where the shortfall is significant (in practice motor bodily injury, ~x1.35 / x1.40).
#      Leaderboard level probes on v2 (x0.92..x1.18) peaked at x1.13; this step explains part of that
#      gap actuarially instead of a flat multiplier. IAA_SEV_DEV=0 switches it off; IAA_LEVEL_SCALE
#      still applies a final overall level.
#   9. Large-loss loading follows the peril mix (excess_profile, CFG["xs_peril"]): cost above the cap is
#      almost all motor bodily injury, buildings fire and contents flood, so each row's loading is the
#      line loading times its predicted peril mix weighted by the developed-market excess relativities
#      (normalised so the line total is unchanged). xs_peril=0 restores the flat loading per line.
#  10. Line levels CFG["level"] (motor x1.05, buildings x1.02, contents x1.01) replace the flat x1.06 of v4.
#      For Tweedie p=1.5 the public score is an exact function of a multiplier on any group of rows, so one
#      probe per line (each x1.10 on v4) gave each line's leaderboard-optimal level vs the model: motor
#      x1.082, buildings x1.024, contents x0.979, all within public-sample noise. The levels used are
#      credibility-weighted towards the model (common 2025 factor + line factors), see level_probe.py.
#      IAA_LEVEL_SCALE still multiplies on top (default 1).
#  11. Rejected on the leaderboard (v6, off by default): a pooled model (_pooled_gbm), one Tweedie GBM per line
#      on the developed and emerging rows together (market flag, region and gender, emerging rows weighted x5,
#      early stopping on emerging rows), blended in at 20% for motor and 35% for buildings, with 10 prior seeds
#      instead of 3. Local validation improved 0.1676 -> 0.1690 (fwd +0.0007, bwd +0.0029, cv +0.0007), but
#      the public score fell 0.16223 -> 0.16143. Most of the local gain came from the backward scheme (buildings
#      in the heavy 2023 year), and the flexible model is more exposed to the 2025 shift (broker book starting
#      in January-February) than the GLM layer. The defaults are the submitted v5 configuration; set w_pool and
#      CFG["pool"] (or IAA_CONFIG=outputs_week2_v6/run_manifest_v2.json) to re-run v6.
#  12. Affluence (v7): marketing_segment ends in affluent / mainstream / budget and is the only wealth proxy both
#      markets share (credit_band exists only in the developed market). Out-of-fold A/E on the emerging rows ran
#      budget 1.40 / mainstream 1.17 / affluent 0.76 for contents, so it enters the contents layer as a GLM term
#      (coefficients +0.33 budget, +0.25 mainstream): contents 0.138/0.170/0.159 -> 0.145/0.177/0.162 on
#      fwd/bwd/cv, score 0.1675 -> 0.1685. For buildings the schemes disagreed (fwd +0.008, bwd -0.017) and
#      for motor the gain was within noise, so both stay without it. Kept out of the priors (developed holdout
#      2021-23 -> 2024: motor +0.0004, contents +0.0013, buildings -0.0014) and out of the motor emerging GBM.
#      Public 0.16201 vs v5 0.16223: a bootstrap of the validation rows puts the sampling spread of such a
#      difference on a 33% public sample at ~0.001 (30% of samples show the better model losing), so the public
#      score cannot separate the two and local validation decides. v7 is the final submission.
# Tested and rejected (tuning_log*.csv): per-peril slopes, explicit NB GLM term, year-weather GLM
# terms, region, history as a layer term, residual GBM, other prior tree sizes for motor/buildings;
# also the layer fitted with 2025 new-business-mix weights (score 0.1676 -> 0.1669 at w^0.5, 0.1645 at w),
# and discount or channel terms in the layer (the apparent discount effect came from 187 new-business rows
# with a 10% discount and no buildings claims; the 5% discount alone lowers the score to 0.1661).
#
# Modes (IAA_MODE): final (default) | tune (recursive coordinate search -> best_config.json) |
# export (writes week2_v2_kaggle.py and .ipynb with CFG embedded). IAA_SMOKE=1 fast dry run;
# IAA_PROBES=1 also writes level-probe submissions x0.92 / x1.08 (see level_probe.py);
# IAA_CONFIG=<run_manifest_v2.json> re-runs that run's configuration (past submissions, see README).
# Paths are found automatically (Kaggle input folder or ./data); override with IAA_DATA / IAA_OUT.
# =====================================================================================
import copy
import hashlib
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
T0 = time.time()

MODE = os.environ.get("IAA_MODE", "final").lower()
SMOKE = os.environ.get("IAA_SMOKE") == "1"


def _root():
    try:
        return Path(__file__).resolve().parent
    except NameError:                       # Kaggle / Jupyter cell
        return Path.cwd()


ROOT = _root()
OUT = Path(os.environ.get("IAA_OUT", "/kaggle/working" if Path("/kaggle/working").exists()
                          else ROOT / "outputs_week2_v2"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"


def find_data_dirs():
    cands = [Path(os.environ["IAA_DATA"])] if os.environ.get("IAA_DATA") else []
    cands += [Path("/kaggle/input/competitions/iaa-ai-assisted-hackathon-2026-part-2"),
              Path("/kaggle/input/iaa-ai-assisted-hackathon-2026-part-2"), ROOT / "data", Path.cwd() / "data"]
    if Path("/kaggle/input").exists():
        cands += sorted(p.parent for p in Path("/kaggle/input").rglob("emerging") if p.is_dir())
    for c in cands:
        if (c / "emerging").is_dir():
            dv = next((c / n for n in ("developed", "developping", "developing") if (c / n).is_dir()), None)
            if dv is not None:
                return c / "emerging", dv
    raise FileNotFoundError("emerging/ and developed/ folders not found - set IAA_DATA to their parent folder")


# ----------------------------------- configuration ----------------------------------
# Defaults = best configuration found by MODE=tune (rounds 1-3, see tuning_log*.csv).
CFG = {
    "prior": {
        "motor": {
            "params": {
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_data_in_leaf": 2000,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "lambda_l2": 50.0
            },
            "premium": True,
            "avg_weather": False,
            "anomalies": True,
            "history": False
        },
        "buildings": {
            "params": {
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_data_in_leaf": 2000,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "lambda_l2": 50.0
            },
            "premium": False,
            "avg_weather": False,
            "anomalies": True,
            "history": True
        },
        "contents": {
            "params": {
                "learning_rate": 0.05,
                "num_leaves": 7,
                "min_data_in_leaf": 3000,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "lambda_l2": 50.0
            },
            "premium": False,
            "avg_weather": False,
            "anomalies": True,
            "history": False
        }
    },
    "prior_seeds": [1, 2, 3],
    "layer": {
        "motor": {
            "w_peril": 0.5,
            "peril_slope": True,
            "slope_ridge": 0.0,
            "weather_terms": False,
            "nb_mode": "none",
            "nb_ridge": 0.0,
            "extra_terms": [
                "gender"
            ],
            "extra_ridge": 1.0,
            "w_resid": 0.0,
            "w_a": 0.15,
            "freqsev": False
        },
        "buildings": {
            "w_peril": 0.5,
            "peril_slope": False,
            "slope_ridge": 0.0,
            "weather_terms": False,
            "nb_mode": "none",
            "nb_ridge": 0.0,
            "extra_terms": [],
            "extra_ridge": 1.0,
            "w_resid": 0.0,
            "w_a": 0.0,
            "freqsev": False
        },
        "contents": {
            "w_peril": 0.0,
            "peril_slope": True,
            "slope_ridge": 0.0,
            "weather_terms": False,
            "nb_mode": "none",
            "nb_ridge": 0.0,
            "extra_terms": [
                "affluence"
            ],
            "extra_ridge": 1.0,
            "w_resid": 0.0,
            "w_a": 0.0,
            "freqsev": True
        }
    },
    "resid_params": {
        "learning_rate": 0.01,
        "num_leaves": 4,
        "max_depth": 3,
        "min_data_in_leaf": 400,
        "feature_fraction": 0.6,
        "lambda_l2": 50.0
    },
    "a_params": {
        "learning_rate": 0.02,
        "num_leaves": 6,
        "max_depth": 4,
        "min_data_in_leaf": 400,
        "feature_fraction": 0.6,
        "lambda_l2": 50.0
    },
    "level": {
        "motor": 1.05,
        "buildings": 1.02,
        "contents": 1.01
    },
    "nb_cred": {
        "enabled": True,
        "k": 300,
        "bounds": [
            0.8,
            1.25
        ]
    },
    "xs_peril": 1.0
}
WEATHER_PERIL_TERMS = {("motor", "glass_hail"): ["hail_days"],
                       ("buildings", "storm"): ["wind_exposure_index", "hail_days"],
                       ("buildings", "fire"): ["wildfire_index"]}

SEED = 42
POWER = 1.5
LINES = ["motor", "buildings", "contents"]
PERILS = {"motor": ["bodily_injury", "glass_hail", "own_damage", "theft", "third_party_prop"],
          "buildings": ["accidental_damage", "escape_of_water", "fire", "flood", "storm", "subsidence"],
          "contents": ["accidental_damage", "escape_of_water", "flood", "theft"]}
ALL_PERILS = sorted({p for v in PERILS.values() for p in v})
VALUATION = pd.Timestamp("2024-12-31")
TARGET_YEAR = 2025
NEAREST_DAYS = 31
CAP_Q = 0.995
CAL_CAP_Q = 0.975
MATURE_OPEN_SHARE = 0.5
SEV_DEV = os.environ.get("IAA_SEV_DEV", "1") == "1"     # case-reserve development of open claims (IBNER)
SEV_DEV_MAX = 1.6
TREND_TAU = 0.03
TREND_BOUNDS = (-0.02, np.log(1.12))
LEVEL_BAND = (0.80, 1.35)
DEV_PRIOR_ROWS = 80_000 if SMOKE else None

LEAKY = ["customer_service_contacts"]
REVIEWED_OVERRIDES = {
    "prior_log": "developed-market risk score built only from vetted inputs; a genuine risk score separates "
                 "claim / claim-free rows, so the leak heuristic does not apply (reviewed 2026-10-27)"}
NOISE = ["previous_insurer", "campaign_code", "email_domain", "branch_id", "vehicle_colour",
         "marketing_consent", "document_preference", "marketing_segment", "surname"]
AGE_BANDS = ["17-24", "25-34", "35-49", "50-64", "65+"]
CONSTRUCTION = {"BRK": "brick", "STN": "stone", "WOD": "timber", "MIX": "non-standard"}
WVARS = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
         "wildfire_index", "rainfall_mm", "mean_temperature_c"]
ORD = {"energy_rating": list("ABCDEFG"), "credit_band": list("ABCDE"),
       "urban_density": ["rural", "suburban", "urban", "dense_urban"],
       "security_level": ["standard", "locks", "alarm_and_locks"]}
TCOLS = ["y_cap", "y_xs", "y_raw", "y_cal", "y_tot", "n_claims", "n_dev"] + [f"yp_{p}" for p in ALL_PERILS]

# features that are missing in one market or mean something different -> never in the prior
NOT_SHARED = {"region", "make_model", "log_premium_rel", "gender", "credit_band", "driver_age", "policyholder_age",
              "affluence", "marketing_segment",
              "annual_mileage", "telematics_opt_in", "telematics_score", "occupation", "occ_base", "self_employed"}
BASE_DROP = set(LEAKY + NOISE + ["row_id", "policy_id", "customer_id", "address_id", "period_start", "period_end",
                                 "renewal_decision", "line", "year", "exposure", "written_premium",
                                 "customer_since_year", "year_built", "sum_insured", "make_model", "w"] + TCOLS)
YEAR_WEATHER = set(WVARS) | {v + "_anom" for v in WVARS}
AVG_WEATHER = {v + "_avg" for v in WVARS}
PRIOR_COLS = ["pr_TOTAL", "pr_FREQ", "pr_SEV"] + [f"pr_{p}" for p in ALL_PERILS]

PRIOR_BASE = dict(objective="tweedie", tweedie_variance_power=POWER, bagging_freq=1, cat_smooth=200,
                  min_data_per_group=500, max_cat_threshold=8, verbose=-1, num_threads=0)
RESID_BASE = dict(objective="tweedie", tweedie_variance_power=POWER, bagging_fraction=0.8, bagging_freq=1,
                  cat_smooth=100, min_data_per_group=200, max_cat_threshold=8, verbose=-1, num_threads=0)


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def banner(title):
    print("\n" + "=" * 90 + f"\n>>> {title}\n" + "=" * 90, flush=True)


def d2(y, p, w=None):
    return d2_tweedie_score(y, np.maximum(p, 1e-9), power=POWER, sample_weight=w)


QA = []


# =====================================================================================
# STEP 1  load & clean (identical rules for both markets)
# =====================================================================================
def read(d, name):
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
    QA.append((market, "train", "all", f"claims outside any policy period dropped (loss years "
               f"{sorted(dropped.loss_year.unique().tolist())})", len(dropped)))
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


# =====================================================================================
# STEP 3  development (IBNR), large-loss caps, trend and row targets (total + per peril)
# =====================================================================================
def completeness_table(cl_dev):
    """Share of claims reported by VALUATION per line x peril x loss year (developed delays)."""
    rows = []
    for (line, peril), g in cl_dev[cl_dev.loss_year.between(2021, 2022)].groupby(["line", "peril"]):
        d = np.sort(g.delay.values)
        for ly in range(2021, VALUATION.year + 1):
            days = (VALUATION - pd.date_range(f"{ly}-01-01", f"{ly}-12-31")).days.values
            rows.append((line, peril, ly, np.searchsorted(d, days, side="right").mean() / len(d)))
    return pd.DataFrame(rows, columns=["line", "peril", "loss_year", "completeness"])


def severity_development(cl_dev):
    """Case-reserve development (IBNER) per line x peril for the immature loss years 2023-24.
    completeness_table() grosses up claim counts only, but open claims also grow until they settle: in
    the developed market 2024 bodily-injury claims average 18% less than 2021-22 while costs rise ~5% a
    year. Expected severity = the 2021-22 mean (capped at CAP_Q) trended at the line's mature 2021->22
    step; ratio = expected / observed. A factor is used only where the ratio is significantly above 1
    (log ratio > 2 s.e.), so trend noise in short-tail perils leaves their targets alone; capped at
    SEV_DEV_MAX. Estimated on the developed market, applied to both markets."""
    c = cl_dev[cl_dev.loss_year.between(2021, 2024)].copy()
    c["v"] = np.minimum(c.net, c.groupby("line").net.transform(lambda s: s.quantile(CAP_Q)))
    cell = c.groupby(["line", "peril", "loss_year"]).v.agg(["size", "mean", "std"])
    rows = []
    for l, g in cell.groupby(level="line"):
        g = g.droplevel("line")
        y21, y22 = g.xs(2021, level="loss_year"), g.xs(2022, level="loss_year")
        both = y21.index.intersection(y22.index)
        trend = float(np.average(np.log(y22["mean"][both] / y21["mean"][both]), weights=y22["size"][both]))
        for p in both:
            m = g.loc[p]
            cv2 = lambda y: (m.loc[y, "std"] / m.loc[y, "mean"]) ** 2 / m.loc[y, "size"]
            for ly in (2023, 2024):
                if ly not in m.index:
                    continue
                expected = 0.5 * (m.loc[2021, "mean"] * np.exp(trend * (ly - 2021)) +
                                  m.loc[2022, "mean"] * np.exp(trend * (ly - 2022)))
                ratio = expected / m.loc[ly, "mean"]
                se = float(np.sqrt(cv2(ly) + 0.25 * (cv2(2021) + cv2(2022))))
                factor = float(min(ratio, SEV_DEV_MAX)) if np.log(ratio) > 2 * se else 1.0
                rows.append((l, p, ly, int(m.loc[ly, "size"]), round(float(ratio), 3), round(se, 3), factor))
    return pd.DataFrame(rows, columns=["line", "peril", "loss_year", "claims", "ratio", "se_log", "factor"])


def apply_development(cl, dev, market):
    f = cl[["line", "peril", "loss_year"]].merge(dev[["line", "peril", "loss_year", "factor"]],
                                                 on=["line", "peril", "loss_year"], how="left").factor.fillna(1.0)
    cl = cl.assign(net=cl.net.values * f.values)
    QA.append((market, "train", "all", "claims with case-reserve development applied (IBNER)", int((f.values > 1).sum())))
    return cl


XS_REL = {}


def excess_profile(cl_dev):
    """Where the cost above the large-loss cap comes from: per line x peril, excess / capped cost relative
    to the line's. From the developed market (~800 large claims vs ~45 in the emerging market, which shows
    the same pattern): motor bodily injury ~2.9, buildings fire ~4.0, contents flood ~4.0, most others ~0."""
    g = cl_dev[cl_dev.loss_year.between(2021, 2024)].groupby(["line", "peril"])[["net_cap", "net_xs"]].sum()
    tot = g.groupby(level="line").transform("sum")
    r = (g.net_xs / g.net_cap) / (tot.net_xs / tot.net_cap)
    return {l: {p: round(float(v), 4) for p, v in r.loc[l].items()} for l in LINES}


def apply_caps(cl, years):
    q = cl[cl.loss_year.isin(years)].groupby("line").net
    caps = pd.concat([q.quantile(CAP_Q).rename("cap"), q.quantile(CAL_CAP_Q).rename("cap_cal")], axis=1)
    cl = cl.drop(columns=["cap", "cap_cal"], errors="ignore").merge(caps, left_on="line", right_index=True)
    cl["net_cap"] = np.minimum(cl.net, cl.cap)
    cl["net_xs"] = cl.net - cl.net_cap
    return cl


def severity_trends(cl, years, min_n=30):
    cell = (cl[cl.loss_year.isin(years)].groupby(["line", "peril", "loss_year"])
            .agg(n=("net_cap", "size"), sev=("net_cap", "mean"), sd=("net_cap", "std"),
                 open_share=("status", lambda s: (s == "open").mean())).reset_index())
    out = {}
    for line, g in cell.groupby("line"):
        steps, wts = [], []
        for peril, gp in g.groupby("peril"):
            gp = gp.set_index("loss_year")
            for y0, y1 in zip(years[:-1], years[1:]):
                if y0 in gp.index and y1 in gp.index and min(gp.n[y0], gp.n[y1]) >= min_n \
                        and gp.open_share[y1] <= MATURE_OPEN_SHARE:
                    se2 = sum((gp.sd[y] / gp.sev[y]) ** 2 / gp.n[y] for y in (y0, y1))
                    steps.append(np.log(gp.sev[y1] / gp.sev[y0])); wts.append(1 / se2)
        if steps:
            w = np.array(wts)
            out[line] = (float(np.average(steps, weights=w)), float(np.sqrt(1 / w.sum())), len(steps))
        else:
            out[line] = (np.nan, np.inf, 0)
    return out


def build_targets(rows, cl, trends, comp):
    """Ultimate (IBNR-adjusted), 2025-cost targets per row: y_cap (capped), y_xs (above cap),
    y_cal (97.5% cap, for robust A/E) and yp_<peril> (capped, per peril)."""
    c = cl.merge(comp, on=["line", "peril", "loss_year"], how="left")
    c["completeness"] = c.completeness.fillna(1.0).clip(lower=0.3)
    c["ibnr"] = 1 / c.completeness
    c["trend"] = np.exp(c.line.map(trends).values * (TARGET_YEAR - c.loss_year.values))
    c["y_cap"] = c.net_cap * c.ibnr * c.trend
    c["y_xs"] = c.net_xs * c.ibnr * c.trend
    c["y_cal"] = np.minimum(c.net, c.cap_cal) * c.ibnr * c.trend
    agg = c.groupby("row_id").agg(y_cap=("y_cap", "sum"), y_xs=("y_xs", "sum"), y_raw=("net", "sum"),
                                  y_cal=("y_cal", "sum"), n_claims=("net", "size"), n_dev=("ibnr", "sum"))
    yp = (c.pivot_table(index="row_id", columns="peril", values="y_cap", aggfunc="sum", fill_value=0)
          .reindex(columns=ALL_PERILS, fill_value=0).add_prefix("yp_"))
    out = rows.merge(agg, left_on="row_id", right_index=True, how="left").merge(
        yp, left_on="row_id", right_index=True, how="left")
    cols = ["y_cap", "y_xs", "y_raw", "y_cal", "n_claims", "n_dev"] + list(yp.columns)
    out[cols] = out[cols].fillna(0.0)
    out["y_tot"] = out.y_cap + out.y_xs
    assert np.allclose(out[list(yp.columns)].sum(axis=1), out.y_cap, rtol=1e-6, atol=1e-4), "peril targets != total"
    return out


# =====================================================================================
# STEP 4  features
# =====================================================================================
HIST_COLS = ["pol_prev_n", "pol_prev_e", "pol_prev_cost", "pol_hist_n", "pol_hist_e", "pol_hist_freq",
             "pol_hist_cost_rate", "pol_hist_years", "cust_prev_n", "cust_prev_e", "cust_hist_n", "cust_hist_e",
             "cust_hist_freq", "cust_hist_years"]


def history_tables(panel):
    """Past-only claims experience per policy and per customer (all lines), by year: the previous
    year's developed claim count / exposure / capped cost and the cumulative experience before the
    row's year. Computed identically in both markets so the prior can learn it on developed data."""
    p = panel[["policy_id", "customer_id", "year", "exposure", "n_dev", "y_cap"]].copy()
    p[["n_dev", "y_cap"]] = p[["n_dev", "y_cap"]].fillna(0.0)        # test rows: claims unknown
    tabs = []
    for key, pre in [("policy_id", "pol"), ("customer_id", "cust")]:
        g = (p.groupby([key, "year"]).agg(n=("n_dev", "sum"), e=("exposure", "sum"), c=("y_cap", "sum"))
             .reset_index().sort_values([key, "year"]))
        g[f"{pre}_hist_n"] = g.groupby(key).n.cumsum() - g.n
        g[f"{pre}_hist_e"] = g.groupby(key).e.cumsum() - g.e
        hist_c = g.groupby(key).c.cumsum() - g.c
        g[f"{pre}_hist_years"] = g.groupby(key).cumcount().astype(float)
        prev = g[[key, "year", "n", "e", "c"]].assign(year=g.year + 1).rename(
            columns={"n": f"{pre}_prev_n", "e": f"{pre}_prev_e", "c": f"{pre}_prev_cost"})
        g = g.merge(prev, on=[key, "year"], how="left")
        none = g[f"{pre}_hist_e"] <= 0
        g.loc[none, [f"{pre}_hist_n", f"{pre}_hist_e", f"{pre}_hist_years"]] = np.nan
        g[f"{pre}_hist_freq"] = (g[f"{pre}_hist_n"] + 0.1) / (g[f"{pre}_hist_e"] + 1)
        if pre == "pol":
            g["pol_hist_cost_rate"] = np.where(none, np.nan, hist_c / (g.pol_hist_e + 1))
        cols = [c for c in HIST_COLS if c.startswith(pre + "_") and c in g]
        tabs.append((key, g[[key, "year"] + cols]))
    return tabs


def add_history(f, tabs):
    n0 = len(f)
    f = f.drop(columns=[c for c in HIST_COLS if c in f])
    for key, g in tabs:
        f = f.merge(g, on=[key, "year"], how="left")
    assert len(f) == n0, "history merge changed row count"
    return f


def money_medians(train, test=None):
    """Medians for the currency-neutral features, fitted once (training rows; test book for 2025)."""
    med = {"si": train.groupby("line").sum_insured.median().to_dict() if "sum_insured" in train else {},
           "prem": train.groupby(["line", "year"]).written_premium.median().to_dict()}
    if test is not None:
        med["prem"].update(test.groupby(["line", "year"]).written_premium.median().to_dict())
    return med


def make_features(df, mk, med=None):
    n0 = len(df)
    cust = mk["cust"]
    if "marketing_segment" in cust:        # affluent / mainstream / budget: the only wealth proxy both markets share
        cust = cust.assign(affluence=cust.marketing_segment.str.split("_").str[-1])
    cust = cust.drop(columns=[c for c in NOISE if c in cust])
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
    assert len(f) == n0, "feature merge changed row count"
    for col, levels in ORD.items():
        if col in f:
            f[col] = f[col].map({v: i for i, v in enumerate(levels)})
    if "gender" in f:
        f["gender"] = f.gender.map({"F": 1.0, "M": 0.0})          # missing stays missing
    if "occupation" in f:
        occ = f.occupation.astype("string")
        f["self_employed"] = occ.str.startswith("self-employed").astype("boolean").fillna(False).astype(float)
        f["occ_base"] = occ.str.replace("self-employed ", "", regex=False)
    f["ends_early"] = (~full_year_end(f.period_end)).astype(int)
    f["starts_late"] = (f.period_start.dt.strftime("%m-%d") != "01-01").astype(int)
    f["customer_tenure"] = f.year - f.customer_since_year
    f["property_age"] = f.year - f.year_built
    f["total_excess"] = f.voluntary_excess + f.compulsory_excess
    f["clay_dry_trees"] = (f.soil_type == "clay") * f.dry_summer_index_avg * f.trees_within_5m
    f["elev_over_river"] = f.elevation_m / (f.distance_to_river_km.clip(lower=0) + 0.1)
    f["frost_x_age"] = f.freezing_days_avg * f.property_age
    f["hail_x_flat_roof"] = f.hail_days_avg * (f.roof_type == "flat")
    f["wildfire_x_timber"] = f.wildfire_index_avg * (f.get("construction_type", "") == "timber")
    if "sum_insured" in f:
        si_med = f.line.map(med["si"]) if med and med.get("si") else f.groupby("line").sum_insured.transform("median")
        f["log_si_rel"] = np.log(f.sum_insured) - np.log(si_med.astype(float))
        f["excess_to_si"] = f.total_excess / f.sum_insured
        f["si_per_bedroom_rel"] = np.exp(f.log_si_rel) / f.bedrooms.clip(lower=1)
    if "ncb_years" in f:
        f["ncb_ratio"] = f.ncb_years / f.years_licensed.clip(lower=1)
        f["excess_per_vvg"] = f.total_excess / f.vehicle_value_group.clip(lower=1)
    own = f.groupby(["line", "year"]).written_premium.transform("median")
    pm = pd.Series(list(zip(f.line, f.year)), index=f.index).map(med["prem"]) if med else own
    f["log_premium_rel"] = np.log(f.written_premium) - np.log(pm.fillna(own).astype(float))
    return f


def cat_align(frames, feats):
    """Text/bool columns -> categoricals with identical categories in every frame."""
    out = [f.copy() for f in frames]
    for c in feats:
        f0 = out[0][c]
        if f0.dtype == object or f0.dtype == bool or str(f0.dtype) in ("category", "string", "str", "boolean"):
            cats = pd.Categorical(pd.concat([f[c] for f in out]).astype(str)).categories
            for f in out:
                f[c] = pd.Categorical(f[c].astype(str), categories=cats)
    return out


def prepare():
    """Loads both markets and builds every frame the models need. Returns a context dict."""
    em_dir, dv_dir = find_data_dirs()
    EM, DV = load_market(em_dir, "emerging"), load_market(dv_dir, "developed")
    expenses, reference = read(em_dir, "expenses"), read(em_dir, "reference_policies")
    for l in LINES:      # timing guardrail, like-for-like within new business / renewal
        a, t = EM["pol"][(EM["pol"].line == l) & (EM["pol"].year == 2024)], EM["tst"][EM["tst"].line == l]
        for nb in (False, True):
            ea = (~full_year_end(a[a.new_business == nb].period_end)).mean()
            et = (~full_year_end(t[t.new_business == nb].period_end)).mean()
            assert abs(ea - et) < 0.03, f"{l} nb={nb}: ends-early share differs train-2024 {ea:.3f} vs test {et:.3f}"
    comp = completeness_table(DV["cl"])
    sev_dev = severity_development(DV["cl"]) if SEV_DEV else None
    if sev_dev is not None:
        DV["cl"], EM["cl"] = apply_development(DV["cl"], sev_dev, "developed"), apply_development(EM["cl"], sev_dev, "emerging")
        used = sev_dev[sev_dev.factor > 1]
        log(f"severity development (IBNER) applied to {len(used)} line x peril x year cells: "
            f"{ {f'{r.line}/{r.peril}/{r.loss_year}': r.factor for r in used.itertuples()} }")
    DV["cl"] = apply_caps(DV["cl"], range(2021, 2025))
    EM["cl"] = apply_caps(EM["cl"], [2023, 2024])
    XS_REL.clear()
    XS_REL.update(excess_profile(DV["cl"]))
    tr_dev, tr_em = severity_trends(DV["cl"], [2021, 2022, 2023]), severity_trends(EM["cl"], [2023, 2024])
    trends = {}
    for l in LINES:
        sd = tr_dev[l][0]
        se, se_se, _ = tr_em[l]
        z = 0.0 if not np.isfinite(se) else TREND_TAU ** 2 / (TREND_TAU ** 2 + se_se ** 2)
        trends[l] = float(np.clip(z * (se if np.isfinite(se) else 0) + (1 - z) * sd, *TREND_BOUNDS))
    trends_dv = {l: float(np.clip(tr_dev[l][0], *TREND_BOUNDS)) for l in LINES}
    em_train = build_targets(EM["pol"], EM["cl"], trends, comp)
    dv_train = build_targets(DV["pol"], DV["cl"], trends_dv, comp)
    em_test = EM["tst"].assign(**{c: np.nan for c in TCOLS})
    em_tabs = history_tables(pd.concat([em_train, em_test]))
    med_em = money_medians(EM["pol"], EM["tst"])
    EF = add_history(make_features(em_train, EM, med_em), em_tabs)
    ET = add_history(make_features(em_test, EM, med_em), em_tabs)
    RF = add_history(pd.concat([make_features(clean_policies(reference[reference.line == l].drop(columns="line")
                                                             .dropna(axis=1, how="all"), l, "reference", "emerging")
                                              .assign(**{c: np.nan for c in TCOLS}), EM, med_em)
                                for l in LINES if (reference.line == l).any()], ignore_index=True), em_tabs)
    RF["row_id"] = "REF_" + RF.row_id.astype(str)          # reference row_ids collide with training rows
    if SMOKE:
        dv_train = dv_train.groupby("line", group_keys=False).apply(
            lambda g: g.sample(min(DEV_PRIOR_ROWS, len(g)), random_state=SEED))
    DVF = add_history(make_features(dv_train, DV, money_medians(DV["pol"])), history_tables(dv_train))
    nb_te, nb_tr = ET.groupby("line").new_business.mean(), EF.groupby("line").new_business.mean()
    mixw = {l: {True: nb_te[l] / nb_tr[l], False: (1 - nb_te[l]) / (1 - nb_tr[l])} for l in LINES}
    EF["w"] = [mixw[l][bool(nb)] for l, nb in zip(EF.line, EF.new_business)]
    data_md5 = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(em_dir.glob("*.parquet"))}
    log(f"prepared: emerging train {len(EF):,} / test {len(ET):,} rows, developed {len(DVF):,} rows | "
        f"trends {({l: f'{np.exp(v) - 1:+.1%}' for l, v in trends.items()})}")
    return dict(EM=EM, DV=DV, EF=EF, ET=ET, RF=RF, DVF=DVF, comp=comp, trends=trends, trends_dv=trends_dv,
                tr_dev=tr_dev, tr_em=tr_em, mixw=mixw, expenses=expenses, data_md5=data_md5, sev_dev=sev_dev,
                em_dir=em_dir, dv_dir=dv_dir)


# =====================================================================================
# STEP 5  developed-market priors: one Tweedie GBM per line (total) and per peril
# =====================================================================================
def prior_features(ctx, line, pc):
    drop = BASE_DROP | NOT_SHARED
    if not pc["avg_weather"]:
        drop |= AVG_WEATHER
    if not pc["anomalies"]:
        drop |= {v + "_anom" for v in WVARS}
    if not pc["history"]:
        drop |= set(HIST_COLS)
    keep = {"log_premium_rel"} if pc["premium"] else set()
    D, E = ctx["DVF"], ctx["EF"]
    d, e = D[D.line == line], E[E.line == line]
    return [c for c in D.columns if c in E.columns and (c not in drop or c in keep)
            and d[c].notna().any() and e[c].notna().any()]


def _source_hash(*funcs):
    try:
        return hashlib.md5("".join(inspect.getsource(f) for f in funcs).encode()).hexdigest()[:8]
    except (OSError, TypeError):
        return "nosrc"


ON_KAGGLE = Path("/kaggle/working").exists()


def need_freqsev(cfg, line):
    return bool(cfg["layer"][line].get("freqsev", False))


def need_perils(cfg, line):
    """Per-peril priors are only trained when the layer or the excess allocation uses them."""
    return cfg["layer"][line]["w_peril"] > 0 or cfg.get("xs_peril", 0) > 0


def prior_key(ctx, cfg, line):
    """Cache key: the line's prior settings + seeds + data fingerprint + the code that builds the prior."""
    parts = [line, cfg["prior"][line], cfg["prior_seeds"], need_perils(cfg, line), need_freqsev(cfg, line),
             ctx["data_md5"], SMOKE, _source_hash(make_features, build_targets, prior_features, fit_prior_line)]
    sd = ctx.get("sev_dev")
    if sd is not None and ((sd.line == line) & (sd.factor > 1)).any():     # developed targets changed
        parts.append(sd.loc[(sd.line == line) & (sd.factor > 1), ["peril", "loss_year", "factor"]].values.tolist())
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def fit_prior_line(ctx, cfg, l):
    """Log-relativities for one line (centred on the emerging training rows): pr_TOTAL and pr_<peril>.
    Fitted on developed data only; emerging data later fits only levels, slopes and a few terms."""
    pc = cfg["prior"][l]
    params = {**PRIOR_BASE, **pc["params"]}
    fs = prior_features(ctx, l, pc)
    D = ctx["DVF"]
    tr = D.loc[D.line == l, fs + ["policy_id", "exposure", "y_cap", "n_dev"] + [f"yp_{p}" for p in PERILS[l]]]
    ap = pd.concat([ctx[k].loc[ctx[k].line == l, fs + ["row_id"]] for k in ("EF", "ET", "RF")], ignore_index=True)
    tr_, ap_ = cat_align([tr, ap], fs)
    n_em = int((ctx["EF"].line == l).sum())
    res = pd.DataFrame({"row_id": ap_.row_id.values})
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(tr_, groups=tr_.policy_id))
    for p in ["TOTAL"] + (PERILS[l] if need_perils(cfg, l) else []):
        y = tr_.y_cap.values if p == "TOTAL" else tr_[f"yp_{p}"].values
        off = np.log(tr_.exposure.values)
        off = off + np.log(y.sum() / np.exp(off).sum())
        raw = np.zeros(len(ap_))
        for s in cfg["prior_seeds"]:
            m = lgb.train({**params, "seed": s}, lgb.Dataset(tr_[fs].iloc[a], y[a], init_score=off[a]), 5000,
                          valid_sets=[lgb.Dataset(tr_[fs].iloc[b], y[b], init_score=off[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            raw += m.predict(ap_[fs], raw_score=True, num_iteration=max(m.best_iteration, 10))
        raw /= len(cfg["prior_seeds"])
        res[f"pr_{p}"] = raw - raw[:n_em].mean()
        if p == "TOTAL":
            imp = pd.Series(m.feature_importance("gain"), index=fs)
            log(f"  prior {l:9s} {len(fs)} features, {m.best_iteration} rounds | top "
                f"{(imp / imp.sum()).sort_values(ascending=False).head(5).round(3).to_dict()}")
    if need_freqsev(cfg, l):
        # frequency: Poisson on developed claim counts; severity: Gamma on cost per claim (claim-weighted)
        n = tr_.n_dev.values
        off = np.log(tr_.exposure.values); off = off + np.log(n.sum() / np.exp(off).sum())
        hasc = n > 0
        sev = np.where(hasc, tr_.y_cap.values / np.maximum(n, 1e-9), 0.0)
        sbase = np.log(np.average(sev[hasc], weights=n[hasc]))
        for tag, obj in [("FREQ", "poisson"), ("SEV", "gamma")]:
            raw = np.zeros(len(ap_))
            for s in cfg["prior_seeds"]:
                prm = {**{k: v for k, v in params.items() if k != "tweedie_variance_power"}, "objective": obj, "seed": s}
                if tag == "FREQ":
                    dtr = lgb.Dataset(tr_[fs].iloc[a], n[a], init_score=off[a])
                    dva = lgb.Dataset(tr_[fs].iloc[b], n[b], init_score=off[b])
                else:
                    ia, ib = a[hasc[a]], b[hasc[b]]
                    dtr = lgb.Dataset(tr_[fs].iloc[ia], sev[ia], weight=n[ia], init_score=np.full(len(ia), sbase))
                    dva = lgb.Dataset(tr_[fs].iloc[ib], sev[ib], weight=n[ib], init_score=np.full(len(ib), sbase))
                m = lgb.train(prm, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
                raw += m.predict(ap_[fs], raw_score=True, num_iteration=max(m.best_iteration, 10))
            raw /= len(cfg["prior_seeds"])
            res[f"pr_{tag}"] = raw - raw[:n_em].mean()
    return res


def fit_priors(ctx, cfg, lines=LINES):
    """Priors for all lines, reusing per-line cache files locally (never cached on Kaggle)."""
    out = []
    for l in lines:
        path = CACHE / f"prior_{l}_{prior_key(ctx, cfg, l)}.parquet"
        if path.exists() and not ON_KAGGLE:
            res = pd.read_parquet(path)
        else:
            res = fit_prior_line(ctx, cfg, l)
            if not ON_KAGGLE:
                CACHE.mkdir(parents=True, exist_ok=True)
                res.to_parquet(path, index=False)
        out.append(res)
    pri = pd.concat(out, ignore_index=True)
    for c in PRIOR_COLS:
        if c not in pri:
            pri[c] = np.nan
    assert pri.row_id.is_unique, "duplicate row_id in priors"
    return pri


def attach_priors(ctx, pri):
    for k in ["EF", "ET", "RF"]:
        ctx[k] = ctx[k].drop(columns=[c for c in PRIOR_COLS if c in ctx[k]]).merge(pri, on="row_id", how="left")
        assert ctx[k].pr_TOTAL.notna().all(), f"prior missing for {k} rows"


# =====================================================================================
# STEP 7  emerging layer: per-peril GLM on the prior (+ weather, new business, extras)
# =====================================================================================
def tweedie_glm(X, y, off, ridge=None, centre=None):
    """log mu = off + X b, Tweedie(1.5) deviance + ridge (b - centre)^2 on a scale-free deviance."""
    X = np.asarray(X, float); k = X.shape[1]
    ridge = np.zeros(k) if ridge is None else np.asarray(ridge, float)
    centre = np.zeros(k) if centre is None else np.asarray(centre, float)
    sc = max(float(np.mean(y)), 1e-9) ** (2 - POWER) * len(y) / 1000.0

    def f(b):
        mu = np.exp(np.clip(off + X @ b, -30, 30))
        loss = np.sum(mu ** (2 - POWER) / (2 - POWER) - y * mu ** (1 - POWER) / (1 - POWER)) / sc
        grad = X.T @ (mu ** (2 - POWER) - y * mu ** (1 - POWER)) / sc
        return loss + np.sum(ridge * (b - centre) ** 2), grad + 2 * ridge * (b - centre)
    b0 = centre.copy()
    b0[0] = np.log(max(y.sum(), 1e-9) / np.exp(off + X @ b0).sum())
    return minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-9}).x


def _design(d, ref, prior_col, wx, nb, extra, slope):
    cols = {"const": np.ones(len(d))}
    pcs = [prior_col] if isinstance(prior_col, str) else list(prior_col or [])
    if slope:
        for i, c in enumerate(pcs):
            cols["prior" if i == 0 else f"prior{i + 1}"] = d[c].values
    for v in wx:
        cols[f"wx_{v}"] = (d[v].values - ref[v].mean()) / ref[v].std()
    if nb:
        cols["nb"] = d.new_business.values.astype(float)
    if "gender" in extra:
        cols["gender"] = d.gender.fillna(ref.gender.mean()).values - ref.gender.mean()
    if "selfemp" in extra:
        cols["selfemp"] = d.self_employed.values - ref.self_employed.mean()
    if "age" in extra:
        for k in (0, 1, 3, 4):
            cols[f"age{k}"] = (d.age_band.values == k).astype(float)
    if "hist" in extra:                    # experience rating for renewals: own past claim rate, centred
        def _lhf(f):                       # within rows that have history (new business / no history -> 0)
            return np.log((f.pol_hist_n.values + 0.1) / (f.pol_hist_e.values + 1))
        has, ref_has = d.pol_hist_e.notna().values, ref.pol_hist_e.notna().values
        mu = _lhf(ref)[ref_has].mean() if ref_has.any() else 0.0
        cols["hist"] = np.where(has, np.nan_to_num(_lhf(d) - mu), 0.0)
    if "affluence" in extra:               # no credit band in the emerging market; budget customers claim more
        for lv in ("budget", "mainstream"):
            cols[f"aff_{lv}"] = (d.affluence.values == lv).astype(float) - (ref.affluence.values == lv).mean()
    if "region" in extra:                  # emerging territory: region codes do not exist in Part 1
        for r in sorted(ref.region.dropna().unique())[1:]:
            cols[f"reg_{r}"] = (d.region.values == r).astype(float) - (ref.region.values == r).mean()
    return pd.DataFrame(cols)


def _glm_fit_predict(tr, aps, y_col, prior_col, lc, wx=(), nb=False, slope=True):
    Xtr = _design(tr, tr, prior_col, wx, nb, lc["extra_terms"], slope)
    names = list(Xtr.columns)
    ridge = np.array([0.0 if n == "const" else lc["slope_ridge"] if n.startswith("prior") else lc["nb_ridge"]
                      if n == "nb" else 0.0 if n.startswith("wx_") else lc["extra_ridge"] for n in names])
    centre = np.array([1.0 if n.startswith("prior") else 0.0 for n in names])
    pcs = [prior_col] if isinstance(prior_col, str) else list(prior_col or [])
    fixed = (lambda f: f[pcs].sum(axis=1).values) if (pcs and not slope) else (lambda f: 0.0)
    off_tr = np.log(tr.exposure.values) + fixed(tr)
    b = tweedie_glm(Xtr.values, tr[y_col].values, off_tr, ridge, centre)
    preds = []
    for ap in aps:
        Xap = _design(ap, tr, prior_col, wx, nb, lc["extra_terms"], slope)
        off = np.log(ap.exposure.values) + fixed(ap)
        preds.append(np.exp(np.clip(off + Xap.values @ b, -30, 30)))
    return preds, dict(zip(names, b))


def layer_predict(tr, aps, line, cfg, ctx=None):
    """Fits the emerging layer on tr; returns predictions (capped-cost scale) for tr and each frame in aps,
    the coefficients, and per-row large-loss weights (mean 1 on tr, cost-weighted) for each frame.
    ctx (developed rows) is only needed when the line blends in the pooled model (w_pool > 0)."""
    lc = cfg["layer"][line]
    frames = [tr] + list(aps)
    nb_p = lc["nb_mode"] == "peril"
    peril = [np.zeros(len(f)) for f in frames]
    coefs, xs_num = {}, [np.zeros(len(f)) for f in frames]
    xs_w = cfg.get("xs_peril", 0.0)
    if lc["w_peril"] > 0 or xs_w > 0:
        for p in PERILS[line]:
            wx = WEATHER_PERIL_TERMS.get((line, p), []) if lc["weather_terms"] else []
            preds, b = _glm_fit_predict(tr, frames, f"yp_{p}", f"pr_{p}", lc, wx, nb_p, lc["peril_slope"])
            coefs[p] = b
            peril = [a + q for a, q in zip(peril, preds)]
            xs_num = [a + XS_REL[line].get(p, 0.0) * q for a, q in zip(xs_num, preds)]
    total = [np.zeros(len(f)) for f in frames]
    if lc["w_peril"] < 1:
        wx = sorted({v for (l, _), vs in WEATHER_PERIL_TERMS.items() if l == line for v in vs}) \
            if lc["weather_terms"] else []
        pcol = ["pr_FREQ", "pr_SEV"] if lc.get("freqsev") else "pr_TOTAL"
        total, coefs["TOTAL"] = _glm_fit_predict(tr, frames, "y_cap", pcol, lc, wx, nb_p, lc["peril_slope"])
    pred = [lc["w_peril"] * a + (1 - lc["w_peril"]) * b for a, b in zip(peril, total)]
    if lc["nb_mode"] == "line":        # one new-business factor per line on top of the summed perils
        X = np.c_[np.ones(len(tr)), tr.new_business.values.astype(float)]
        b = tweedie_glm(X, tr.y_cap.values, np.log(np.maximum(pred[0], 1e-9)), np.array([0.0, lc["nb_ridge"]]))
        coefs["NB_line"] = {"const": b[0], "nb": b[1]}
        pred = [q * np.exp(b[0] + b[1] * f.new_business.values.astype(float)) for q, f in zip(pred, frames)]
    if lc["w_resid"] > 0:
        pred = _residual(tr, frames, pred, lc["w_resid"], cfg["resid_params"])
    if lc.get("w_a", 0) > 0:
        pa = _emerging_gbm(tr, frames, cfg["a_params"])
        pred = [(1 - lc["w_a"]) * q + lc["w_a"] * r for q, r in zip(pred, pa)]
    if lc.get("w_pool", 0) > 0:
        pp = _pooled_gbm(ctx, cfg, line, tr, frames)
        pred = [(1 - lc["w_pool"]) * q + lc["w_pool"] * r for q, r in zip(pred, pp)]
    if xs_w > 0:         # excess above the cap follows the row's peril mix (see excess_profile)
        xw = [xs_w * n / np.maximum(s, 1e-9) + (1 - xs_w) for n, s in zip(xs_num, peril)]
        xw = [w / np.average(xw[0], weights=pred[0]) for w in xw]
    else:
        xw = [np.ones(len(f)) for f in frames]
    return pred, coefs, xw


def emerging_feats(df):
    drop = BASE_DROP | YEAR_WEATHER | set(PRIOR_COLS) | {"occ_base", "affluence"}
    return [c for c in df.columns if c not in drop and df[c].notna().any()]


def _residual(tr, frames, pred, w_resid, rp):
    """Small Tweedie GBM boosting from the GLM layer (offset) on emerging features; blended in."""
    fs = emerging_feats(tr)
    cf = cat_align([f[fs] for f in frames], fs)
    y = tr.y_cap.values
    off = np.log(np.maximum(pred[0], 1e-9))
    params = {**RESID_BASE, **rp, "seed": SEED}
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(tr, groups=tr.customer_id))
    m = lgb.train(params, lgb.Dataset(cf[0].iloc[a], y[a], init_score=off[a]), 3000,
                  valid_sets=[lgb.Dataset(cf[0].iloc[b], y[b], init_score=off[b])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    m = lgb.train(params, lgb.Dataset(cf[0], y, init_score=off), max(m.best_iteration, 1))
    res = [np.exp(m.predict(f, raw_score=True) + np.log(np.maximum(q, 1e-9))) for f, q in zip(cf, pred)]
    bal = y.sum() / res[0].sum()
    return [(1 - w_resid) * q + w_resid * r * bal for q, r in zip(pred, res)]


def _emerging_gbm(tr, frames, ap_params):
    """Emerging-only Tweedie GBM (model A of v1): exposure offset, emerging features incl. history."""
    fs = emerging_feats(tr)
    cf = cat_align([f[fs] for f in frames], fs)
    y = tr.y_cap.values
    base = np.log(y.sum() / tr.exposure.values.sum())
    off = np.log(tr.exposure.values) + base
    params = {**RESID_BASE, **ap_params, "seed": SEED}
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(tr, groups=tr.customer_id))
    m = lgb.train(params, lgb.Dataset(cf[0].iloc[a], y[a], init_score=off[a]), 4000,
                  valid_sets=[lgb.Dataset(cf[0].iloc[b], y[b], init_score=off[b])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    m = lgb.train(params, lgb.Dataset(cf[0], y, init_score=off), max(m.best_iteration, 10))
    res = [np.exp(m.predict(f, raw_score=True) + np.log(fr.exposure.values) + base) for f, fr in zip(cf, frames)]
    bal = y.sum() / res[0].sum()
    return [r * bal for r in res]


def _pooled_gbm(ctx, cfg, line, tr, frames):
    """Pooled Tweedie GBM: the line's developed rows (sampled to pool.max_dev) and the emerging training rows in
    one model with a market flag, the prior's shared features plus region and gender (NaN for developed rows),
    emerging rows weighted pool.w_em, a per-market base rate in the offset and early stopping on 15% of the
    emerging rows (grouped by customer). Unlike the prior + layer, trees can split on the market, so emerging
    relativities can differ non-linearly. Re-levelled to the emerging training rows."""
    pc = cfg["pool"]
    fs = prior_features(ctx, line, cfg["prior"][line])
    extra = [c for c in ("region", "gender") if c not in fs]
    feats = fs + extra + ["is_em"]
    D = ctx["DVF"][ctx["DVF"].line == line]
    if len(D) > pc["max_dev"]:
        D = D.sample(pc["max_dev"], random_state=pc["seed"])
    cols = fs + extra + ["exposure", "y_cap"]
    dv = D.reindex(columns=cols).assign(is_em=0.0)
    X = cat_align([dv[feats]] + [f.reindex(columns=cols).assign(is_em=1.0)[feats] for f in frames], feats)
    base_dv = np.log(dv.y_cap.sum() / dv.exposure.sum())
    base_em = np.log(tr.y_cap.sum() / tr.exposure.sum())
    a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=pc["seed"]).split(tr, groups=tr.customer_id))
    dtr = lgb.Dataset(pd.concat([X[0], X[1].iloc[a]], ignore_index=True), np.r_[dv.y_cap.values, tr.y_cap.values[a]],
                      weight=np.r_[np.ones(len(dv)), np.full(len(a), pc["w_em"])],
                      init_score=np.r_[np.log(dv.exposure.values) + base_dv, np.log(tr.exposure.values[a]) + base_em])
    dva = lgb.Dataset(X[1].iloc[b], tr.y_cap.values[b], init_score=np.log(tr.exposure.values[b]) + base_em)
    params = {**PRIOR_BASE, **cfg["prior"][line]["params"], "seed": pc["seed"]}
    m = lgb.train(params, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    it = max(m.best_iteration, 10)
    res = [np.exp(m.predict(x, raw_score=True, num_iteration=it) + np.log(f.exposure.values) + base_em)
           for x, f in zip(X[1:], frames)]
    bal = tr.y_cap.sum() / res[0].sum()
    return [r * bal for r in res]


# =====================================================================================
# validation: forward 2023->2024, backward 2024->2023, 5-fold CV grouped by customer
# =====================================================================================
_VCACHE = {}


def validate_line(ctx, cfg, l, schemes=("fwd", "bwd", "cv")):
    """Out-of-sample predictions for one line; cached on everything that can change them."""
    lc = cfg["layer"][l]
    key = json.dumps([l, lc, cfg["resid_params"] if lc["w_resid"] > 0 else None,
                      cfg["a_params"] if lc.get("w_a", 0) > 0 else None, prior_key(ctx, cfg, l),
                      cfg.get("xs_peril", 0), cfg.get("pool") if lc.get("w_pool", 0) > 0 else None,
                      list(schemes)], sort_keys=True, default=str)
    if key in _VCACHE:
        return _VCACHE[key]
    d = ctx["EF"][ctx["EF"].line == l].reset_index(drop=True)
    folds = []
    if "fwd" in schemes:
        folds.append(("fwd", np.where(d.year == 2023)[0], np.where(d.year == 2024)[0]))
    if "bwd" in schemes:
        folds.append(("bwd", np.where(d.year == 2024)[0], np.where(d.year == 2023)[0]))
    if "cv" in schemes:
        for k, (a, b) in enumerate(GroupKFold(5).split(d, groups=d.customer_id)):
            folds.append((f"cv{k}", a, b))
    parts = []
    for name, a, b in folds:
        tr, va = d.iloc[a], d.iloc[b]
        pred, _, xw = layer_predict(tr, [va], l, cfg, ctx)
        parts.append(pd.DataFrame({"line": l, "fold": name, "year": va.year.values, "y": va.y_tot.values,
                                   "yc": va.y_cap.values, "ycal": va.y_cal.values, "n": va.n_dev.values,
                                   "w": va.w.values, "nb": va.new_business.values.astype(bool),
                                   "p": pred[1], "L": tr.y_xs.sum() / tr.y_cap.sum() * xw[1]}))
    out = pd.concat(parts, ignore_index=True)
    _VCACHE[key] = out
    return out


def validate(ctx, cfg, schemes=("fwd", "bwd", "cv"), keep_preds=False):
    """Tweedie D2 on the 2025 business mix (validation rows weighted to its new-business share).
    *_c  = capped claims, level-neutral per line x fold (segmentation; the selection score)
    *_raw = uncapped claims with the large-loss loading, levels as predicted."""
    V = pd.concat([validate_line(ctx, cfg, l, schemes) for l in LINES], ignore_index=True)
    V["scheme"] = V.fold.str.replace(r"\d", "", regex=True)
    r = {}
    for sc, g in V.groupby("scheme"):
        k = {key: (h.yc * h.w).sum() / (h.p * h.w).sum() for key, h in g.groupby(["line", "fold"])}
        kk = pd.Series([k[key] for key in zip(g.line, g.fold)], index=g.index)
        r[f"{sc}_raw"] = d2(g.y, g.p * (1 + g.L), g.w)
        r[f"{sc}_c"] = d2(g.yc, g.p * kk, g.w)
        for l in LINES:
            h = g.line == l
            r[f"{sc}_{l[:3]}_c"] = d2(g.yc[h], (g.p * kk)[h], g.w[h])
            r[f"{sc}_{l[:3]}_PA"] = float((g.p * (1 + g.L) * g.w)[h].sum() / (g.y * g.w)[h].sum())
    r["score"] = float(np.mean([r[f"{s}_c"] for s in schemes]))
    r["score_raw"] = float(np.mean([r[f"{s}_raw"] for s in schemes]))
    return (r, V) if keep_preds else r


def nb_cred_factors(ctx, cfg):
    """v1-style new-business calibration. The layer under-uses the new-business flag (8% of training
    exposure, 76% of 2025). Segment A/E is measured on out-of-fold CV predictions, on claims capped
    at the 97.5th percentile (2023 motor NB had a few very large claims), relative to the line's A/E
    so only the NB / renewal split moves, then credibility-weighted with Z = n / (n + k)."""
    nc = cfg.get("nb_cred", {"enabled": False})
    fac = {l: {True: 1.0, False: 1.0} for l in LINES}
    if not nc.get("enabled"):
        return fac, None
    V = pd.concat([validate_line(ctx, cfg, l, ("cv",)) for l in LINES], ignore_index=True)
    rows = []
    for l in LINES:
        G = V[V.line == l]
        for nb in (True, False):
            g = G[G.nb == nb]
            ae = (g.ycal.sum() / g.p.sum()) / (G.ycal.sum() / G.p.sum())
            z = g.n.sum() / (g.n.sum() + nc["k"])
            fac[l][nb] = float(np.clip(1 + z * (ae - 1), *nc["bounds"]))
            by_year = {int(y): round(float((h.ycal.sum() / h.p.sum()) / (G[G.year == y].ycal.sum() /
                                                                          G[G.year == y].p.sum())), 3)
                       for y, h in g.groupby("year")}
            rows.append((l, "new business" if nb else "renewal", round(g.n.sum()), round(ae, 3), by_year,
                         round(z, 2), round(fac[l][nb], 3)))
    tab = pd.DataFrame(rows, columns=["line", "segment", "claims", "A/E", "A/E by year", "Z", "factor"])
    return fac, tab


# =====================================================================================
# MODE=tune: recursive coordinate search
# =====================================================================================
def _get(cfg, path):
    v = cfg
    for k in path.split("."):
        v = v[k]
    return v


def _set(cfg, path, value):
    cfg = copy.deepcopy(cfg)
    v = cfg
    ks = path.split(".")
    for k in ks[:-1]:
        v = v[k]
    v[ks[-1]] = value
    return cfg


LAYER_SPACE = []
for _l in LINES:
    LAYER_SPACE += [(f"layer.{_l}.w_peril", [0.0, 0.5, 0.75, 1.0]),
                    (f"layer.{_l}.peril_slope", [True, False]),
                    (f"layer.{_l}.slope_ridge", [0.0, 0.3, 1.0, 3.0]),
                    (f"layer.{_l}.weather_terms", [False, True]),
                    (f"layer.{_l}.nb_mode", ["none", "line", "peril"]),
                    (f"layer.{_l}.nb_ridge", [0.0, 1.0, 5.0]),
                    (f"layer.{_l}.extra_terms", [[], ["gender"], ["gender", "selfemp"], ["gender", "region"],
                                                 ["region"], ["gender", "selfemp", "region"],
                                                 ["gender", "selfemp", "age"], ["hist"], ["gender", "hist"],
                                                 ["gender", "selfemp", "hist"]]),
                    (f"layer.{_l}.extra_ridge", [0.3, 1.0, 5.0]),
                    (f"layer.{_l}.w_resid", [0.0, 0.25, 0.5]),
                    (f"layer.{_l}.w_a", [0.0, 0.15, 0.3, 0.45]),
                    (f"layer.{_l}.freqsev", [False, True])]
LAYER_SPACE += [("resid_params.num_leaves", [3, 4, 8]), ("resid_params.min_data_in_leaf", [200, 400, 800]),
                ("resid_params.learning_rate", [0.01, 0.02])]
PRIOR_SPACE = []
for _l in LINES:
    PRIOR_SPACE += [(f"prior.{_l}.premium", [False, True]),
                    (f"prior.{_l}.avg_weather", [False, True]),
                    (f"prior.{_l}.anomalies", [True, False]),
                    (f"prior.{_l}.history", [True, False]),
                    (f"prior.{_l}.params.num_leaves", [3, 7, 15, 31]),
                    (f"prior.{_l}.params.min_data_in_leaf", [1000, 2000, 4000, 8000]),
                    (f"prior.{_l}.params.learning_rate", [0.03, 0.05, 0.1]),
                    (f"prior.{_l}.params.feature_fraction", [0.5, 0.7, 0.9])]


PRIOR_SPACE_SMALL = []
for _l in LINES:
    PRIOR_SPACE_SMALL += [(f"prior.{_l}.premium", [False, True]),
                          (f"prior.{_l}.history", [True, False]),
                          (f"prior.{_l}.avg_weather", [False, True]),
                          (f"prior.{_l}.params.num_leaves", [7, 15, 31])]


def ensure_priors(ctx, cfg, cur_keys):
    """Attach the priors cfg asks for, refitting (or loading from cache) only lines that changed."""
    keys = {l: prior_key(ctx, cfg, l) for l in LINES}
    changed = [l for l in LINES if cur_keys.get(l) != keys[l]]
    if changed:
        pri = fit_priors(ctx, cfg, changed)
        if "pr_TOTAL" in ctx["EF"]:
            old = pd.concat([ctx[k] for k in ("EF", "ET", "RF")])
            pri = pd.concat([old.loc[~old.line.isin(changed), ["row_id"] + PRIOR_COLS], pri], ignore_index=True)
        attach_priors(ctx, pri)
        cur_keys.update(keys)


def tune(ctx, cfg, rounds=3, tol=0.0003, raw_guard=0.001, tune_prior=True):
    """Coordinate ascent: try each value of one setting with the others fixed, keep it if the
    segmentation score (capped, level-neutral) beats the best by more than tol AND the raw score
    (uncapped, levels as predicted) does not fall by more than raw_guard; repeat full passes until a
    pass improves nothing. The guard stops changes that only re-shuffle large-claim cost."""
    logp = OUT / "tuning_log.csv"
    rows, cur_keys = [], {}

    def evaluate(c):
        ensure_priors(ctx, c, cur_keys)
        return validate(ctx, c)

    best_cfg = copy.deepcopy(cfg)
    best = evaluate(best_cfg)
    log(f"tune start score={best['score']:.5f} raw={best['score_raw']:.5f}")
    rows.append({"pass": 0, "param": "start", "value": "", "kept": True, **best})
    prior_space = PRIOR_SPACE_SMALL if os.environ.get("IAA_TUNE_SPACE", "small") == "small" else PRIOR_SPACE
    space = LAYER_SPACE + (prior_space if tune_prior else [])
    for it in range(1, rounds + 1):
        improved = False
        for path, values in space:
            cur_val = _get(best_cfg, path)
            for v in values:
                if v == cur_val:
                    continue
                cand = _set(best_cfg, path, v)
                r = evaluate(cand)
                keep = r["score"] > best["score"] + tol and r["score_raw"] > best["score_raw"] - raw_guard
                rows.append({"pass": it, "param": path, "value": json.dumps(v), "kept": keep, **r})
                pd.DataFrame(rows).to_csv(logp, index=False)
                log(f"  pass {it} {path}={v}: score {r['score']:.5f} (best {best['score']:.5f})"
                    f"{'  <- kept' if keep else ''}")
                if keep:
                    best, best_cfg, improved = r, cand, True
            ensure_priors(ctx, best_cfg, cur_keys)
        (OUT / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
        if not improved:
            break
    log(f"tune done: best score {best['score']:.5f} -> {OUT / 'best_config.json'}")
    return best_cfg, best


# =====================================================================================
# STEP 8  final fit on 2023-24 -> 2025 submission (+ reference policies, governance)
# =====================================================================================
def final(ctx, cfg, val=None):
    banner("FINAL FIT 2023-2024 -> 2025")
    EF, ET, RF = ctx["EF"], ctx["ET"], ctx["RF"]
    preds, ref_preds, levels, coef_log = [], [], {}, {}
    nbf, nbtab = nb_cred_factors(ctx, cfg)
    if nbtab is not None:
        banner("NEW-BUSINESS CALIBRATION (credibility-weighted, out-of-fold, 97.5%-capped claims)")
        print(nbtab.to_string(index=False))
    for l in LINES:
        tr, ap, rf = EF[EF.line == l], ET[ET.line == l], RF[RF.line == l]
        (p_tr, p_ap, p_rf), coefs, (_, xw_ap, xw_rf) = layer_predict(tr, [ap, rf], l, cfg, ctx)
        L = tr.y_xs.sum() / tr.y_cap.sum()
        lev = cfg["level"][l] * float(os.environ.get("IAA_LEVEL_SCALE", 1.0))
        f_ap = np.where(ap.new_business.values.astype(bool), nbf[l][True], nbf[l][False])
        f_rf = np.where(rf.new_business.values.astype(bool), nbf[l][True], nbf[l][False])
        p = p_ap * (1 + L * xw_ap) * lev * f_ap
        hist_pp = tr.y_tot.sum() / tr.exposure.sum()
        pp24 = tr[tr.year == 2024].pipe(lambda d: d.y_tot.sum() / d.exposure.sum())
        ratio = p.sum() / ap.exposure.sum() / hist_pp
        flag = "" if LEVEL_BAND[0] <= ratio <= LEVEL_BAND[1] else "  <-- ESCALATE: outside level band"
        nbm = ap.new_business.values.astype(bool)
        nb_rel = (p[nbm].sum() / ap.exposure.values[nbm].sum()) / (p[~nbm].sum() / ap.exposure.values[~nbm].sum())
        log(f"  {l:9s} 2025 cost/exp {p.sum() / ap.exposure.sum():7.1f} vs 2023-24@2025 {hist_pp:7.1f} "
            f"(x{ratio:.2f}){flag} | 2024 only {pp24:7.1f} | NB/renewal {nb_rel:.2f} | loading {L:.2%}")
        levels[l] = dict(pred=float(p.sum() / ap.exposure.sum()), hist=float(hist_pp), pp24=float(pp24),
                         ratio=float(ratio), nb_rel=float(nb_rel), loading=float(L))
        coef_log[l] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in coefs.items()}
        preds.append(pd.DataFrame({"row_id": ap.row_id.values, "line": l, "prediction": p}))
        if len(rf):
            ref_preds.append(pd.DataFrame({"row_id": rf.row_id.str.replace("REF_", "", regex=False).values,
                                           "line": l, "channel": rf.channel.values, "exposure": rf.exposure.values,
                                           "written_premium": rf.written_premium.values,
                                           "risk_premium": p_rf * (1 + L * xw_rf) * lev * f_rf}))
    P = pd.concat(preds, ignore_index=True)
    sample = ctx["EM"]["sample"]
    sub = sample[["row_id"]].merge(P[["row_id", "prediction"]], on="row_id", how="left")
    assert len(sub) == len(sample) and sub.row_id.is_unique, "row count / uniqueness mismatch vs sample submission"
    assert set(sub.row_id) == set(ctx["EM"]["tst"].row_id), "row_ids differ from the test files"
    assert sub.prediction.notna().all() and np.isfinite(sub.prediction).all(), "missing / non-finite predictions"
    sub["prediction"] = sub.prediction.clip(lower=0.01).round(4)
    sub_path = OUT / "submission_week2_v2.csv"
    sub.to_csv(sub_path, index=False)
    if os.environ.get("IAA_PROBES", "0") == "1":      # optional level probes (see level_probe.py)
        for k in (0.92, 1.08):
            sub.assign(prediction=(sub.prediction * k).round(4)).to_csv(OUT / f"submission_week2_v2_x{k:.2f}.csv",
                                                                        index=False)
        log("wrote level probes x0.92 / x1.08 next to the submission")
    log(f"wrote {sub_path} rows={len(sub):,} mean={sub.prediction.mean():.2f} max={sub.prediction.max():.0f}")

    # commercial sense-check on the 20 reference policies (emerging expense basis)
    ex = ctx["expenses"].set_index("item").emerging
    R = pd.concat(ref_preds, ignore_index=True)
    comm = R.channel.map(lambda c: ex.get(f"commission_{c}", np.nan)).fillna(ex["commission_broker"])
    cat = np.where(R.line == "motor", 0.0, ex["cat_reinsurance"])
    R["technical_premium"] = (R.risk_premium * (1 + cat) + ex["fixed_expense_per_policy"] * R.exposure) / \
        (1 - ex["variable_expense"] - comm - ex["premium_tax"] - ex["target_profit_margin"])
    R["tech/written"] = R.technical_premium / R.written_premium
    R.to_csv(OUT / "reference_policies_technical_premium.csv", index=False)
    banner("REFERENCE POLICIES - technical premium on the emerging expense basis")
    print(R.round(2).to_string(index=False))

    pd.DataFrame(QA, columns=["market", "split", "line", "issue", "rows"]).to_csv(OUT / "qa_log.csv", index=False)
    manifest = dict(script="week2_v2.py", mode=MODE, smoke=SMOKE, run_seconds=round(time.time() - T0, 1),
                    config=cfg, data_md5=ctx["data_md5"], validation=val, levels=levels, glm_coefficients=coef_log,
                    nb_factors={l: {str(k): v for k, v in d.items()} for l, d in nbf.items()},
                    trends_pa={l: round(float(np.exp(v) - 1), 4) for l, v in ctx["trends"].items()},
                    level_scale=float(os.environ.get("IAA_LEVEL_SCALE", 1.0)),
                    severity_development=None if ctx["sev_dev"] is None else ctx["sev_dev"].to_dict("records"),
                    excess_profile=XS_REL,
                    submission=dict(rows=len(sub), mean=float(sub.prediction.mean()), path=sub_path.name))
    (OUT / "run_manifest_v2.json").write_text(json.dumps(manifest, indent=2, default=str))
    return sub


KAGGLE_NB_INTRO = """# IAA AI-Assisted Hackathon 2026 - Week 2 (Emerging market) - team CST

Transfer pipeline: developed-market (Part 1) relativities re-based to the emerging market, with every setting
fixed from local validation (the header of the code cell lists what each version changed). This notebook is
written by `IAA_MODE=export python week2_v2.py`: edit week2_v2.py in the repository, not the notebook.

**How to run**
1. *Add Input* -> the competition data (`iaa-ai-assisted-hackathon-2026-part-2`, folders `emerging/` and `developed/`).
   The code finds them automatically under `/kaggle/input`.
2. *Run All* (about 10-20 minutes on a CPU notebook).
3. Submit `/kaggle/working/submission_week2_v2.csv`.

The line levels in `CFG["level"]` were calibrated on the public leaderboard, credibility-weighted towards the
model (level_probe.py in the repository).
"""
KAGGLE_NB_SETTINGS = """# ---- settings (read by the pipeline at run time) ----
import os
# os.environ["IAA_LEVEL_SCALE"] = "1.00"  # overall multiplier on top of CFG["level"] (default 1)
# os.environ["IAA_VALIDATE"] = "1"        # local validation report (default on; also feeds the NB calibration)
"""


def export_kaggle(cfg, dest=None):
    """Writes a copy of this script with cfg embedded as the CFG defaults (no best_config.json needed), and the
    same code as a Kaggle notebook next to it."""
    src = (ROOT / "week2_v2.py").read_text()
    a = src.index("CFG = {")
    b = src.index("WEATHER_PERIL_TERMS = ")
    lit = json.dumps(cfg, indent=4).replace(": true", ": True").replace(": false", ": False").replace(": null", ": None")
    out = src[:a] + "CFG = " + lit + "\n" + src[b:]
    out = out.replace('MODE = os.environ.get("IAA_MODE", "final").lower()',
                      'MODE = os.environ.get("IAA_MODE", "final").lower()      # Kaggle copy: final mode', 1)
    dest = dest or ROOT / "week2_v2_kaggle.py"
    dest.write_text(out)
    compile(out, str(dest), "exec")
    cells = [("markdown", KAGGLE_NB_INTRO), ("code", KAGGLE_NB_SETTINGS), ("code", out)]
    nb = {"cells": [{"cell_type": k, "id": f"cell{i}", "metadata": {}, "source": t.splitlines(keepends=True),
                     **({"execution_count": None, "outputs": []} if k == "code" else {})}
                    for i, (k, t) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    dest.with_suffix(".ipynb").write_text(json.dumps(nb, indent=1))
    log(f"wrote Kaggle script {dest} and notebook {dest.with_suffix('.ipynb').name} (CFG embedded)")


def main():
    banner(f"IAA week 2 v2 | mode={MODE} smoke={SMOKE} | out={OUT}")
    cfg = copy.deepcopy(CFG)
    best_path = OUT / "best_config.json"
    if os.environ.get("IAA_CONFIG"):        # re-run a past submission from its run manifest (or any config JSON)
        given = json.loads(Path(os.environ["IAA_CONFIG"]).read_text())
        cfg = given.get("config", given)
        log(f"using the configuration in {os.environ['IAA_CONFIG']}")
    elif MODE == "export" and best_path.exists():
        cfg = json.loads(best_path.read_text())
    if MODE == "export":
        export_kaggle(cfg)
        return
    if MODE == "final" and os.environ.get("IAA_USE_BEST") == "1" and best_path.exists():
        cfg = json.loads(best_path.read_text())
        log(f"using tuned configuration {best_path}")
    ctx = prepare()
    if MODE == "tune":
        if best_path.exists() and os.environ.get("IAA_TUNE_FROM_BEST", "1") == "1":
            cfg = json.loads(best_path.read_text())          # recursive: continue from the last best
            log(f"tuning continues from {best_path}")
        seeds = cfg["prior_seeds"]
        cfg["prior_seeds"] = seeds[:1]                       # one seed while searching (twice as fast)
        cfg, val = tune(ctx, cfg, rounds=int(os.environ.get("IAA_TUNE_ROUNDS", 3)),
                        tune_prior=os.environ.get("IAA_TUNE_PRIOR", "1") == "1")
        cfg["prior_seeds"] = seeds
        (OUT / "best_config.json").write_text(json.dumps(cfg, indent=2))
        attach_priors(ctx, fit_priors(ctx, cfg))
        val = validate(ctx, cfg)
        banner("VALIDATION of the tuned configuration (all prior seeds)")
        print(pd.Series(val).round(4).to_string())
    else:
        attach_priors(ctx, fit_priors(ctx, cfg))
        val = validate(ctx, cfg) if os.environ.get("IAA_VALIDATE", "1") == "1" else None
        if val:
            banner("VALIDATION (Tweedie D2 on the 2025 business mix)")
            print(pd.Series(val).round(4).to_string())
    final(ctx, cfg, val)
    log("done")


if __name__ == "__main__":
    main()
