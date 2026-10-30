"""
kaggle_part2.py
===============
IAA AI-Assisted Hackathon 2026 - Part 2 (Week 2: Emerging Market Expansion)
Complete Actuarial Pricing & Prediction Pipeline

This script implements:
1. Automated loading & cleaning of Part 2 Emerging Market data via load_part2.py.
2. Actuarial development (IBNR/IBNER), severity trend estimation, and large-loss capping.
3. Feature engineering tailored to emerging market schema (age bands, gender, weather anomalies, vehicle/property risk factors).
4. Validation fold on 2024 evaluating Tweedie D^2 score (power = 1.5).
5. Multi-model ensemble:
   - Model A: LightGBM Tweedie with log-exposure offset
   - Model B: Complementary regularized LightGBM Tweedie
   - Model C: Frequency (Poisson) x Severity (Gamma) two-part actuarial model
6. Commercial pricing analysis using expenses.parquet and reference_policies.parquet.
7. Final 2025 predictions and submission validation for 209,063 test rows.
"""

import os
import time
import warnings
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from load_part2 import load_part2_data, resolve_data_dir

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
T0 = time.time()

def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

# ----------------------------------- Config -----------------------------------------
SMOKE = os.environ.get("IAA_SMOKE") == "1"
POWER = 1.5
VALUATION_YEAR = 2024
TARGET_YEAR = 2025
CAP_Q = 0.995
SUSPECT_X = 10
MATURE_OPEN_SHARE = 0.35
MAX_TREND = np.log(1.15)
TREND_MULT = {"motor": 1.0, "buildings": 1.0, "contents": 1.0}
FS_CLIP = (0.5, 2.0)
TE_SMOOTH = 200
SEED = 42
SEEDS = [42] if SMOKE else [42, 7, 2025]
LINES = ["motor", "buildings", "contents"]
LEAKY = ["customer_service_contacts"]
NOISE = ["previous_insurer", "campaign_code", "email_domain", "branch_id", "vehicle_colour",
         "marketing_consent", "document_preference", "marketing_segment", "surname"]

MIN_LEAF = 20 if SMOKE else 300
LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.03,
                  num_leaves=12, min_data_in_leaf=MIN_LEAF,
                  feature_fraction=0.7, bagging_fraction=0.75, bagging_freq=1,
                  lambda_l2=30, cat_smooth=100,
                  min_data_per_group=(20 if SMOKE else 150), max_cat_threshold=8, seed=SEED, verbose=-1)

LINEB_PARAMS = {**LGB_PARAMS, "learning_rate": 0.02, "num_leaves": 18,
                "min_data_in_leaf": (20 if SMOKE else 200), "feature_fraction": 0.6, "lambda_l2": 60}

_BASE_NO_TW = {k: v for k, v in LGB_PARAMS.items() if k != "tweedie_variance_power"}
FREQ_PARAMS = {**_BASE_NO_TW, "objective": "poisson", "learning_rate": 0.03,
               "num_leaves": 14, "min_data_in_leaf": (20 if SMOKE else 250)}
SEV_PARAMS = {**_BASE_NO_TW, "objective": "gamma", "learning_rate": 0.03,
              "num_leaves": 10, "min_data_in_leaf": (15 if SMOKE else 100),
              "min_data_per_group": (15 if SMOKE else 80)}

def d2(y, p):
    return d2_tweedie_score(y, np.maximum(p, 1e-6), power=POWER)

def full_year_end(s):
    # Standard full year ends on 12-30 (in 2024) or 12-31 (in 2023 & 2025)
    return s.dt.strftime("%m-%d").isin(["12-30", "12-31"])

# ------------------------------- Step 1: Load Data -----------------------------------
log("Loading cleaned Part 2 dataset...")
data = load_part2_data()

cust = data["customers"]
addr = data["addresses"]
wthr = data["weather"]
claims_raw = data["claims"]
pol = data["policies"]
tst = data["test"]
sample_sub = data["sample_submission"]
expenses = data["expenses"]
reference_policies = data["reference_policies"]
qa = data["qa_log"]

log(f"Data summary: {len(pol):,} train rows, {len(tst):,} test rows, {len(claims_raw):,} claims linked.")

# ------------------------------- Step 2: Target Construction ------------------------
ALL_PERILS = sorted(claims_raw["peril"].unique())
YP_COLS = ["yp_" + p for p in ALL_PERILS]
PERILS = {l: sorted(claims_raw.loc[claims_raw["line"] == l, "peril"].unique()) for l in LINES}
print("\n>>> Perils by line of business:", PERILS)

# Large-loss caps (99.5th percentile per line x peril)
caps = claims_raw[claims_raw["loss_year"] <= VALUATION_YEAR].groupby(["line", "peril"])["net"].agg(
    cap=lambda s: s.quantile(CAP_Q), p999=lambda s: s.quantile(0.999)).reset_index()
cl = claims_raw.merge(caps, on=["line", "peril"], how="left")
cl["suspect"] = cl["net"] > SUSPECT_X * cl["p999"]
cl["net_cap"] = np.minimum(cl["net"], cl["cap"])
cl["net_xs"] = np.where(cl["suspect"], 0.0, cl["net"] - cl["net_cap"])

if cl["suspect"].any():
    print("\n>>> Suspect large claims (capped for modeling):")
    print(cl.loc[cl["suspect"], ["claim_id", "policy_id", "line", "peril", "loss_year", "status", "net", "cap"]])

def compute_development(cl_df, max_year):
    """Computes IBNR and IBNER factors for emerging market claims."""
    # Group by line, peril, loss_year
    cell = (cl_df[cl_df["loss_year"] >= 2023].groupby(["line", "peril", "loss_year"])
            .agg(n=("net", "size"), sev=("net_cap", "mean"),
                 open_share=("status", lambda s: (s == "open").mean())).reset_index())
    
    # Severity trend by line between 2023 and 2024 mature claims
    trends = {}
    for line, g in cell.groupby("line"):
        m23 = g[g["loss_year"] == 2023]
        m24 = g[(g["loss_year"] == 2024) & (g["open_share"] <= MATURE_OPEN_SHARE)]
        if len(m24) > 0 and len(m23) > 0:
            tr_est = np.log(m24["sev"].mean() / m23["sev"].mean())
            trends[line] = float(np.clip(tr_est, 0.0, MAX_TREND))
        else:
            trends[line] = 0.04  # Actuarial default annual trend 4%
        
        # Development factors: undeveloped cells get adjusted to mature level
        idx = g.index
        cell.loc[idx, "slope"] = trends[line]
        immature = (g["open_share"] > MATURE_OPEN_SHARE).values
        # For immature 2024 claims, estimate IBNER based on mature 2023 level + trend
        expected_sev = m23["sev"].mean() * np.exp(trends[line]) if len(m23) else g["sev"].mean()
        cell.loc[idx, "ibner"] = np.where(immature, np.clip(expected_sev / g["sev"].values, 1.0, 2.0), 1.0)
        # Reporting delay IBNR: based on open share
        cell.loc[idx, "ibnr"] = np.where(immature, 1.0 + 0.3 * g["open_share"].values, 1.0)
    
    cell["dev"] = (cell["ibnr"] * cell["ibner"]).clip(upper=2.5)
    return cell, trends

def build_targets(rows, cl_df, max_train_year, to_year):
    cell, trends = compute_development(cl_df, max_train_year)
    c = cl_df.merge(cell[["line", "peril", "loss_year", "dev", "ibnr", "slope"]],
                    on=["line", "peril", "loss_year"], how="left")
    c["dev"] = c["dev"].fillna(1.0)
    c["n_dev"] = c["ibnr"].fillna(1.0).clip(upper=2.0)
    c["trend"] = np.exp(c["slope"].fillna(0.04) * (to_year - c["loss_year"].values))
    c["y_cap"] = c["net_cap"] * c["dev"] * c["trend"]
    c["y_xs"] = c["net_xs"] * c["dev"] * c["trend"]

    agg = c.groupby("row_id").agg(
        y_cap=("y_cap", "sum"), y_xs=("y_xs", "sum"),
        y_raw=("net", "sum"), n_claims=("net", "size"), n_dev=("n_dev", "sum"))
    out = rows.merge(agg, on="row_id", how="left")
    out[["y_cap", "y_xs", "y_raw", "n_claims", "n_dev"]] = out[["y_cap", "y_xs", "y_raw", "n_claims", "n_dev"]].fillna(0.0)

    pp = (c.pivot_table(index="row_id", columns="peril", values="y_cap", aggfunc="sum", fill_value=0)
            .reindex(columns=ALL_PERILS, fill_value=0))
    pp.columns = YP_COLS
    out = out.merge(pp, left_on="row_id", right_index=True, how="left")
    out[YP_COLS] = out[YP_COLS].fillna(0.0)
    return out, cell, trends

# ------------------------------- Step 3: Feature Engineering -------------------------
log("Engineering features for Emerging Market...")

# Customer features (Emerging market has gender, no credit_band, no date_of_birth)
cust_f = cust.drop(columns=[c for c in NOISE if c in cust]).copy()
if "gender" in cust_f:
    cust_f["gender"] = cust_f["gender"].map({"M": 0, "F": 1, "Male": 0, "Female": 1}).fillna(0)

# Weather features
wvars = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
         "wildfire_index", "rainfall_mm", "mean_temperature_c"]
w_mean = wthr.groupby("address_id")[wvars].mean().add_suffix("_avg")
wthr_f = wthr.merge(w_mean, left_on="address_id", right_index=True)
for v in wvars:
    wthr_f[v + "_anom"] = wthr_f[v] - wthr_f[v + "_avg"]
WCOLS = ["address_id", "year"] + wvars + [v + "_anom" for v in wvars]

# Ordinal encodings
ORD = {
    "urban_density": ["rural", "suburban", "urban", "dense_urban"],
    "security_level": ["standard", "locks", "alarm_and_locks"],
    "driver_age_band": ["17-24", "25-34", "35-49", "50-64", "65+"],
    "policyholder_age_band": ["17-24", "25-34", "35-49", "50-64", "65+"],
}

def make_features(df):
    n0 = len(df)
    f = df.merge(cust_f, on="customer_id", how="left").merge(addr, on="address_id", how="left")
    f = f.merge(wthr_f[WCOLS], on=["address_id", "year"], how="left")
    f = f.merge(w_mean, left_on="address_id", right_index=True, how="left")
    assert len(f) == n0, "Row count changed during feature merge"

    # Fill weather missing with averages
    for v in wvars:
        miss = f[v].isna()
        if miss.any():
            f.loc[miss, v] = f.loc[miss, v + "_avg"]
            f.loc[miss, v + "_anom"] = 0.0

    # Ordinal mappings
    for col, levels in ORD.items():
        if col in f:
            f[col] = f[col].map({v: i for i, v in enumerate(levels)}).fillna(-1)

    # Unified age band
    if "driver_age_band" in f and "policyholder_age_band" in f:
        f["age_band"] = np.where(f["driver_age_band"] >= 0, f["driver_age_band"], f["policyholder_age_band"])
    elif "driver_age_band" in f:
        f["age_band"] = f["driver_age_band"]
    elif "policyholder_age_band" in f:
        f["age_band"] = f["policyholder_age_band"]

    # Actuarial & Risk ratios
    f["log_premium"] = np.log(f["written_premium"].clip(lower=1.0))
    f["ends_early"] = (~full_year_end(f["period_end"])).astype(int)
    f["starts_late"] = (f["period_start"].dt.strftime("%m-%d") != "01-01").astype(int)
    f["rows_in_policy_year"] = f.groupby(["policy_id", "year"])["row_id"].transform("size")
    f["customer_tenure"] = f["year"] - f["customer_since_year"]
    f["property_age"] = f["year"] - f["year_built"]
    f["clay_dry_trees"] = (f["soil_type"] == "clay").astype(float) * f["dry_summer_index"] * f["trees_within_5m"].astype(float)
    f["elev_over_river"] = f["elevation_m"] / (f["distance_to_river_km"].clip(lower=0) + 0.1)
    f["frost_x_age"] = f["freezing_days"] * f["property_age"]
    f["hail_x_flat_roof"] = f["hail_days"] * (f["roof_type"] == "flat").astype(float)
    f["wind_x_age"] = f["wind_exposure_index"] * f["property_age"]
    f["total_excess"] = f["voluntary_excess"] + f["compulsory_excess"]

    if "sum_insured" in f:
        f["excess_to_si"] = (f["total_excess"] / f["sum_insured"].clip(lower=1.0)).replace([np.inf, -np.inf], np.nan)
        f["log_si"] = np.log(f["sum_insured"].clip(lower=1.0))
        f["si_per_bedroom"] = f["sum_insured"] / f["bedrooms"].clip(lower=1)
    if "vehicle_value_group" in f:
        f["excess_per_vvg"] = f["total_excess"] / f["vehicle_value_group"].clip(lower=1)
    if "ncb_years" in f and "years_licensed" in f:
        f["ncb_ratio"] = f["ncb_years"] / f["years_licensed"].clip(lower=1)

    grp = [f["line"], f["year"]]
    f["log_premium_norm"] = f["log_premium"] - f.groupby(grp)["log_premium"].transform("median")

    return f

DROP = set(LEAKY + NOISE + YP_COLS + ["row_id", "policy_id", "customer_id", "address_id",
           "period_start", "period_end", "renewal_decision", "y_cap", "y_xs", "y_raw", "n_claims", "n_dev",
           "line", "year", "exposure", "written_premium", "make_model",
           "customer_since_year", "year_built", "sum_insured"] + [v + "_avg" for v in wvars])

def prep(tr, ap):
    tr, ap = tr.copy(), ap.copy()
    feats = [c for c in tr.columns if c not in DROP]
    for c in feats:
        if tr[c].dtype == object or tr[c].dtype == bool or str(tr[c].dtype) in ("category", "str", "string"):
            cats = pd.Categorical(pd.concat([tr[c], ap[c]]).astype(str)).categories
            tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
            ap[c] = pd.Categorical(ap[c].astype(str), categories=cats)
    return tr, ap, feats

# ------------------------------- Step 4: Model Engine --------------------------------
def fit_tweedie(tr, feats, n_rounds=None, seed=SEED, target="y_cap", params_base=None):
    """Fits LightGBM Tweedie with log(exposure) + log(base rate) init_score offset."""
    params = {**(params_base or LGB_PARAMS), "seed": seed}
    y = tr[target].values
    base = np.log(y.sum() / tr["exposure"].sum())
    off = np.log(tr["exposure"].values) + base

    if n_rounds is None:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(tr, groups=tr["policy_id"]))
        dtr = lgb.Dataset(tr[feats].iloc[a], y[a], init_score=off[a])
        dva = lgb.Dataset(tr[feats].iloc[b], y[b], init_score=off[b])
        mdl = lgb.train(params, dtr, 2500, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        n_rounds = max(int(mdl.best_iteration), 25)
    else:
        mdl = lgb.train(params, lgb.Dataset(tr[feats], y, init_score=off), n_rounds)

    mdl.base = base
    mdl.bal = y.sum() / np.exp(mdl.predict(tr[feats], raw_score=True, num_iteration=n_rounds) + off).sum()
    mdl.n_rounds = n_rounds
    return mdl, n_rounds

def predict_tweedie(mdl, df, feats, loading):
    raw = mdl.predict(df[feats], raw_score=True, num_iteration=mdl.n_rounds)
    return np.exp(raw + mdl.base + np.log(df["exposure"].values)) * mdl.bal * (1.0 + loading)

def fit_sev(tr, feats, n_rounds=None, seed=SEED):
    """Fits Gamma model for claim severity on positive claims."""
    d = tr[tr["n_claims"] > 0]
    y = (d["y_cap"] / d["n_dev"]).values
    w = d["n_dev"].values
    base = np.log((w * y).sum() / w.sum())
    params = {**SEV_PARAMS, "seed": seed}
    X = d[feats]

    if n_rounds is None:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(d, groups=d["policy_id"]))
        dtr = lgb.Dataset(X.iloc[a], y[a], weight=w[a], init_score=np.full(len(a), base))
        dva = lgb.Dataset(X.iloc[b], y[b], weight=w[b], init_score=np.full(len(b), base))
        mdl = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
        n_rounds = max(int(mdl.best_iteration), 20)
    else:
        mdl = lgb.train(params, lgb.Dataset(X, y, weight=w, init_score=np.full(len(d), base)), n_rounds)

    mdl.base = base
    pr = np.exp(mdl.predict(X, raw_score=True, num_iteration=n_rounds) + base)
    mdl.bal = (w * y).sum() / (w * pr).sum()
    mdl.n_rounds = n_rounds
    return mdl, n_rounds

def run_freq_sev(tr, ap, feats, loading, nr=None, seed=SEED):
    nr = nr or {}
    mf, uf = fit_tweedie(tr, feats, n_rounds=nr.get("f"), seed=seed, target="n_dev", params_base=FREQ_PARAMS)
    ms, us = fit_sev(tr, feats, n_rounds=nr.get("s"), seed=seed)
    p_freq = predict_tweedie(mf, ap, feats, 0.0)
    p_sev = np.exp(ms.predict(ap[feats], raw_score=True, num_iteration=ms.n_rounds) + ms.base) * ms.bal
    return p_freq * p_sev * (1.0 + loading), {"f": uf, "s": us}

# ------------------------------- Step 5: Validation on 2024 -------------------------
log("Running validation fold: Train on 2023 -> Validate on 2024...")
tr23_raw, _, _ = build_targets(pol[pol["year"] == 2023], cl, 2023, 2024)
va24_raw, _, _ = build_targets(pol[pol["year"] == 2024], cl, 2023, 2024)

tr23_f = make_features(tr23_raw)
va24_f = make_features(va24_raw)

val_results = []
best_iters_A = {}
best_iters_B = {}
best_iters_FS = {}

for line in LINES:
    tr, va = tr23_f[tr23_f["line"] == line], va24_f[va24_f["line"] == line]
    loading = tr["y_xs"].sum() / tr["y_cap"].sum()
    tr, va, feats = prep(tr, va)

    # Model A: Tweedie Standard
    mdl_A, best_iters_A[line] = fit_tweedie(tr, feats)
    p_A = predict_tweedie(mdl_A, va, feats, loading)

    # Model B: Tweedie Deeper Regularized
    mdl_B, best_iters_B[line] = fit_tweedie(tr, feats, params_base=LINEB_PARAMS)
    p_B = predict_tweedie(mdl_B, va, feats, loading)

    # Model C: Frequency x Severity
    p_FS, best_iters_FS[line] = run_freq_sev(tr, va, feats, loading)
    p_FS = np.clip(p_FS, FS_CLIP[0] * p_A, FS_CLIP[1] * p_A)

    # Baseline simple rate
    base_rate = (tr["y_cap"].sum() + tr["y_xs"].sum()) / tr["exposure"].sum()

    val_results.append(pd.DataFrame({
        "line": line,
        "row_id": va["row_id"].values,
        "y": va["y_raw"].values,
        "y_dev": (va["y_cap"] + va["y_xs"]).values,
        "pred_A": p_A,
        "pred_B": p_B,
        "pred_AB": 0.5 * (p_A + p_B),
        "pred_FS": p_FS,
        "exp_x_rate": va["exposure"].values * base_rate
    }))

    log(f"  {line:9s} - Iterations A={best_iters_A[line]:3d}, B={best_iters_B[line]:3d}, FS={best_iters_FS[line]} | Excess Loading={loading:.2%}")

V = pd.concat(val_results, ignore_index=True)

print("\n" + "=" * 80)
print(">>> VALIDATION FOLD 2024: Tweedie D^2 Score Evaluation (power = 1.5)")
print("=" * 80)
for model_name in ["exp_x_rate", "pred_A", "pred_B", "pred_AB", "pred_FS"]:
    scores = {l: d2(g["y"], g[model_name]) for l, g in V.groupby("line")}
    scores["COMBINED"] = d2(V["y"], V[model_name])
    scores["COMBINED (vs Developed Actuals)"] = d2(V["y_dev"], V[model_name])
    print(f"Model: {model_name:<12} | Combined D2: {scores['COMBINED']:.5f} | vs Dev: {scores['COMBINED (vs Developed Actuals)']:.5f} | " +
          " | ".join([f"{l}: {scores[l]:.4f}" for l in LINES]))

# ------------------------------- Step 6: Commercial Pricing Analysis ----------------
print("\n" + "=" * 80)
print(">>> COMMERCIAL PRICING & EXPENSE ANALYSIS (expenses.parquet)")
print("=" * 80)
if expenses is not None:
    exp_dict = expenses.set_index("item")["emerging"].to_dict()
    fixed_exp = exp_dict.get("fixed_expense_per_policy", 85.0)
    var_exp = exp_dict.get("variable_expense", 0.15)
    comm_broker = exp_dict.get("commission_broker", 0.20)
    tax = exp_dict.get("premium_tax", 0.08)
    cat_reins = exp_dict.get("cat_reinsurance", 0.12)
    profit = exp_dict.get("target_profit_margin", 0.08)

    denom = 1.0 - (var_exp + comm_broker + tax + profit)
    print(f"Emerging Market Loading Structure:")
    print(f"  Fixed Expense: {fixed_exp:.2f} CU")
    print(f"  Variable Share: Variable={var_exp:.0%}, Broker Comm={comm_broker:.0%}, Tax={tax:.0%}, Profit={profit:.0%}")
    print(f"  Total Margin Denominator (1 - sum): {denom:.2f}")
    print(f"  Property Cat Reinsurance Load: +{cat_reins:.0%}")

    if reference_policies is not None:
        ref_df = reference_policies.copy()
        print(f"\nCommercial Premium Comparison for 20 Reference Policies:")
        ref_display = ref_df[["row_id", "line", "driver_age_band", "policyholder_age_band", "written_premium"]].copy()
        print(ref_display.head(6).to_string(index=False))

# ------------------------------- Step 7: Final Model Training (2023-2024 -> 2025) ----
log("Training Final Models on All Emerging Data (2023-2024) to predict 2025...")

full_train, cell_f, trend_f = build_targets(pol, cl, VALUATION_YEAR, TARGET_YEAR)
tst_dummy = tst.assign(y_cap=np.nan, y_xs=np.nan, y_raw=np.nan, n_claims=np.nan, n_dev=np.nan,
                       **{c: np.nan for c in YP_COLS})

full_train_f = make_features(full_train)
tst_f = make_features(tst_dummy)

all_preds = []

for line in LINES:
    tr = full_train_f[full_train_f["line"] == line]
    ap = tst_f[tst_f["line"] == line]
    loading = tr["y_xs"].sum() / tr["y_cap"].sum()
    tr, ap, feats = prep(tr, ap)

    n_a = int(best_iters_A[line] * 1.20)
    n_b = int(best_iters_B[line] * 1.20)

    # Multi-seed averaging
    preds_line = []
    for s in SEEDS:
        mdl_a, _ = fit_tweedie(tr, feats, n_rounds=n_a, seed=s)
        pa = predict_tweedie(mdl_a, ap, feats, loading)

        mdl_b, _ = fit_tweedie(tr, feats, n_rounds=n_b, seed=s, params_base=LINEB_PARAMS)
        pb = predict_tweedie(mdl_b, ap, feats, loading)

        preds_line.append(0.6 * pa + 0.4 * pb)

    p_final = np.mean(preds_line, axis=0)

    # Actuarial guardrail: ensure positive and non-zero
    p_final = np.clip(p_final, a_min=0.01, a_max=None)

    mean_cost = p_final.sum() / ap["exposure"].sum()
    log(f"  {line:9s} - 2025 Expected Ultimate Cost / Exposure = {mean_cost:7.2f} CU (Total rows={len(ap):,})")

    all_preds.append(pd.DataFrame({
        "row_id": ap["row_id"].values,
        "prediction": p_final
    }))

sub_df = pd.concat(all_preds, ignore_index=True)

# ------------------------------- Step 8: Submission Verification & Export ------------
out_csv = Path("submission_part2.csv")
if sample_sub is not None:
    final_sub = sample_sub[["row_id"]].merge(sub_df, on="row_id", how="left")
    assert len(final_sub) == len(sample_sub), f"Row count mismatch! {len(final_sub)} vs {len(sample_sub)}"
    assert final_sub["prediction"].notna().all(), "Found NaN predictions!"
    assert (final_sub["prediction"] >= 0).all(), "Found negative predictions!"
    final_sub["prediction"] = final_sub["prediction"].round(4)
    final_sub.to_csv(out_csv, index=False)
    log(f"SUCCESS! Wrote {len(final_sub):,} valid Part 2 predictions to {out_csv.resolve()}")
    print("\n>>> Submission preview:")
    print(final_sub.head(10).to_string(index=False))
else:
    sub_df.to_csv(out_csv, index=False)
    log(f"Wrote {len(sub_df):,} predictions to {out_csv.resolve()}")

print(f"\nPart 2 Pricing Pipeline Complete in {time.time() - T0:.1f}s!")
