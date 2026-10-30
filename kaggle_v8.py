# =====================================================================================
# IAA AI-Assisted Hackathon 2026 - Part 1 (developed market) - team CST
# Predicts the expected ultimate net claim cost of every 2025 policy row for motor, buildings and
# contents. Kaggle scores it with the Tweedie D2 at power 1.5.
#
# RUN   IAA_DATA=data/developping python3 kaggle_v8.py      (on Kaggle the input folder is found by itself)
#       IAA_SMOKE=1 IAA_DATA=data/developping IAA_OUT=outputs_smoke python3 kaggle_v8.py   (fast dry run)
#       Writes submission.csv and preds_both.parquet (raw and calibrated predictions) to IAA_OUT.
#
# PIPELINE (section banners below, in the order they run)
#   load & clean    exposure stored in days, sums insured 1000x too large, construction codes; claims linked
#                   to their policy row. Every fix is counted in the data-quality log printed at the end.
#   STEP 3a         large losses capped per line and peril; the excess comes back as a loading.
#   STEP 2          claims developed to ultimate (late reporting, open claims), the motor bodily-injury break
#                   restated, severity trended to the target year per peril.
#   STEP 4 / 4b     rating features, then a leakage and train/test shift check on every feature.
#   model harness   per line: two LightGBM Tweedie models and a Poisson x Gamma frequency-severity pair.
#   STEP 1          rolling-origin validation (score 2023, then 2024) fixes the blend and a line calibration.
#   final fit       2021-2024 -> 2025 over three seeds, with level checks and submission checks.
#
# KAGGLE (Part 1 closed 2026-10-23 16:00 UTC)
#   v7, submitted 14:05 UTC: public 0.12811, private 0.13766. Our ranked entry: 7th of 51 teams.
#   v8 (this file), submitted 16:05 UTC, after the deadline: public 0.12791, private 0.13766.
#
# v8 = v7 + targeted fixes (search "# v8")
# CHANGES vs v7 (public LB 0.12811):
#   1. BUG FIX ends_early: every 2024 training row ends on 2024-12-30 (not 12-31), so v7 flagged
#      100% of 2024 rows as "ends early" vs 22% in test. Now 12-30 and 12-31 both count as full year.
#   2. Per-peril severity trends (credibility-shrunk to the line trend) instead of one trend per line.
#      e.g. buildings fire is flat (~2.6%/yr) while escape-of-water/storm run ~8%/yr.
#   3. Structural break: motor bodily-injury severity fell ~25% from 2023-Q1 (same maturity, every
#      quantile shifts). 2021-22 BI claims are restated to the post-break basis; 2024 BI IBNER now
#      comes from the post-break fit (1.06 instead of 1.29), so the crude BI_DEV_CAP=1.6 is no longer needed.
#   4. With (2)+(3) the motor trend hedge is unnecessary: TREND_MULT back to 1.0 for all lines.
#   5. Fold table also reports D2 against DEVELOPED actuals (2024 raw BI is only ~57% reported).
# Validation (local, same seeds): 2024 fold vs developed actuals 0.13736 -> 0.13803 (+0.0007).
# 2023 fold -0.001: the 2023 BI break cannot be seen from 2021-22 training data (expected).
# Only ONE csv is written: submission.csv
# =====================================================================================
import os, time, warnings
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
T0 = time.time()

# ----------------------------------- config -----------------------------------------
# Every modelling choice is a constant in this block, so the block defines the run.
SMOKE = os.environ.get("IAA_SMOKE") == "1"            # dry run: one seed, small leaves, low peril threshold
def find_data_dir():                                  # v8: works on Kaggle and locally
    # first folder under /kaggle/input (or ./Dataset next to this script) that holds claims.parquet or .csv
    for root in [Path("/kaggle/input"), Path(__file__).resolve().parent / "Dataset" if "__file__" in globals() else Path("Dataset")]:
        if root.exists():
            hits = sorted(root.rglob("claims.parquet")) or sorted(root.rglob("claims.csv"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("claims.parquet not found")
DATA = Path(os.environ["IAA_DATA"]) if "IAA_DATA" in os.environ else find_data_dir()
OUT_DIR = Path(os.environ.get("IAA_OUT", "/kaggle/working" if Path("/kaggle/working").exists() else "."))
OUT_CSV = OUT_DIR / "submission.csv"
OUT_BOTH = OUT_DIR / "preds_both.parquet"             # raw and calibrated predictions side by side
# model switches
EXPERIMENT_ONLY = False                               # True: validation folds only, no final fit and no csv
USE_CALIBRATION = True                                # submit the line-calibrated predictions (see STEP 1)
USE_PERIL_MODELS = False                              # one model per peril, summed: off, no validation gain
USE_FS_MODEL = True                                   # Poisson frequency x Gamma severity pair per line
USE_LINE_B = True                                     # second, deeper Tweedie model averaged 50/50 with A
REG_WEATHER = False                                   # region-year weather anomalies: off, no validation gain
RECENCY = 1.0                                         # weight per year of age (1.0 = all years equal)
USE_TIMING_FEATS = True                               # ends_early, starts_late, rows_in_policy_year
PERIL_TREND = True                                   # v8 2: per-peril severity trend
BREAKS = {("motor", "bodily_injury"): 2023}          # v8 3: first loss year of the new severity regime
TREND_TAU = 0.03                                     # v8 2: prior sd of a peril trend around its line trend
# actuarial basis
VALUATION = pd.Timestamp("2024-12-31")                # claims are valued at this date
POWER = 1.5                                           # Tweedie power of the metric; the models use it too
VAL_YEARS = [2023, 2024]                              # each fold trains on earlier years and scores this one
TARGET_YEAR = 2025                                    # the year being priced
CAP_Q = 0.995                                         # large-loss cap: 99.5th percentile per line and peril
SUSPECT_X = 10                                        # a claim above 10x its 99.9th percentile is a suspect
MATURE_OPEN_SHARE = 0.5                               # a cell with more than half its claims open is immature
MAX_TREND = np.log(1.10)                              # severity trend bounded at +10% a year
TREND_MULT = {"motor": 1.0, "buildings": 1.0, "contents": 1.0}   # v8 4: was motor 0.5
FS_CLIP = (0.5, 2.0)                                  # freq x sev kept within 0.5-2x of the Tweedie model
TE_SMOOTH = 500                                       # exposure-years of credibility in the make_model encoding
MAKE_SUBMISSION = not EXPERIMENT_ONLY
SEED = 42
SEEDS = [42] if SMOKE else [42, 7, 2025]              # the final fit averages three seeds
BI_DEV_CAP = 3.0                                     # v8 3: was 1.6 (now handled by the BI break)
CAL_SHRINK = 0.75                                     # calibration keeps 75% of the fold miss ...
CAL_BOUNDS = (0.90, 1.10)                             # ... and never moves a line by more than 10%
LEVEL_BAND = (0.75, 1.30)                             # 2025 cost / trended 2024 cost outside this: human review
NEAREST_DAYS = 31                                     # a claim outside its period links to a row within 31 days
MIN_PERIL_CLAIMS = 20 if SMOKE else 300               # fewer claims: the peril gets a flat rate (peril models)
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]                   # blend weights searched in STEP 1
LINES = ["motor", "buildings", "contents"]
LEAKY = ["customer_service_contacts"]                 # leak: 3.4 contacts on claim rows, 0.4 elsewhere and in test
NOISE = ["previous_insurer", "campaign_code", "email_domain", "branch_id", "vehicle_colour",
         "marketing_consent", "document_preference", "marketing_segment", "surname"]   # no signal: dropped
MIN_LEAF = 40 if SMOKE else 3000                      # large leaves: smooth relativities, little overfitting
# LightGBM settings. LGB_PARAMS = model A (shallow Tweedie, power 1.5 like the metric); LINEB_PARAMS = model B
# (deeper); FREQ_PARAMS = Poisson on developed claim counts; SEV_PARAMS = Gamma on cost per claim. All use
# row and feature subsampling and L2 shrinkage; categorical levels need 2,000 rows (A) before they split.
LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=POWER, learning_rate=0.02,
                  num_leaves=7, min_data_in_leaf=MIN_LEAF,
                  feature_fraction=0.6,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=50, cat_smooth=200,
                  min_data_per_group=(40 if SMOKE else 2000), max_cat_threshold=8, seed=SEED, verbose=-1)
PERIL_PARAMS = {**LGB_PARAMS, "learning_rate": 0.04}
LINEB_PARAMS = {**LGB_PARAMS, "learning_rate": 0.03, "num_leaves": 15,
                "min_data_in_leaf": (40 if SMOKE else 1500), "feature_fraction": 0.5, "lambda_l2": 100}
_BASE_NO_TW = {k: v for k, v in LGB_PARAMS.items() if k != "tweedie_variance_power"}
FREQ_PARAMS = {**_BASE_NO_TW, "objective": "poisson", "learning_rate": 0.04,
               "num_leaves": 15, "min_data_in_leaf": (40 if SMOKE else 1500)}
SEV_PARAMS = {**_BASE_NO_TW, "objective": "gamma", "learning_rate": 0.04,
              "min_data_in_leaf": (30 if SMOKE else 400), "min_data_per_group": (30 if SMOKE else 300)}

def d2(y, p):                                         # the competition metric (predictions floored at 1e-6)
    return d2_tweedie_score(y, np.maximum(p, 1e-6), power=POWER)

def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

def full_year_end(s):                                 # v8 1: 2024 rows end 12-30, other years 12-31
    return s.dt.strftime("%m-%d").isin(["12-30", "12-31"])

# ------------------------------- load & clean ---------------------------------------
# Training (policies_*) and test (test_*) rows go through the same rules. qa collects one
# (split, line, issue, rows) entry per fix; it is printed as the DATA-QUALITY LOG at the end of the run.
ID_COLS = {c: str for c in ["row_id", "policy_id", "customer_id", "address_id", "claim_id"]}
def load(n):                                          # parquet, or csv with ids read as text
    p = DATA / f"{n}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(DATA / f"{n}.csv", dtype=ID_COLS, low_memory=False)
print("data dir:", DATA)
print(f"CONFIG: EXPERIMENT_ONLY={EXPERIMENT_ONLY} LINE_B={USE_LINE_B} PERIL_TREND={PERIL_TREND} BREAKS={BREAKS} "
      f"BI_DEV_CAP={BI_DEV_CAP} TIMING_FEATS={USE_TIMING_FEATS} TREND_MULT={TREND_MULT}")
cust, addr, wthr, claims_raw = load("customers"), load("addresses"), load("weather"), load("claims")
sample_sub = load("sample_submission")
qa = []

def clean_policies(df, line, split):
    df = df.copy()
    df["line"] = line
    df["period_start"] = pd.to_datetime(df.period_start)
    df["period_end"] = pd.to_datetime(df.period_end)
    df["year"] = df.period_start.dt.year                          # policy year = year the period starts
    # an exposure above one year was recorded in days: recompute it from the period dates
    days_exp = (((df.period_end - df.period_start).dt.days + 1) / 365).clip(upper=1)
    bad = df.exposure > 1.0001
    df.loc[bad, "exposure"] = days_exp[bad]
    df["exposure"] = df.exposure.clip(lower=1 / 365)              # at least one day
    qa.append((split, line, "exposure recorded in days -> recomputed from dates", int(bad.sum())))
    if "sum_insured" in df:                                       # property lines only
        big = df.sum_insured > 100 * df.sum_insured.median()      # entered about 1000x too large
        df.loc[big, "sum_insured"] /= 1000
        qa.append((split, line, "sum_insured ~1000x too large -> /1000", int(big.sum())))
    if "construction_type" in df:
        df["construction_type"] = df.construction_type.str.lower()   # same code in mixed case
    if "smoke_alarms" in df:
        df["smoke_alarms"] = df.smoke_alarms.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return df

pol = pd.concat([clean_policies(load(f"policies_{l}"), l, "train") for l in LINES], ignore_index=True)
tst = pd.concat([clean_policies(load(f"test_{l}"), l, "test") for l in LINES], ignore_index=True)

# v8 1: timing guardrail now uses the 12-30/12-31 rule; train-2024 and test should match
# The timing features only mean the same thing in training and test if these shares agree. v7 set
# ends_early on every 2024 row (22% in test); the assertion now stops a run with that kind of gap.
print("\n>>> TIMING GUARDRAIL (train-2024 vs test-2025, v8 full-year rule)")
for _l in LINES:
    _a, _t = pol[(pol.line == _l) & (pol.year == VALUATION.year)], tst[tst.line == _l]
    _ea, _et = (~full_year_end(_a.period_end)).mean(), (~full_year_end(_t.period_end)).mean()
    _sa = (_a.period_start.dt.strftime("%m-%d") != "01-01").mean()
    _st = (_t.period_start.dt.strftime("%m-%d") != "01-01").mean()
    print(f"  {_l:9s} ends_early {_ea:.4f} vs {_et:.4f} | starts_late {_sa:.4f} vs {_st:.4f} | "
          f"mean exposure {_a.exposure.mean():.4f} vs {_t.exposure.mean():.4f}")
    assert abs(_ea - _et) < 0.05, "ends_early distribution differs between train-2024 and test"
qa.append(("train", "all", "2024 full-year rows end 12-30 (not 12-31) -> treated as full year",
           int((pol.period_end.dt.strftime("%m-%d") == "12-30").sum())))

# Claims: net cost = incurred to date - recoveries (floored at 0), zero for repudiated claims; claims with
# no net cost are dropped. delay (days from loss to report) drives the late-reporting factors in STEP 2.
cl = claims_raw.copy()
cl["loss_date"] = pd.to_datetime(cl.loss_date)
cl["report_date"] = pd.to_datetime(cl.report_date)
cl["loss_year"] = cl.loss_date.dt.year
cl["delay"] = (cl.report_date - cl.loss_date).dt.days
cl["net"] = np.where(cl.status == "repudiated", 0.0, (cl.incurred_to_date - cl.recoveries).clip(lower=0))
cl = cl[cl.net > 0].copy()

# Link each claim to the row of its policy whose period contains the loss date (asserted unique). A claim
# outside every period goes to the nearest row of the same policy within NEAREST_DAYS, otherwise it is dropped.
_pcols = ["row_id", "policy_id", "period_start", "period_end"]
m_all = cl.merge(pol[_pcols], on="policy_id", how="inner")
m = m_all[(m_all.loss_date >= m_all.period_start) & (m_all.loss_date <= m_all.period_end)]
assert not m.claim_id.duplicated().any(), "claim matched to >1 policy row"
un = cl[~cl.claim_id.isin(m.claim_id)]
near = un.merge(pol[_pcols], on="policy_id", how="inner")
near["gap"] = np.where(near.loss_date < near.period_start, (near.period_start - near.loss_date).dt.days,
                       np.where(near.loss_date > near.period_end, (near.loss_date - near.period_end).dt.days, 0))
near = near.sort_values("gap").drop_duplicates("claim_id")       # nearest row per claim
near = near[near.gap <= NEAREST_DAYS].drop(columns="gap")
qa.append(("train", "all", "claims mapped to nearest policy row (<=31d gap)", int(len(near))))
cl_before = len(cl)
cl = pd.concat([m, near], ignore_index=True).drop(columns=["period_start", "period_end"])
assert not cl.claim_id.duplicated().any(), "duplicate claim after nearest mapping"
qa.append(("train", "all", "claims still not matched (dropped)", int(cl_before - len(cl))))
log(f"loaded: {len(pol):,} train rows, {len(tst):,} test rows, {len(cl):,} linked paid/reserved claims")

ALL_PERILS = sorted(cl.peril.unique())
YP_COLS = ["yp_" + p for p in ALL_PERILS]            # per-peril targets (developed capped cost per row)
PERILS = {l: sorted(cl.loc[cl.line == l, "peril"].unique()) for l in LINES}
print("perils by line:", PERILS)

# ---------------------- STEP 3a: large-loss caps & suspects --------------------------
# The models fit cost capped at the 99.5th percentile of its line and peril; the excess over the cap returns
# as a flat loading per line (excess / capped cost in training). A claim above SUSPECT_X times its 99.9th
# percentile is treated as a possible data error: capped, given no excess credit and printed for review.
caps = cl[cl.loss_year <= VALUATION.year].groupby(["line", "peril"]).net.agg(
    cap=lambda s: s.quantile(CAP_Q), p999=lambda s: s.quantile(0.999)).reset_index()
cl = cl.merge(caps, on=["line", "peril"], how="left")
cl["suspect"] = cl.net > SUSPECT_X * cl.p999
cl["net_cap"] = np.minimum(cl.net, cl.cap)
cl["net_xs"] = np.where(cl.suspect, 0.0, cl.net - cl.net_cap)
if cl.suspect.any():
    print("\n>>> SUSPECT CLAIMS - escalate for human review (capped, excluded from excess loading):")
    print(cl.loc[cl.suspect, ["claim_id", "policy_id", "line", "peril", "loss_year", "status",
                              "paid_to_date", "incurred_to_date", "net", "cap"]].to_string(index=False))

# v8 3: evidence for the BI break - severity quantiles by loss year (same maturity at 2022-Q4 vs 2023-Q1)
# Every quantile falls by about a quarter from 2023, while the quarters either side of the break have
# similar open shares: the claims changed, not their maturity.
_bi = cl[(cl.line == "motor") & (cl.peril == "bodily_injury") & (cl.loss_year >= 2021) & ~cl.suspect]
print("\n>>> MOTOR BI severity quantiles by loss year (break check):")
print(_bi.groupby("loss_year").net.quantile([0.25, 0.5, 0.75, 0.9]).unstack().round(0).to_string())
_q = _bi.groupby(_bi.loss_date.dt.to_period("Q")).agg(open_share=("status", lambda s: (s == "open").mean()),
                                                       median=("net", "median"))
print(_q.loc[[pd.Period("2022Q4"), pd.Period("2023Q1")]].round(2).to_string())

# ---------------------- STEP 2: development & trend ---------------------------------
# Claims are valued at 31 Dec 2024, so recent loss years are incomplete: some claims are not yet reported
# (IBNR) and open claims are still under-reserved (IBNER). Each claim is developed to ultimate, restated to
# the post-break basis where a break applies, and trended to the year being predicted.
def _peril_trend(fp, line_slope, brk_year):
    """v8 2/3: WLS of log mean severity on year (+ break dummy), shrunk to the line slope by credibility."""
    # fp = mature cells (one per loss year) of one line and peril. Returns the slope (log per year), the
    # break effect (log), the break year (9999 = none) and a0, the log severity level at 2021.
    has_brk = (fp.loss_year >= brk_year).any() and (fp.loss_year < brk_year).any()
    t = (fp.loss_year - 2021).values.astype(float)
    W = fp.n.values / np.maximum(fp.cv.fillna(1.0).values, 0.1) ** 2   # 1 / sampling variance of log mean
    cols = [np.ones_like(t), t] + ([(fp.loss_year >= brk_year).values.astype(float)] if has_brk else [])
    X = np.column_stack(cols)                                         # intercept, year, break dummy
    slope, brk = line_slope, 0.0
    if len(fp) >= X.shape[1]:
        A = X.T @ (X * W[:, None])
        b = np.linalg.solve(A, X.T @ (np.log(fp.sev.values) * W))     # weighted least squares
        se2 = np.linalg.inv(A)[1, 1]                                  # sampling variance of the slope
        # credibility Z = tau^2 / (tau^2 + se^2): a noisy peril slope is pulled towards its line's slope
        z = 0.0 if (has_brk and (fp.loss_year < brk_year).sum() < 2) else TREND_TAU ** 2 / (TREND_TAU ** 2 + se2)
        slope = float(np.clip(z * b[1] + (1 - z) * line_slope, -0.02, MAX_TREND))
        brk = float(b[2]) if has_brk else 0.0
        if has_brk and z == 0.0:                                     # slope not identified pre-break
            # break = weighted mean detrended level after the break minus the level before it
            pre = (fp.loss_year < brk_year).values
            lvl = np.log(fp.sev.values) - slope * t
            brk = float(np.average(lvl[~pre], weights=W[~pre]) - np.average(lvl[pre], weights=W[pre]))
    lvl = np.log(fp.sev.values) - slope * t - brk * (fp.loss_year.values >= brk_year)
    a0 = float(np.average(lvl, weights=W)) if len(fp) else np.nan
    return slope, (brk if has_brk else 0.0), (brk_year if has_brk else 9999), a0

def development_factors(cl, max_year):
    # Per line x peril x loss year: dev = IBNR x IBNER. Also returns the severity trend of each line.
    # IBNR: completeness = expected share of claims reported by the valuation date, from the reporting delays
    # of the mature loss years 2021-22, averaged over loss dates spread evenly through the loss year.
    rows = []
    for (line, peril), g in cl[cl.loss_year.between(2021, 2022)].groupby(["line", "peril"]):
        d = np.sort(g.delay.values)
        for ly in range(2021, VALUATION.year + 1):
            days = (VALUATION - pd.date_range(f"{ly}-01-01", f"{ly}-12-31")).days.values
            rows.append((line, peril, ly, np.searchsorted(d, days, side="right").mean() / len(d)))
    comp = pd.DataFrame(rows, columns=["line", "peril", "loss_year", "completeness"])

    # cell statistics: claim count, mean capped severity, its coefficient of variation, share still open
    cell = (cl[cl.loss_year >= 2021].groupby(["line", "peril", "loss_year"])
            .agg(n=("net", "size"), sev=("net_cap", "mean"), cv=("net_cap", lambda s: s.std() / s.mean()),
                 open_share=("status", lambda s: (s == "open").mean())).reset_index())
    cell = cell.merge(comp, on=["line", "peril", "loss_year"], how="left")
    cell["slope"], cell["brk"], cell["brk_year"] = np.nan, 0.0, 9999
    trends = {}
    for line, g in cell.groupby("line"):
        # line trend: log mean severity on peril dummies + year, weighted by sqrt(claims), on mature cells up
        # to max_year only (a validation fold never sees its own year), bounded to 0 .. +10% a year
        fit = g[(g.open_share <= MATURE_OPEN_SHARE) & (g.loss_year <= max_year)]
        X = pd.get_dummies(fit.peril).astype(float)
        X["t"] = fit.loss_year - 2021
        w = np.sqrt(fit.n.values)
        beta = np.linalg.lstsq(X.values * w[:, None], np.log(fit.sev.values) * w, rcond=None)[0]
        b = dict(zip(X.columns, beta))
        trends[line] = float(np.clip(b["t"], 0, MAX_TREND))
        idx = g.index
        if not PERIL_TREND:
            cell.loc[idx, "slope"] = trends[line]
            fitted = np.exp(g.peril.map(b).fillna(np.log(g.sev)).values + trends[line] * (g.loss_year.values - 2021))
        else:
            # v8: trend (and break) per peril, credibility-weighted towards the line trend
            fitted = pd.Series(np.nan, index=idx)
            for peril, gp in g.groupby("peril"):
                fp = gp[(gp.open_share <= MATURE_OPEN_SHARE) & (gp.loss_year <= max_year)]
                by = BREAKS.get((line, peril), 9999)
                slope, brk, byr, a0 = _peril_trend(fp, trends[line], by)
                if np.isnan(a0):
                    a0 = np.log(gp.sev.mean())
                cell.loc[gp.index, ["slope", "brk", "brk_year"]] = [slope, brk, byr]
                fitted[gp.index] = np.exp(a0 + slope * (gp.loss_year.values - 2021) + brk * (gp.loss_year.values >= byr))
            fitted = fitted.values
        # IBNER: an immature cell's mean severity is lifted to the fitted trend severity (factor 1 to 2)
        immature = (g.open_share > MATURE_OPEN_SHARE).values
        cell.loc[idx, "ibner"] = np.where(immature, np.clip(fitted / g.sev.values, 1, 2), 1.0)
    cell["ibnr"] = 1 / cell.completeness.clip(lower=0.3)              # at most 3.3x
    cell["dev"] = (cell.ibnr * cell.ibner).clip(upper=3)
    bi = (cell.line == "motor") & (cell.peril == "bodily_injury")
    cell.loc[bi, "dev"] = cell.loc[bi, "dev"].clip(upper=BI_DEV_CAP)  # not binding at 3.0 since v8
    return cell, trends

def build_targets(rows, cl, max_train_year, to_year):
    # Targets per policy row: y_cap = developed and trended capped cost (what the models fit), y_xs = the
    # same for the excess over the cap, y_raw = recorded net cost (for scoring), n_claims, n_dev = claims
    # grossed up for late reporting, yp_<peril> = y_cap by peril. Factors use loss years up to
    # max_train_year; costs are trended to to_year.
    cell, trends = development_factors(cl, max_train_year)
    c = cl.merge(cell[["line", "peril", "loss_year", "dev", "ibnr"]], on=["line", "peril", "loss_year"], how="left")
    c["dev"] = c.dev.fillna(1.0)
    c["n_dev"] = c.ibnr.fillna(1.0).clip(upper=3)                    # developed claim count (IBNR only)
    # v8 2/3: trend per peril; claims before a break year are restated to the post-break level
    pt = cell.drop_duplicates(["line", "peril"]).set_index(["line", "peril"])[["slope", "brk", "brk_year"]]
    k = pd.MultiIndex.from_arrays([c.line, c.peril])
    sl = pt.slope.reindex(k).fillna(c.line.map(trends)).values
    bk, byr = pt.brk.reindex(k).fillna(0).values, pt.brk_year.reindex(k).fillna(9999).values
    c["trend"] = np.exp(sl * c.line.map(TREND_MULT).values * (to_year - c.loss_year.values)
                        + bk * ((to_year >= byr).astype(float) - (c.loss_year.values >= byr).astype(float)))
    c["y_cap"] = c.net_cap * c.dev * c.trend
    c["y_xs"] = c.net_xs * c.dev * c.trend
    agg = c.groupby("row_id").agg(y_cap=("y_cap", "sum"), y_xs=("y_xs", "sum"),
                                  y_raw=("net", "sum"), n_claims=("net", "size"), n_dev=("n_dev", "sum"))
    out = rows.merge(agg, on="row_id", how="left")
    out[["y_cap", "y_xs", "y_raw", "n_claims", "n_dev"]] = out[["y_cap", "y_xs", "y_raw", "n_claims", "n_dev"]].fillna(0)
    pp = (c.pivot_table(index="row_id", columns="peril", values="y_cap", aggfunc="sum", fill_value=0)
            .reindex(columns=ALL_PERILS, fill_value=0))
    pp.columns = YP_COLS
    out = out.merge(pp, left_on="row_id", right_index=True, how="left")
    out[YP_COLS] = out[YP_COLS].fillna(0)
    return out, cell, trends

# ---------------------- STEP 4: features --------------------------------------------
ORD = {"credit_band": list("ABCDE"), "energy_rating": list("ABCDEFG"),           # ordered levels -> 0, 1, 2, ...
       "urban_density": ["rural", "suburban", "urban", "dense_urban"],
       "security_level": ["standard", "locks", "alarm_and_locks"]}

cust_f = cust.drop(columns=[c for c in NOISE if c in cust]).copy()
cust_f["birth_year"] = pd.to_datetime(cust_f.date_of_birth).dt.year
cust_f = cust_f.drop(columns=["date_of_birth"])
# weather: each year's value per address, the address's long-run average, and the anomaly (value - average)
wvars = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
         "wildfire_index", "rainfall_mm", "mean_temperature_c"]
w_mean = wthr.groupby("address_id")[wvars].mean().add_suffix("_avg")
wthr_f = wthr.merge(w_mean, left_on="address_id", right_index=True)
for v in wvars:
    wthr_f[v + "_anom"] = wthr_f[v] - wthr_f[v + "_avg"]
WCOLS = ["address_id", "year"] + wvars + [v + "_anom" for v in wvars]
print("weather years available:", sorted(wthr.year.unique()))

if REG_WEATHER:                                       # off: region-year mean anomalies
    _acols = [v + "_anom" for v in wvars]
    _wr = wthr_f[["address_id", "year"] + _acols].merge(addr[["address_id", "region"]], on="address_id", how="left")
    reg_w = (_wr.groupby(["region", "year"], observed=True)[_acols].mean()
                .add_suffix("_reg").reset_index())
    assert not reg_w.duplicated(["region", "year"]).any()

def history(panel):
    """Prior-year DEVELOPED claim counts by policy, customer and address-x-line. Strictly years < row year."""
    # Developed counts (n_dev) so that the recent, partly reported years are not under-counted.
    p = panel[["policy_id", "customer_id", "address_id", "line", "year", "exposure", "n_dev"]].copy()
    p["n_claims"] = p.n_dev.fillna(0)                 # test rows carry no claims, only exposure
    p["addr_line"] = p.address_id.astype(str) + "|" + p.line
    feats = []
    for key, pre in [("policy_id", "pol"), ("customer_id", "cust"), ("addr_line", "addr")]:
        g = p.groupby([key, "year"]).agg(n=("n_claims", "sum"), e=("exposure", "sum")).reset_index()
        g = g.sort_values([key, "year"])
        g[f"{pre}_hist_n"] = g.groupby(key).n.cumsum() - g.n      # claims in all earlier years
        g[f"{pre}_hist_e"] = g.groupby(key).e.cumsum() - g.e      # exposure in all earlier years
        prev = g[[key, "year", "n"]].assign(year=g.year + 1).rename(columns={"n": f"{pre}_prev_n"})   # last year
        g = g.merge(prev, on=[key, "year"], how="left")
        feats.append((key, g.drop(columns=["n", "e"])))
    lines_held = p.groupby(["customer_id", "year"]).line.nunique().rename("cust_lines_held").reset_index()
    return feats, lines_held

def make_features(df, hist, lines_held):
    # Joins customer, address, weather and claim history to the policy rows and derives the rating features.
    n0 = len(df)
    f = df.merge(cust_f, on="customer_id", how="left").merge(addr, on="address_id", how="left")
    f = f.merge(wthr_f[WCOLS], on=["address_id", "year"], how="left")
    f = f.merge(w_mean, left_on="address_id", right_index=True, how="left")
    if REG_WEATHER:
        f = f.merge(reg_w, on=["region", "year"], how="left")
    miss_share = float(f[wvars[0]].isna().mean())
    if miss_share > 0:                                # a year missing from the weather table: use the average
        print(f"   weather missing for {miss_share:.1%} of {n0:,} rows -> filled with address average")
        for v in wvars:
            miss = f[v].isna()
            f.loc[miss, v] = f.loc[miss, v + "_avg"]
            f.loc[miss, v + "_anom"] = np.where(f.loc[miss, v + "_avg"].notna(), 0.0, np.nan)
    f["addr_line"] = f.address_id.astype(str) + "|" + f.line
    for key, g in hist:
        f = f.merge(g, on=[key, "year"], how="left")
    f = f.merge(lines_held, on=["customer_id", "year"], how="left")
    assert len(f) == n0, "feature merge changed row count (duplicate keys)"
    for pre in ["pol", "cust", "addr"]:               # claims per exposure-year, smoothed towards 0.1
        f[f"{pre}_hist_freq"] = (f[f"{pre}_hist_n"] + 0.1) / (f[f"{pre}_hist_e"] + 1)
    for col, levels in ORD.items():
        if col in f:
            f[col] = f[col].map({v: i for i, v in enumerate(levels)})
    f["log_premium"] = np.log(f.written_premium)
    f["ends_early"] = (~full_year_end(f.period_end)).astype(int)            # v8 1: was != "12-31"
    f["starts_late"] = (f.period_start.dt.strftime("%m-%d") != "01-01").astype(int)
    f["rows_in_policy_year"] = f.groupby(["policy_id", "year"]).row_id.transform("size")   # mid-term changes
    f["customer_tenure"] = f.year - f.customer_since_year
    f["property_age"] = f.year - f.year_built
    # physical interactions: subsidence (clay soil x dry summer x trees), flood (elevation per km to a river),
    # burst pipes (frost x property age), hail on flat roofs, wind on old buildings
    f["clay_dry_trees"] = (f.soil_type == "clay").astype(float) * f.dry_summer_index * f.trees_within_5m
    f["elev_over_river"] = f.elevation_m / (f.distance_to_river_km.clip(lower=0) + 0.1)
    f["frost_x_age"] = f.freezing_days * f.property_age
    f["hail_x_flat_roof"] = f.hail_days * (f.roof_type == "flat").astype(float)
    f["wind_x_age"] = f.wind_exposure_index * f.property_age
    f["total_excess"] = f.voluntary_excess + f.compulsory_excess
    if "sum_insured" in f:
        f["excess_to_si"] = (f.total_excess / f.sum_insured).replace([np.inf, -np.inf], np.nan)
    if "vehicle_value_group" in f:
        f["excess_per_vvg"] = f.total_excess / f.vehicle_value_group.clip(lower=1)
    if "driver_age" in f:
        f["age_first_licensed"] = f.driver_age - f.years_licensed
        f["ncb_ratio"] = f.ncb_years / f.years_licensed.clip(lower=1)
    if "sum_insured" in f:
        f["log_si"] = np.log(f.sum_insured)
        f["si_per_bedroom"] = f.sum_insured / f.bedrooms
    # premium and sum insured relative to their line-year median: the target is already trended, so raw
    # money values would count inflation twice, and 2025 values would sit above the training range
    grp = [f.line, f.year]
    f["log_premium"] = f.log_premium - f.groupby(grp).log_premium.transform("median")
    if "log_si" in f:
        f["log_si"] = f.log_si - f.groupby(grp).log_si.transform("median")
        f["si_per_bedroom"] = f.si_per_bedroom / f.groupby(grp).si_per_bedroom.transform("median")
    return f

def target_encode(tr, ap, col, k=5):
    # Credibility-weighted claim frequency per level of col (used for make_model):
    # (claims + TE_SMOOTH x overall frequency) / (exposure + TE_SMOOTH). Training rows get out-of-fold
    # values (5 folds grouped by policy), so no row is encoded with its own claims.
    prior = tr.n_claims.sum() / tr.exposure.sum()
    def enc(src, dst):
        s = src.groupby(col).agg(n=("n_claims", "sum"), e=("exposure", "sum"))
        r = (s.n + TE_SMOOTH * prior) / (s.e + TE_SMOOTH)
        return dst[col].map(r).fillna(prior).values
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(k).split(tr, groups=tr.policy_id):
        oof[b] = enc(tr.iloc[a], tr.iloc[b])
    return oof, enc(tr, ap)

# Never features: ids, dates, targets, the leak and noise fields, raw values replaced by derived ones
# (premium, sum insured, make_model, years), renewal_decision (only known after the period) and the
# weather averages (the yearly levels and anomalies carry them).
DROP = set(LEAKY + NOISE + YP_COLS + ["row_id", "policy_id", "customer_id", "address_id", "period_start",
           "period_end", "renewal_decision", "y_cap", "y_xs", "y_raw", "n_claims", "n_dev", "addr_line",
           "line", "year", "exposure", "written_premium", "make_model",
           "customer_since_year", "year_built", "sum_insured"] + [v + "_avg" for v in wvars])
if not USE_TIMING_FEATS:
    DROP |= {"ends_early", "starts_late", "rows_in_policy_year"}

# ---------------------- STEP 4b: leakage / shift guardrail -------------------------
# Population stability index (PSI) of every numeric feature: 2024 training rows against the 2025 test rows,
# and 2024 rows WITHOUT claims against the test rows. A field filled in after a claim makes test rows look
# like the claim-free rows (large first PSI, small second): that is the leak signature. The run stops if the
# known leak is not found; any other feature with the signature is dropped and printed for review.
def psi(a, b, bins=10):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if a.nunique() <= 1 or len(b) == 0: return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))        # deciles of the reference sample
    if len(edges) < 3: edges = np.unique(np.r_[a.min() - 1, np.sort(a.unique())[:bins], a.max() + 1])
    pa = np.histogram(a, edges)[0] / len(a) + 1e-4
    pb = np.histogram(b.clip(edges[0], edges[-1]), edges)[0] / len(b) + 1e-4
    return float(np.sum((pa - pb) * np.log(pa / pb)))

_y24, _, _ = build_targets(pol[pol.year == VALUATION.year], cl, VALUATION.year, VALUATION.year)
num_cols = [c for c in pol.select_dtypes("number").columns if c not in ("year", "exposure") and c not in YP_COLS]
shift = []
for line in LINES:
    a, t = _y24[_y24.line == line], tst[tst.line == line]
    for c in num_cols:
        if a[c].notna().any():
            shift.append((line, c, psi(a[c], t[c]), psi(a.loc[a.n_claims == 0, c], t[c])))
shift = pd.DataFrame(shift, columns=["line", "feature", "psi_train_vs_test", "psi_noclaim_vs_test"])
shift["leak_signature"] = (shift.psi_train_vs_test > 0.02) & (shift.psi_noclaim_vs_test < 0.5 * shift.psi_train_vs_test)
print("\n>>> SHIFT GUARDRAIL (train-2024 vs test-2025; leak = test looks like *no-claim* train rows):")
print(shift.sort_values("psi_train_vs_test", ascending=False).head(8).round(4).to_string(index=False))
flagged = sorted(shift.loc[shift.leak_signature, "feature"].unique())
if not SMOKE:
    assert set(LEAKY) <= set(flagged), "expected leak not detected - re-check data"
extra = [c for c in flagged if c not in LEAKY]
if extra:
    print(f">>> additional features with leak signature (auto-dropped, review!): {extra}")
    DROP |= set(extra)
del _y24

# ---------------------- model harness ----------------------------------------------
def prep(tr, ap):
    # Encodes make_model, builds the feature list (everything not in DROP), gives text columns the same
    # categories in training and apply rows, and stops if a target column slipped into the features.
    tr, ap = tr.copy(), ap.copy()
    if "make_model" in tr:
        tr["make_model_te"], ap["make_model_te"] = target_encode(tr, ap, "make_model")
    feats = [c for c in tr.columns if c not in DROP]
    for c in feats:
        if tr[c].dtype == object or tr[c].dtype == bool or str(tr[c].dtype) in ("category", "str", "string"):
            cats = pd.Categorical(pd.concat([tr[c], ap[c]]).astype(str)).categories
            tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
            ap[c] = pd.Categorical(ap[c].astype(str), categories=cats)
    leaked = [c for c in feats if c.startswith("yp_") or c in ("y_cap", "y_xs", "y_raw", "n_claims", "n_dev")]
    assert not leaked, f"target columns in feature list: {leaked}"
    return tr, ap, feats

def _recency_w(df):                                   # off while RECENCY = 1.0
    if RECENCY >= 1.0:
        return None
    return (RECENCY ** (df.year.max() - df.year.values)).astype(float)

def fit_line(tr, feats, n_rounds=None, seed=SEED, target="y_cap", params_base=None):
    """LightGBM with log(exposure)+log(base rate) offset. target = column to model."""
    # The offset makes the trees learn relativities to the average cost per exposure-year.
    params = {**(params_base or LGB_PARAMS), "seed": seed}
    y = tr[target].values
    base = np.log(y.sum() / tr.exposure.sum())
    off = np.log(tr.exposure.values) + base
    wt = _recency_w(tr)
    if n_rounds is None:
        # early stopping on 15% of policies held out (grouped, so a policy is never on both sides)
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(tr, groups=tr.policy_id))
        dtr = lgb.Dataset(tr[feats].iloc[a], y[a], weight=None if wt is None else wt[a], init_score=off[a])
        dva = lgb.Dataset(tr[feats].iloc[b], y[b], weight=None if wt is None else wt[b], init_score=off[b])
        mdl = lgb.train(params, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        n_rounds = max(int(mdl.best_iteration), 30)
    else:                                             # fixed number of rounds (final fit)
        mdl = lgb.train(params, lgb.Dataset(tr[feats], y, weight=wt, init_score=off), n_rounds)
    mdl.base = base
    mdl.bal = y.sum() / np.exp(mdl.predict(tr[feats], raw_score=True, num_iteration=n_rounds) + off).sum()   # balance
    mdl.n_rounds = n_rounds
    return mdl, n_rounds

def predict(mdl, df, feats, loading):
    # expected cost = exp(score + offset) x balance factor x (1 + large-loss loading)
    raw = mdl.predict(df[feats], raw_score=True, num_iteration=mdl.n_rounds)
    return np.exp(raw + mdl.base + np.log(df.exposure.values)) * mdl.bal * (1 + loading)

def run_peril(tr, ap, feats, line, n_rounds_map=None, seed=SEED):
    # one Tweedie model per peril, summed; perils with few claims get their average rate (off in v8)
    total = np.zeros(len(ap)); rounds = {}
    for p in PERILS[line]:
        tgt = f"yp_{p}"
        npos = int((tr[tgt] > 0).sum())
        if npos < MIN_PERIL_CLAIMS:
            total += tr[tgt].sum() / tr.exposure.sum() * ap.exposure.values
            rounds[p] = 0
            continue
        nr = (n_rounds_map or {}).get(p, 0) or None
        mdl, used = fit_line(tr, feats, n_rounds=nr, seed=seed, target=tgt, params_base=PERIL_PARAMS)
        total += predict(mdl, ap, feats, 0.0)
        rounds[p] = used
    return total, rounds

def fit_sev(tr, feats, n_rounds=None, seed=SEED):
    """Gamma on cost per developed claim, weighted by developed claim count (x recency weight if used)."""
    # Rows with claims only; early stopping on held-out policies; the balance factor keeps the
    # claim-weighted mean severity equal to the actual one.
    d = tr[tr.n_claims > 0]
    y = (d.y_cap / d.n_dev).values
    w = d.n_dev.values
    wt = _recency_w(d)
    w_fit = w if wt is None else w * wt
    base = np.log((w * y).sum() / w.sum())
    params = {**SEV_PARAMS, "seed": seed}
    X = d[feats]
    if n_rounds is None:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(d, groups=d.policy_id))
        dtr = lgb.Dataset(X.iloc[a], y[a], weight=w_fit[a], init_score=np.full(len(a), base))
        dva = lgb.Dataset(X.iloc[b], y[b], weight=w_fit[b], init_score=np.full(len(b), base))
        mdl = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        n_rounds = max(int(mdl.best_iteration), 20)
    else:
        mdl = lgb.train(params, lgb.Dataset(X, y, weight=w_fit, init_score=np.full(len(d), base)), n_rounds)
    mdl.base = base
    pr = np.exp(mdl.predict(X, raw_score=True, num_iteration=n_rounds) + base)
    mdl.bal = (w * y).sum() / (w * pr).sum()
    mdl.n_rounds = n_rounds
    return mdl, n_rounds

def predict_sev(mdl, df, feats):
    return np.exp(mdl.predict(df[feats], raw_score=True, num_iteration=mdl.n_rounds) + mdl.base) * mdl.bal

def run_fs(tr, ap, feats, loading, nr=None, seed=SEED):
    # expected cost = frequency (Poisson on developed claim counts) x severity (Gamma) x (1 + loading)
    nr = nr or {}
    mf, uf = fit_line(tr, feats, n_rounds=nr.get("f") or None, seed=seed, target="n_dev", params_base=FREQ_PARAMS)
    ms, us = fit_sev(tr, feats, n_rounds=nr.get("s") or None, seed=seed)
    pred = predict(mf, ap, feats, 0.0) * predict_sev(ms, ap, feats) * (1 + loading)
    return pred, {"f": uf, "s": us}

# =====================================================================================
# STEP 1: rolling-origin validation
# Fold 2023 trains on 2021-22 and fold 2024 on 2021-23; development factors and trends use only the years
# before the fold. Scores: D2 against recorded claims, recorded claims with suspects capped, and developed
# claims (2024 is only partly reported at the valuation date). Benchmarks: exposure x average cost and
# written premium x loss ratio. The early-stopping rounds found here are reused in the final fit.
# =====================================================================================
def run_fold(val_year, verbose=True):
    log(f"validation fold: train <{val_year} -> score {val_year}")
    train_rows = pol[pol.year < val_year]
    val_rows = pol[pol.year == val_year]
    tr_all, _, _ = build_targets(train_rows, cl, val_year - 1, val_year)
    va_all, _, _ = build_targets(val_rows, cl, val_year - 1, val_year)
    hist, held = history(pd.concat([tr_all, va_all]))
    tr_all, va_all = make_features(tr_all, hist, held), make_features(va_all, hist, held)
    res, best, best_b, best_p, best_f, imps = [], {}, {}, {}, {}, {}
    for line in LINES:
        tr, va = tr_all[tr_all.line == line], va_all[va_all.line == line]
        loading = tr.y_xs.sum() / tr.y_cap.sum()                      # large-loss loading of the line
        tr, va, feats = prep(tr, va)
        mdl, best[line] = fit_line(tr, feats)
        imps[line] = pd.Series(mdl.feature_importance("gain"), index=feats).sort_values(ascending=False)
        pred_A = predict(mdl, va, feats, loading)
        if USE_LINE_B:
            mdl_b, best_b[line] = fit_line(tr, feats, params_base=LINEB_PARAMS)
            pred_B = predict(mdl_b, va, feats, loading)
            pred = 0.5 * (pred_A + pred_B)                            # line model = mean of A and B
        else:
            best_b[line], pred_B, pred = 0, pred_A, pred_A
        if USE_PERIL_MODELS:
            pp, best_p[line] = run_peril(tr, va, feats, line)
            pred_peril = pp * (1 + loading)
        else:
            best_p[line], pred_peril = {}, pred
        if USE_FS_MODEL:
            pred_fs, best_f[line] = run_fs(tr, va, feats, loading)
            pred_fs = np.clip(pred_fs, FS_CLIP[0] * pred, FS_CLIP[1] * pred)
        else:
            best_f[line], pred_fs = {}, pred
        rate = (tr.y_cap.sum() + tr.y_xs.sum()) / tr.exposure.sum()     # benchmark: average cost
        lr = (tr.y_cap.sum() + tr.y_xs.sum()) / tr.written_premium.sum()   # benchmark: loss ratio
        res.append(pd.DataFrame({"line": line, "y": va.y_raw.values,
                                 "y_dev": (va.y_cap + va.y_xs).values,
                                 "pred_A": pred_A, "pred_B": pred_B,
                                 "pred": pred, "pred_peril": pred_peril, "pred_fs": pred_fs,
                                 "exp_x_rate": va.exposure.values * rate,
                                 "prem_x_lr": va.written_premium.values * lr, "row_id": va.row_id.values}))
        log(f"  {line:9s} rounds A={best[line]:4d} B={best_b[line]:4d} fs rounds={best_f[line]} "
            f"excess loading={loading:.3%} balance factor={mdl.bal:.3f}")
    V = pd.concat(res, ignore_index=True)
    # actuals with the excess of suspect claims removed (the "suspect capped" score)
    sus = cl[cl.suspect & (cl.loss_year == val_year)].groupby("row_id").apply(lambda g: (g.net - g.cap).sum())
    V["y_nosuspect"] = V.y - V.row_id.map(sus).fillna(0) if len(sus) else V.y
    if verbose:
        print(f"\n>>> VALIDATION FOLD {val_year} (Tweedie D2 power 1.5; raw actuals for 2024 are immature)")
        tab = {}
        names = ["exp_x_rate", "prem_x_lr", "pred_A", "pred_B", "pred", "pred_fs"]
        for name in names:
            r = {l: d2(g.y, g[name]) for l, g in V.groupby("line")}
            r["COMBINED"] = d2(V.y, V[name])
            r["COMBINED (suspect capped)"] = d2(V.y_nosuspect, V[name])
            r["COMBINED vs developed"] = d2(V.y_dev, V[name])                 # v8 5
            tab[name] = r
        print(pd.DataFrame(tab).T.round(4).to_string())
        print("A/E line model:", (V.groupby("line").pred.sum() / V.groupby("line").y_dev.sum()).round(3).to_dict())
        print("A/E freq x sev:", (V.groupby("line").pred_fs.sum() / V.groupby("line").y_dev.sum()).round(3).to_dict())
        for line in LINES:
            print(f"top features {line}:", (imps[line] / imps[line].sum()).head(8).round(3).to_dict())
    return V, best, best_b, best_p, best_f

folds, best_iters, best_iters_b, best_peril, best_fs = {}, {}, {}, {}, {}
for vy in VAL_YEARS:
    folds[vy], best_iters[vy], best_iters_b[vy], best_peril[vy], best_fs[vy] = run_fold(vy)

def level_neutral_d2(V, col):
    # each line rescaled to the fold's developed actual total, so a level error cannot choose the blend
    s = V.groupby("line").y_dev.sum() / V.groupby("line")[col].sum()
    return d2(V.y_nosuspect, V[col] * V.line.map(s))

# Blend weights (line model, peril models, freq x sev) from the grid, chosen on the mean level-neutral D2
# of the two folds.
combos = []
for wp, wf in product(WGRID if USE_PERIL_MODELS else [0.0], WGRID if USE_FS_MODEL else [0.0]):
    if wp + wf <= 1.0001:
        combos.append((round(1 - wp - wf, 2), wp, wf))
blend_scores = {}
for wl, wp, wf in combos:
    sc = []
    for vy, V in folds.items():
        V["blend"] = wl * V.pred + wp * V.pred_peril + wf * V.pred_fs
        sc.append(level_neutral_d2(V, "blend"))
    blend_scores[(wl, wp, wf)] = float(np.mean(sc))
W_LINE, W_PERIL, W_FS = max(blend_scores, key=blend_scores.get)
print("\n>>> BLEND (line, peril, freq-sev) -> level-neutral mean D2 over both folds (top 8):")
for k, v in sorted(blend_scores.items(), key=lambda kv: -kv[1])[:8]:
    print(f"   {k}: {v:.5f}")
print(f">>> CHOSEN weights: line={W_LINE} peril={W_PERIL} freq-sev={W_FS}")
for vy, V in folds.items():
    V["pred_final"] = W_LINE * V.pred + W_PERIL * V.pred_peril + W_FS * V.pred_fs

# Line calibration from the folds (actual developed / predicted). Applied only when both folds miss in the
# same direction: the smaller miss, shrunk to 75% and bounded to +-10%. Otherwise the line keeps 1.0.
fold_factors = {vy: (V.groupby("line").y_dev.sum() / V.groupby("line").pred_final.sum()).to_dict()
                for vy, V in folds.items()}
CAL = {}
for line in LINES:
    fs_ = [fold_factors[vy][line] for vy in VAL_YEARS]
    if all(f > 1 for f in fs_) or all(f < 1 for f in fs_):
        f = min(fs_, key=lambda x: abs(x - 1))
        CAL[line] = float(np.clip(1 + CAL_SHRINK * (f - 1), *CAL_BOUNDS))
    else:
        CAL[line] = 1.0
print("\n>>> CALIBRATION (actual developed / predicted) per fold:",
      {vy: {k: round(v, 3) for k, v in d.items()} for vy, d in fold_factors.items()})
print(f">>> CALIBRATION APPLIED (agree-in-direction, shrunk {CAL_SHRINK:.0%}, bounded):", {k: round(v, 3) for k, v in CAL.items()})
print(">>> v7 REFERENCE (public LB 0.12811): fold 2023 blend cal=0.13838, fold 2024 blend cal=0.13277")
_tot = []
for vy, V in folds.items():
    pc = V.pred_final * V.line.map(CAL)
    _s = d2(V.y_nosuspect, pc); _tot.append(_s)
    print(f"  fold {vy}: line-model A={d2(V.y_nosuspect, V.pred_A):.5f}  B={d2(V.y_nosuspect, V.pred_B):.5f}  "
          f"A+B={d2(V.y_nosuspect, V.pred):.5f}  blend uncal={d2(V.y_nosuspect, V.pred_final):.5f}  blend cal={_s:.5f}  "
          f"vs developed={d2(V.y_dev, pc):.5f}")
print(f">>> MEAN blend-cal D2 over folds: {np.mean(_tot):.5f}   (v7 reference mean = 0.13558)")

# =====================================================================================
# Final fit 2021-24 -> 2025 submission
# Targets use all four years, trended to 2025. Each model runs 1.15x the mean early-stopping rounds of the
# two folds (the final fit has more data than either fold) and is averaged over three seeds. A line whose
# 2025 cost per exposure is outside LEVEL_BAND x its trended 2024 cost is flagged for human review.
# =====================================================================================
if MAKE_SUBMISSION:
    log("final fit: train 2021-2024 -> predict 2025")
    full, cell_f, trend_f = build_targets(pol, cl, VALUATION.year, TARGET_YEAR)
    assert np.allclose(full[YP_COLS].sum(axis=1).values, full.y_cap.values, rtol=1e-6, atol=1e-6), "peril targets do not add up"
    _c = full[full.n_claims > 0]
    assert np.allclose((_c.n_dev * (_c.y_cap / _c.n_dev)).values, _c.y_cap.values), "freq x sev identity broken"
    assert (_c.n_dev > 0).all(), "rows with claims but zero developed count"
    # test rows carry no targets; they join the panel only so their history features can be built
    tst_rows = tst.assign(y_cap=np.nan, y_xs=np.nan, y_raw=np.nan, n_claims=np.nan, n_dev=np.nan,
                          **{c: np.nan for c in YP_COLS})
    hist, held = history(pd.concat([full, tst_rows]))
    full, tst_f = make_features(full, hist, held), make_features(tst_rows, hist, held)
    assert full.loc[full.year == 2024, "ends_early"].mean() < 0.5, "v8 timing fix not applied"
    # diagnostics for review: development factors, trends and breaks, developed experience by line and year
    print("\n>>> STEP 2 - development factors used for training (immature cells only):")
    print(cell_f[cell_f.dev > 1.02][["line", "peril", "loss_year", "n", "open_share", "completeness", "ibnr", "ibner", "dev"]]
          .round(3).to_string(index=False))
    print("fitted line severity trend p.a.:", {k: f"{np.exp(v) - 1:.2%}" for k, v in trend_f.items()})
    _pt = cell_f.drop_duplicates(["line", "peril"])[["line", "peril", "slope", "brk", "brk_year"]].copy()
    _pt["trend_pa"] = (np.exp(_pt.slope) - 1).round(4)
    _pt["break_effect"] = (np.exp(_pt.brk) - 1).round(4)
    print("v8 per-peril severity trend (credibility-shrunk) and structural breaks:")
    print(_pt.drop(columns=["slope", "brk"]).to_string(index=False))
    dg = full.groupby(["line", "year"]).agg(exposure=("exposure", "sum"), n_dev=("n_dev", "sum"),
                                            y_cap=("y_cap", "sum"), prem=("written_premium", "sum")).reset_index()
    dg["dev_freq"] = dg.n_dev / dg.exposure
    dg["cost_per_exp(2025 level)"] = dg.y_cap / dg.exposure
    dg["premium_per_exp"] = dg.prem / dg.exposure
    print("\n>>> TREND DIAGNOSTIC (developed claim frequency & cost per exposure-year by line/year):")
    print(dg.drop(columns=["n_dev", "y_cap", "prem"]).round(4).to_string(index=False))

    preds_raw, preds_cal = [], []
    for line in LINES:
        tr, ap = full[full.line == line], tst_f[tst_f.line == line]
        loading = tr.y_xs.sum() / tr.y_cap.sum()
        tr, ap, feats = prep(tr, ap)
        p_line = p_peril = p_fs = None
        bals = []
        if W_LINE > 0.0 or W_FS > 0.0:
            # line model: A and B averaged, then averaged over the seeds
            n_a = int(np.mean([best_iters[vy][line] for vy in VAL_YEARS]) * 1.15)
            n_b = int(np.mean([best_iters_b[vy][line] for vy in VAL_YEARS]) * 1.15) if USE_LINE_B else 0
            ps = []
            for s in SEEDS:
                mdl, _ = fit_line(tr, feats, n_rounds=n_a, seed=s)
                pa = predict(mdl, ap, feats, loading); bals.append(mdl.bal)
                if USE_LINE_B:
                    mdl_b, _ = fit_line(tr, feats, n_rounds=n_b, seed=s, params_base=LINEB_PARAMS)
                    pa = 0.5 * (pa + predict(mdl_b, ap, feats, loading))
                ps.append(pa)
            p_line = np.mean(ps, axis=0)
        if W_PERIL > 0.0:
            rm = {}
            for p in PERILS[line]:
                rr = [best_peril[vy][line].get(p, 0) for vy in VAL_YEARS]
                rr = [r for r in rr if r > 0]
                rm[p] = int(np.mean(rr) * 1.15) if rr else 0
            pp, _ = run_peril(tr, ap, feats, line, n_rounds_map=rm, seed=SEED)
            p_peril = pp * (1 + loading)
        if W_FS > 0.0:
            nr = {k: int(np.mean([best_fs[vy][line][k] for vy in VAL_YEARS]) * 1.15) for k in ("f", "s")}
            p_fs = np.mean([run_fs(tr, ap, feats, loading, nr=nr, seed=s)[0] for s in SEEDS], axis=0)
            p_fs = np.clip(p_fs, FS_CLIP[0] * p_line, FS_CLIP[1] * p_line)
        p_raw = (W_LINE * (p_line if p_line is not None else 0.0)
                 + W_PERIL * (p_peril if p_peril is not None else 0.0)
                 + W_FS * (p_fs if p_fs is not None else 0.0))
        # level check: predicted 2025 cost per exposure against the 2024 developed cost at 2025 level
        hist_pp = tr[tr.year == VALUATION.year].pipe(lambda d: (d.y_cap.sum() + d.y_xs.sum()) / d.exposure.sum())
        lvl_raw = p_raw.sum() / ap.exposure.sum() / hist_pp
        p_cal = p_raw * CAL[line]
        lvl_cal = p_cal.sum() / ap.exposure.sum() / hist_pp
        use_cal = CAL[line]
        if not (LEVEL_BAND[0] <= lvl_cal <= LEVEL_BAND[1]) and (LEVEL_BAND[0] <= lvl_raw <= LEVEL_BAND[1]):
            print(f"  !! {line}: calibrated level ratio {lvl_cal:.2f} outside band -> calibration dropped")
            p_cal, use_cal = p_raw, 1.0
        if not (LEVEL_BAND[0] <= lvl_raw <= LEVEL_BAND[1]):
            print(f"  !! WARNING {line}: uncalibrated 2025 level is {lvl_raw:.2f}x 2024@2025 -> HUMAN REVIEW before submitting")
        preds_raw.append(pd.DataFrame({"row_id": ap.row_id.values, "line": line, "prediction": p_raw}))
        preds_cal.append(pd.DataFrame({"row_id": ap.row_id.values, "line": line, "prediction": p_cal}))
        bal_txt = f"{np.mean(bals):.3f}" if bals else "n/a"
        log(f"  {line:9s} 2025 cost/exp raw={p_raw.sum()/ap.exposure.sum():7.1f} cal={p_cal.sum()/ap.exposure.sum():7.1f} "
            f"| 2024@2025={hist_pp:7.1f} | ratio raw={lvl_raw:.2f} | cal factor={use_cal:.3f} | loading={loading:.3%} bal={bal_txt}")
        comps = {k: v for k, v in [("line", p_line), ("peril", p_peril), ("fs", p_fs)] if v is not None}
        if len(comps) > 1:                            # how far the blended components disagree
            ks = list(comps)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a_, b_ = comps[ks[i]], comps[ks[j]]
                    print(f"     {ks[i]} vs {ks[j]}: level ratio={b_.sum() / a_.sum():.3f}  correlation={np.corrcoef(a_, b_)[0, 1]:.4f}")

    def finish(preds, path=None):
        # sample-submission order; stops on a wrong row count, unknown or missing ids, or non-finite values
        sub = sample_sub[["row_id"]].merge(pd.concat(preds)[["row_id", "prediction"]], on="row_id", how="left")
        assert len(sub) == len(sample_sub), "row count mismatch"
        assert sub.row_id.is_unique and set(sub.row_id) == set(sample_sub.row_id), "row_id mismatch"
        assert sub.prediction.notna().all(), "missing predictions"
        assert np.isfinite(sub.prediction).all(), "non-finite predictions"
        sub["prediction"] = sub.prediction.clip(lower=0.01).round(4)
        assert (sub.prediction > 0).all()
        if path is not None:
            sub.to_csv(path, index=False)
            log(f"wrote {path}  rows={len(sub):,}  mean={sub.prediction.mean():.2f}  max={sub.prediction.max():.0f}")
        return sub
    sub_raw = finish(preds_raw)
    sub_cal = finish(preds_cal)
    print("\ncorrelation raw vs calibrated:", round(float(np.corrcoef(sub_raw.prediction, sub_cal.prediction)[0, 1]), 5),
          "| mean ratio cal/raw:", round(float(sub_cal.prediction.mean() / sub_raw.prediction.mean()), 4))
    both = pd.concat(preds_raw).rename(columns={"prediction": "p_raw"}).merge(
        pd.concat(preds_cal).rename(columns={"prediction": "p_cal"})[["row_id", "p_cal"]], on="row_id")
    both.to_parquet(OUT_BOTH, index=False)
    print(f">>> USE_CALIBRATION={USE_CALIBRATION} -> writing {'calibrated' if USE_CALIBRATION else 'raw'} predictions to {OUT_CSV}")
    final_sub = finish(preds_cal if USE_CALIBRATION else preds_raw, OUT_CSV)
else:
    print("\n>>> EXPERIMENT_ONLY: validation folds done, no final fit, no csv written.")

print("\n>>> DATA-QUALITY LOG"); print(pd.DataFrame(qa, columns=["split", "line", "issue", "rows"]).to_string(index=False))
log("done")
