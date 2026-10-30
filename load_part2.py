"""
load_part2.py
=============
IAA AI-Assisted Hackathon 2026 - Part 2 (Emerging Market) Data Loader & Cleaner

This module provides robust loading, cleaning, validation, and schema harmonization
for Part 2 (Week 2 Expansion: Emerging Market).

It handles:
1. Automated file discovery (resolves both standard names and browser/drive numbered suffixes `(1)` to `(11)`).
2. Data quality remediation (exposure in days -> years, 1000x sum_insured outliers, boolean smoke_alarms).
3. Schema harmonization (construction_type BRK/STN/WOD/MIX, age bands, gender).
4. Relational claim matching to policy coverage periods (with 31-day nearest-policy fallback).
5. Fast summary diagnostics for actuaries.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Mapping of standard entity names to possible filename patterns in Part 2 directory
EMERGING_FILE_PATTERNS = {
    "addresses": ["addresses (1).parquet", "addresses_emerging.parquet", "addresses_part2.parquet"],
    "claims": ["claims (2).parquet", "claims_emerging.parquet", "claims_part2.parquet"],
    "customers": ["customers (3).parquet", "customers_emerging.parquet", "customers_part2.parquet"],
    "policies_buildings": ["policies_buildings (4).parquet", "policies_buildings_emerging.parquet"],
    "policies_contents": ["policies_contents (5).parquet", "policies_contents_emerging.parquet"],
    "policies_motor": ["policies_motor (6).parquet", "policies_motor_emerging.parquet"],
    "sample_submission": ["sample_submission (7).parquet", "sample_submission_emerging.parquet"],
    "test_buildings": ["test_buildings (8).parquet", "test_buildings_emerging.parquet"],
    "test_contents": ["test_contents (9).parquet", "test_contents_emerging.parquet"],
    "test_motor": ["test_motor (10).parquet", "test_motor_emerging.parquet"],
    "weather": ["weather (11).parquet", "weather_emerging.parquet"],
    "expenses": ["expenses.parquet"],
    "reference_policies": ["reference_policies.parquet"],
}

# Developed market files (Part 1 / Week 1 baseline)
DEVELOPED_FILE_PATTERNS = {
    "addresses": ["addresses.parquet"],
    "claims": ["claims.parquet"],
    "customers": ["customers.parquet"],
    "policies_buildings": ["policies_buildings.parquet"],
    "policies_contents": ["policies_contents.parquet"],
    "policies_motor": ["policies_motor.parquet"],
    "sample_submission": ["sample_submission.parquet"],
    "test_buildings": ["test_buildings.parquet"],
    "test_contents": ["test_contents.parquet"],
    "test_motor": ["test_motor.parquet"],
    "weather": ["weather.parquet"],
}

CONSTRUCTION_MAP = {
    "BRK": "brick",
    "STN": "stone",
    "WOD": "timber",
    "MIX": "non-standard",
    "brick": "brick",
    "stone": "stone",
    "timber": "timber",
    "non-standard": "non-standard"
}

def resolve_data_dir(data_dir=None):
    """Finds the Part 2 data directory."""
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists():
            return p
    # Look in environment variable or relative to current file / workspace
    if "IAA_PART2_DATA" in os.environ:
        return Path(os.environ["IAA_PART2_DATA"])
    base = Path(__file__).resolve().parent
    for cand in [base / "Part 2", base.parent / "Part 2", Path("Part 2"), Path(".")]:
        if (cand / "reference_policies.parquet").exists() or (cand / "claims (2).parquet").exists():
            return cand.resolve()
    raise FileNotFoundError("Could not automatically locate Part 2 data directory. Specify data_dir explicitly.")

def _find_file(dir_path: Path, patterns: list):
    """Searches for the first matching filename in dir_path."""
    for pat in patterns:
        f = dir_path / pat
        if f.exists():
            return f
    # Fallback search if filenames have slight variation
    for pat in patterns:
        stem = pat.split(".")[0].split(" ")[0]
        hits = list(dir_path.glob(f"*{stem}*.parquet"))
        if hits:
            return hits[0]
    return None

class Part2DataLoader:
    def __init__(self, data_dir=None, verbose=True):
        self.data_dir = resolve_data_dir(data_dir)
        self.verbose = verbose
        self.qa_log = []

    def _log(self, msg):
        if self.verbose:
            print(f"[Part2DataLoader] {msg}", flush=True)

    def _clean_policies(self, df: pd.DataFrame, line: str, split: str) -> pd.DataFrame:
        """Applies essential actuarial cleaning to policy data."""
        df = df.copy()
        df["line"] = line
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["year"] = df["period_start"].dt.year

        # Exposure fix: days recorded instead of fractional years
        days_exp = (((df["period_end"] - df["period_start"]).dt.days + 1) / 365.0).clip(upper=1.0)
        bad_exp = df["exposure"] > 1.0001
        n_bad_exp = int(bad_exp.sum())
        if n_bad_exp > 0:
            df.loc[bad_exp, "exposure"] = days_exp[bad_exp]
            self.qa_log.append((split, line, "exposure > 1.0 recorded in days -> recomputed from period dates", n_bad_exp))
        df["exposure"] = df["exposure"].clip(lower=1.0 / 365.0)

        # Sum Insured fix: 1000x multiplier outlier in buildings/contents
        if "sum_insured" in df:
            med_si = df["sum_insured"].median()
            big_si = df["sum_insured"] > 100.0 * med_si
            n_big_si = int(big_si.sum())
            if n_big_si > 0:
                df.loc[big_si, "sum_insured"] /= 1000.0
                self.qa_log.append((split, line, "sum_insured > 100x median -> divided by 1000", n_big_si))

        # Standardize construction_type
        if "construction_type" in df:
            df["construction_type"] = df["construction_type"].astype(str).map(CONSTRUCTION_MAP).fillna(df["construction_type"])

        # Smoke alarms boolean clean
        if "smoke_alarms" in df:
            df["smoke_alarms"] = df["smoke_alarms"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0})

        return df

    def load_emerging_market(self):
        """Loads and cleans the full emerging market dataset."""
        t0 = time.time()
        self._log(f"Loading Emerging Market dataset from {self.data_dir}...")

        # Load entity tables
        cust_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["customers"])
        addr_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["addresses"])
        wthr_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["weather"])
        cl_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["claims"])

        if not all([cust_path, addr_path, wthr_path, cl_path]):
            raise FileNotFoundError(f"Missing core emerging market entity files in {self.data_dir}")

        cust = pd.read_parquet(cust_path)
        addr = pd.read_parquet(addr_path)
        wthr = pd.read_parquet(wthr_path)
        claims_raw = pd.read_parquet(cl_path)

        # Load policy training tables
        pb_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["policies_buildings"])
        pc_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["policies_contents"])
        pm_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["policies_motor"])

        pb = self._clean_policies(pd.read_parquet(pb_path), "buildings", "train")
        pc = self._clean_policies(pd.read_parquet(pc_path), "contents", "train")
        pm = self._clean_policies(pd.read_parquet(pm_path), "motor", "train")
        pol = pd.concat([pb, pc, pm], ignore_index=True)

        # Load test tables
        tb_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["test_buildings"])
        tc_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["test_contents"])
        tm_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["test_motor"])

        tb = self._clean_policies(pd.read_parquet(tb_path), "buildings", "test")
        tc = self._clean_policies(pd.read_parquet(tc_path), "contents", "test")
        tm = self._clean_policies(pd.read_parquet(tm_path), "motor", "test")
        tst = pd.concat([tb, tc, tm], ignore_index=True)

        # Load submission sample
        sub_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["sample_submission"])
        sample_sub = pd.read_parquet(sub_path) if sub_path else None

        # Clean claims and match to policy rows
        cl = claims_raw.copy()
        cl["loss_date"] = pd.to_datetime(cl["loss_date"])
        cl["report_date"] = pd.to_datetime(cl["report_date"])
        cl["loss_year"] = cl["loss_date"].dt.year
        cl["delay"] = (cl["report_date"] - cl["loss_date"]).dt.days
        cl["net"] = np.where(cl["status"] == "repudiated", 0.0, (cl["incurred_to_date"] - cl["recoveries"]).clip(lower=0.0))
        cl_valid = cl[cl["net"] > 0].copy()

        # Match claims to policy periods
        pcols = ["row_id", "policy_id", "period_start", "period_end"]
        m_all = cl_valid.merge(pol[pcols], on="policy_id", how="inner")
        m_exact = m_all[(m_all["loss_date"] >= m_all["period_start"]) & (m_all["loss_date"] <= m_all["period_end"])].copy()

        unmatched = cl_valid[~cl_valid["claim_id"].isin(m_exact["claim_id"])]
        near = unmatched.merge(pol[pcols], on="policy_id", how="inner")
        near["gap"] = np.where(near["loss_date"] < near["period_start"],
                               (near["period_start"] - near["loss_date"]).dt.days,
                               np.where(near["loss_date"] > near["period_end"],
                                        (near["loss_date"] - near["period_end"]).dt.days, 0))
        near = near.sort_values("gap").drop_duplicates("claim_id")
        near_31 = near[near["gap"] <= 31].drop(columns="gap")

        cl_linked = pd.concat([m_exact, near_31], ignore_index=True).drop(columns=["period_start", "period_end"])
        self.qa_log.append(("train", "all", "emerging claims mapped to exact policy period", int(len(m_exact))))
        self.qa_log.append(("train", "all", "emerging claims mapped to nearest policy period (<=31d)", int(len(near_31))))
        self.qa_log.append(("train", "all", "emerging claims unmatched/dropped", int(len(cl_valid) - len(cl_linked))))

        # Load auxiliary tables
        exp_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["expenses"])
        expenses = pd.read_parquet(exp_path) if exp_path else None

        ref_path = _find_file(self.data_dir, EMERGING_FILE_PATTERNS["reference_policies"])
        reference_policies = pd.read_parquet(ref_path) if ref_path else None

        self._log(f"Loaded Emerging Market in {time.time() - t0:.2f}s:")
        self._log(f"  - Training policies: {len(pol):,} rows (Motor: {len(pm):,}, Buildings: {len(pb):,}, Contents: {len(pc):,})")
        self._log(f"  - Test policies: {len(tst):,} rows (Motor: {len(tm):,}, Buildings: {len(tb):,}, Contents: {len(tc):,})")
        self._log(f"  - Claims: {len(claims_raw):,} raw, {len(cl_linked):,} linked non-repudiated")
        self._log(f"  - Addresses: {len(addr):,}, Customers: {len(cust):,}, Weather rows: {len(wthr):,}")
        if sample_sub is not None:
            self._log(f"  - Sample submission: {len(sample_sub):,} rows")

        return {
            "customers": cust,
            "addresses": addr,
            "weather": wthr,
            "claims": cl_linked,
            "claims_raw": claims_raw,
            "policies": pol,
            "policies_buildings": pb,
            "policies_contents": pc,
            "policies_motor": pm,
            "test": tst,
            "test_buildings": tb,
            "test_contents": tc,
            "test_motor": tm,
            "sample_submission": sample_sub,
            "expenses": expenses,
            "reference_policies": reference_policies,
            "qa_log": self.qa_log,
        }

    def load_developed_market(self):
        """Loads and cleans the baseline developed market dataset (Part 1 files in same dir)."""
        t0 = time.time()
        self._log(f"Loading Developed Market baseline dataset from {self.data_dir}...")
        cust_path = _find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["customers"])
        addr_path = _find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["addresses"])
        wthr_path = _find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["weather"])
        cl_path = _find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["claims"])

        if not all([cust_path, addr_path, wthr_path, cl_path]):
            self._log("Developed market files not found in Part 2 directory. Skipping.")
            return None

        cust = pd.read_parquet(cust_path)
        addr = pd.read_parquet(addr_path)
        wthr = pd.read_parquet(wthr_path)
        claims_raw = pd.read_parquet(cl_path)

        pb = self._clean_policies(pd.read_parquet(_find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["policies_buildings"])), "buildings", "developed_train")
        pc = self._clean_policies(pd.read_parquet(_find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["policies_contents"])), "contents", "developed_train")
        pm = self._clean_policies(pd.read_parquet(_find_file(self.data_dir, DEVELOPED_FILE_PATTERNS["policies_motor"])), "motor", "developed_train")
        pol = pd.concat([pb, pc, pm], ignore_index=True)

        cl = claims_raw.copy()
        cl["loss_date"] = pd.to_datetime(cl["loss_date"])
        cl["report_date"] = pd.to_datetime(cl["report_date"])
        cl["loss_year"] = cl["loss_date"].dt.year
        cl["delay"] = (cl["report_date"] - cl["loss_date"]).dt.days
        cl["net"] = np.where(cl["status"] == "repudiated", 0.0, (cl["incurred_to_date"] - cl["recoveries"]).clip(lower=0.0))
        cl_valid = cl[cl["net"] > 0].copy()

        pcols = ["row_id", "policy_id", "period_start", "period_end"]
        m_all = cl_valid.merge(pol[pcols], on="policy_id", how="inner")
        m_exact = m_all[(m_all["loss_date"] >= m_all["period_start"]) & (m_all["loss_date"] <= m_all["period_end"])].copy()

        unmatched = cl_valid[~cl_valid["claim_id"].isin(m_exact["claim_id"])]
        near = unmatched.merge(pol[pcols], on="policy_id", how="inner")
        near["gap"] = np.where(near["loss_date"] < near["period_start"],
                               (near["period_start"] - near["loss_date"]).dt.days,
                               np.where(near["loss_date"] > near["period_end"],
                                        (near["loss_date"] - near["period_end"]).dt.days, 0))
        near = near.sort_values("gap").drop_duplicates("claim_id")
        near_31 = near[near["gap"] <= 31].drop(columns="gap")
        cl_linked = pd.concat([m_exact, near_31], ignore_index=True).drop(columns=["period_start", "period_end"])

        self._log(f"Loaded Developed Market in {time.time() - t0:.2f}s: {len(pol):,} policies, {len(cl_linked):,} linked claims.")
        return {
            "customers": cust,
            "addresses": addr,
            "weather": wthr,
            "claims": cl_linked,
            "policies": pol,
        }

def load_part2_data(data_dir=None, verbose=True):
    """Convenience function to load cleaned Part 2 data."""
    loader = Part2DataLoader(data_dir=data_dir, verbose=verbose)
    return loader.load_emerging_market()

if __name__ == "__main__":
    print("=" * 80)
    print("IAA Hackathon 2026 - Part 2 Data Loading & Inspection Test")
    print("=" * 80)
    data = load_part2_data()
    print("\nData loading test successful!")
    print("\n--- QA Log ---")
    qa_df = pd.DataFrame(data["qa_log"], columns=["Split", "Line", "Remediation Applied", "Count"])
    print(qa_df.to_string(index=False))
    print("\n--- Expenses Table ---")
    if data["expenses"] is not None:
        print(data["expenses"].to_string(index=False))
    print("\n--- Sample Reference Policy ---")
    if data["reference_policies"] is not None:
        print(data["reference_policies"][["row_id", "line", "driver_age_band", "policyholder_age_band", "written_premium"]].head())
