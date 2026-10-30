"""Leaderboard ledger: ties every Kaggle submission to the local validation that produced it.

week2_v2.py scores each run locally (Tweedie D2, power 1.5, on fwd / bwd / 5-fold CV) and writes
run_manifest_v2.json. This script checks a submission file, submits it, waits for the public score
and appends one row to experiments/ledger.csv, so local and leaderboard scores can be compared:

    python kaggle_lb.py status                     # leaderboard position + ledger
    python kaggle_lb.py check FILE                 # format checks only, nothing is sent
    python kaggle_lb.py submit FILE -m "message"   # checks, confirms, submits, waits, records
    python kaggle_lb.py sync                       # pull public scores into the ledger
    python kaggle_lb.py compare                    # local vs public score, rank agreement

The local `score` is level-neutral (claims rescaled per line x fold), so it measures segmentation
only. The leaderboard also rewards the overall level, which is what level_probe.py calibrates;
compare local and public scores between submissions at the same level (same mean prediction).
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COMP = "iaa-ai-assisted-hackathon-2026-part-2"
ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "emerging" / "sample_submission.parquet"
LEDGER = ROOT / "experiments" / "ledger.csv"
DEFAULT_MANIFEST = ROOT / "outputs_week2_v2" / "run_manifest_v2.json"
LINES = ["motor", "buildings", "contents"]
COLS = ["time_utc", "file", "md5", "rows", "mean_pred", "message", "script", "config_md5",
        "local_score", "local_cv_c", "local_fwd_c", "local_bwd_c", "local_score_raw",
        "level_motor", "level_buildings", "level_contents", "kaggle_ref", "status", "public_score"]


def kaggle_bin():
    for c in (shutil.which("kaggle"), Path.home() / "Library/Python/3.13/bin/kaggle"):
        if c and Path(c).exists():
            return str(c)
    sys.exit("kaggle CLI not found - pip3 install --user kaggle")


def kaggle(*args):
    r = subprocess.run([kaggle_bin(), *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}{r.stderr}")
    return r.stdout


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def remote_submissions():
    out = kaggle("competitions", "submissions", "-c", COMP, "--csv")
    return pd.read_csv(io.StringIO(out[out.index("ref,"):]), dtype={"ref": str})


def read_ledger():
    if LEDGER.exists():
        return pd.read_csv(LEDGER, dtype={"kaggle_ref": str, "md5": str})
    return pd.DataFrame(columns=COLS)


def append(led, row):
    new = pd.DataFrame([row], columns=COLS)
    return new if led.empty else pd.concat([led, new], ignore_index=True)


def write_ledger(df):
    LEDGER.parent.mkdir(exist_ok=True)
    df[COLS].to_csv(LEDGER, index=False, quoting=csv.QUOTE_MINIMAL)


# =====================================================================================
# checks run before anything leaves the machine
# =====================================================================================
def check_file(path):
    sub = pd.read_csv(path)
    sample = pd.read_parquet(SAMPLE)
    problems = []
    if list(sub.columns) != ["row_id", "prediction"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['row_id', 'prediction']")
    else:
        if not sub.row_id.is_unique:
            problems.append(f"{sub.row_id.duplicated().sum()} duplicated row_id")
        missing, extra = set(sample.row_id) - set(sub.row_id), set(sub.row_id) - set(sample.row_id)
        if missing or extra:
            problems.append(f"{len(missing)} row_id missing, {len(extra)} unexpected vs sample_submission")
        p = sub.prediction
        if p.isna().any() or not np.isfinite(p).all():
            problems.append(f"{(~np.isfinite(p)).sum()} non-finite predictions")
        if (p < 0).any():
            problems.append(f"{(p < 0).sum()} negative predictions")
        if p.max() > 100 * p.median():
            problems.append(f"max {p.max():.0f} is over 100x the median {p.median():.1f}")
    by_line = {}
    if "prediction" in sub:
        pre = sub.row_id.str[0].map({"M": "motor", "B": "buildings", "C": "contents"})
        by_line = sub.groupby(pre).prediction.mean().round(2).to_dict()
    return sub, problems, by_line


def manifest_fields(path):
    if not path or not Path(path).exists():
        return {}
    m = json.loads(Path(path).read_text())
    v, lv = m.get("validation", {}), m.get("levels", {})
    return {"script": m.get("script", ""),
            "config_md5": hashlib.md5(json.dumps(m.get("config", {}), sort_keys=True).encode()).hexdigest()[:10],
            "local_score": v.get("score"), "local_cv_c": v.get("cv_c"), "local_fwd_c": v.get("fwd_c"),
            "local_bwd_c": v.get("bwd_c"), "local_score_raw": v.get("score_raw"),
            **{f"level_{l}": lv.get(l, {}).get("ratio") for l in LINES}}


# =====================================================================================
# commands
# =====================================================================================
def cmd_check(a):
    sub, problems, by_line = check_file(a.file)
    print(f"{a.file}: {len(sub):,} rows, mean {sub.prediction.mean():.2f}, by line {by_line}")
    print("\n".join(f"  FAIL {p}" for p in problems) or "  all checks passed")
    return 1 if problems else 0


def cmd_submit(a):
    sub, problems, by_line = check_file(a.file)
    if problems:
        print("\n".join(f"FAIL {p}" for p in problems))
        sys.exit("not submitted - fix the file first")
    h = md5(a.file)
    led = read_ledger()
    if h in set(led.md5.dropna()) and not a.force:
        prev = led[led.md5 == h].iloc[-1]
        sys.exit(f"identical file already submitted ({prev.time_utc}, public {prev.public_score}); --force to resend")

    rs = remote_submissions()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_today = int(rs.date.astype(str).str.startswith(today).sum())
    mf = manifest_fields(None if a.no_manifest else a.manifest)
    tag = f"[{h[:8]}]"
    msg = f"{a.message} {tag}"
    print(f"file      {a.file}\nrows      {len(sub):,}  mean {sub.prediction.mean():.2f}  by line {by_line}")
    print(f"message   {msg}")
    print(f"local     score {mf.get('local_score')}  (manifest: {'none' if not mf else a.manifest})")
    print(f"today     {n_today} submission(s) already made (UTC day)")
    if not a.yes and input("submit to Kaggle? [y/N] ").strip().lower() != "y":
        sys.exit("cancelled")

    kaggle("competitions", "submit", "-c", COMP, "-f", str(a.file), "-m", msg,
           "--wait", str(a.wait), "--poll-interval", "15")
    rs = remote_submissions()
    hit = rs[rs.description.fillna("").str.contains(tag, regex=False)]
    row = {c: "" for c in COLS}
    row.update(mf)
    row.update(time_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), file=str(a.file), md5=h,
               rows=len(sub), mean_pred=round(sub.prediction.mean(), 4), message=a.message)
    if len(hit):
        r = hit.iloc[0]
        row.update(kaggle_ref=r.ref, status=str(r.status).split(".")[-1], public_score=r.publicScore)
    write_ledger(append(led, row))
    print(f"\nrecorded: ref {row['kaggle_ref']}  status {row['status']}  public {row['public_score']}")
    if row["status"] != "COMPLETE":
        print("not scored yet - run `python kaggle_lb.py sync` in a minute")


def cmd_sync(a):
    led, rs = read_ledger(), remote_submissions()
    known = set(led.kaggle_ref.dropna())
    for i, r in led.iterrows():
        m = rs[rs.ref == r.kaggle_ref] if r.kaggle_ref else rs.iloc[0:0]
        if not len(m) and isinstance(r.md5, str):         # submitted but ref not captured yet
            m = rs[rs.description.fillna("").str.contains(f"[{r.md5[:8]}]", regex=False)]
        if len(m):
            led.loc[i, ["kaggle_ref", "status", "public_score"]] = [
                m.iloc[0].ref, str(m.iloc[0].status).split(".")[-1], m.iloc[0].publicScore]
    new = rs[~rs.ref.isin(known | set(led.kaggle_ref.dropna()))]
    for _, r in new.iterrows():                           # made outside this script (website, notebook)
        row = {c: "" for c in COLS}
        row.update(time_utc=str(r.date)[:19], file=r.fileName, message=r.description if isinstance(
            r.description, str) else "(submitted outside kaggle_lb.py)", kaggle_ref=r.ref,
                   status=str(r.status).split(".")[-1], public_score=r.publicScore)
        led = append(led, row)
    write_ledger(led.sort_values("time_utc"))
    print(f"ledger synced: {len(led)} rows ({len(new)} new from Kaggle)")


def cmd_status(a):
    cmd_sync(a)
    lb = pd.read_csv(io.StringIO(_lb_csv()))
    best = pd.to_numeric(read_ledger().public_score, errors="coerce").max()
    if np.isfinite(best):
        rank = int((lb.score > best).sum()) + 1
        print(f"best public {best:.5f}  ->  rank ~{rank} of {len(lb)} shown  |  leader {lb.score.max():.5f}"
              f"  |  top-10 cut {lb.score.nlargest(10).min():.5f}")
    cmd_compare(a)


def _lb_csv():
    out = kaggle("competitions", "leaderboard", "-c", COMP, "--show", "--csv", "--page-size", "200")
    return out[out.index("teamId"):]


def cmd_compare(a):
    led = read_ledger()
    show = ["time_utc", "message", "mean_pred", "local_score", "local_score_raw", "public_score", "status"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(led[show].to_string(index=False))
    d = led.assign(ls=pd.to_numeric(led.local_score, errors="coerce"),
                   ps=pd.to_numeric(led.public_score, errors="coerce")).dropna(subset=["ls", "ps"])
    # level probes rescale one run, so they share its local score: keep each model's best level only
    d = d.sort_values("ps").drop_duplicates(["config_md5", "ls"], keep="last")
    if len(d) >= 3:
        print(f"\nlocal vs public over {len(d)} distinct models: pearson {d.ls.corr(d.ps):.2f}, "
              f"spearman {d.ls.corr(d.ps, method='spearman'):.2f}")
    else:
        print(f"\n{len(d)} distinct model(s) with both scores - need 3+ to judge whether local tracks public")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    sp.add_parser("sync")
    sp.add_parser("compare")
    c = sp.add_parser("check")
    c.add_argument("file", type=Path)
    s = sp.add_parser("submit")
    s.add_argument("file", type=Path)
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help="run manifest holding the local validation for this file")
    s.add_argument("--no-manifest", action="store_true", help="file has no matching local validation")
    s.add_argument("--wait", type=int, default=600, help="seconds to wait for scoring")
    s.add_argument("--force", action="store_true", help="resubmit a file already in the ledger")
    s.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()
    sys.exit({"status": cmd_status, "sync": cmd_sync, "compare": cmd_compare, "check": cmd_check,
              "submit": cmd_submit}[a.cmd](a))


if __name__ == "__main__":
    main()
