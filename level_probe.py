"""Leaderboard level calibration with the exact Tweedie(1.5) curve.

For power 1.5 the unit deviance is -8 sqrt(y) + 4 y/sqrt(m) + 4 sqrt(m). Multiplying the predictions of a group
of rows G by k therefore changes the public score by exactly
    dD2 = -b_G (k^-1/2 - 1) - c_G (k^1/2 - 1),   b_G = 4 sum_G y/sqrt(m) / N,   c_G = 4 sum_G sqrt(m) / N
(N = null deviance of the public rows), and the group's best multiplier is k* = b_G / c_G.

  overall  three or more multipliers of one file (the v2 probes x0.92 .. x1.18 fit this curve within 0.00003):
               python level_probe.py overall outputs_week2_v2/submission_week2_v2.csv \
                   0.92:0.15471 1.00:0.15852 1.08:0.16031 1.134:0.16058 1.18:0.16023
           fits D2(k) = a - b k^-1/2 - c k^1/2, prints the best multiplier and c (reused by `lines`).

  lines    one probe per line on a base file (that line x K, the other lines unchanged):
               python level_probe.py lines outputs_week2_v4/submission_v4_x1.06.csv 0.16210 --base-level 1.06 \
                   --ref outputs_week2_v2/submission_week2_v2.csv --c 0.5393 M:1.10:0.16168 B:1.10:0.16148 C:1.10:0.16156
           c for a line = c(ref) x its sum sqrt(m) in the base file / sum sqrt(m) of the ref file (the public rows are
           a random sample, so the public share cancels). Prints each line's optimum and the credibility-weighted
           levels for CFG["level"]: log level = common 2025 factor + line factor with normal priors (TAU_COMMON,
           TAU_LINE) and public-sample noise SIGMA, so a line moves to its public optimum only as far as that
           optimum is more signal than noise.

The levels are a calibration to 2025 experience on the public rows: report them as such in the write-up.
"""
import argparse

import numpy as np
import pandas as pd

LINES = {"M": "motor", "B": "buildings", "C": "contents"}
TAU_COMMON = 0.08                              # prior sd of a 2025 shift common to all lines (vs the model)
TAU_LINE = {"M": 0.05, "B": 0.30, "C": 0.05}   # buildings swung x1.26 / x0.73 between 2023 and 2024, others ~2%
SIGMA = {"M": 0.061, "B": 0.129, "C": 0.074}   # sd of a line optimum from public-sample noise alone (bootstrap
                                               # of v4 validation predictions, 30% public share)


def root_sum(path, prefix=None):
    s = pd.read_csv(path)
    if prefix:
        s = s[s.row_id.str[0] == prefix]
    return float(np.sqrt(s.prediction).sum())


def cmd_overall(a):
    k, s = (np.array(v) for v in zip(*[map(float, p.split(":")) for p in a.probes]))
    if len(k) < 3:
        raise SystemExit("need three or more multiplier:score pairs")
    X = np.c_[np.ones_like(k), -k ** -0.5, -k ** 0.5]
    coef = np.linalg.lstsq(X, s, rcond=None)[0]
    a0, b, c = coef
    for ki, si, fi in zip(k, s, X @ coef):
        print(f"  x{ki:<6} public {si:.5f}  curve {fi:.5f}")
    best = b / c
    print(f"best multiplier x{best:.4f} (public {a0 - b * best ** -0.5 - c * best ** 0.5:.5f}) | c = {c:.4f}, "
          f"sum sqrt(m) of the file = {root_sum(a.file):,.0f}")


def cmd_lines(a):
    b_ref = root_sum(a.ref)
    rows = []
    for p in a.probes:
        g, k, sc = p.split(":")
        k, sc = float(k), float(sc)
        c = a.c * root_sum(a.file, g) / b_ref
        b = (sc - a.base_score + c * (k ** 0.5 - 1)) / (1 - k ** -0.5)
        rows.append((g, c, b, b / c))
    T = pd.DataFrame(rows, columns=["line", "c", "b", "opt_vs_base"]).set_index("line")
    T["opt_vs_model"] = T.opt_vs_base * a.base_level
    tau = np.array([TAU_LINE[g] for g in T.index])
    sig = np.array([SIGMA[g] for g in T.index])
    S = TAU_COMMON ** 2 + np.diag(tau ** 2 + sig ** 2)          # covariance of the public log optima
    C = TAU_COMMON ** 2 + np.diag(tau ** 2)                     # covariance of true levels with them
    T["credibility"] = np.exp(C @ np.linalg.solve(S, np.log(T.opt_vs_model.values)))
    T["used"] = T.credibility.round(2)
    kk = T.used / a.base_level
    T["public_gain"] = -T.b * (kk ** -0.5 - 1) - T.c * (kk ** 0.5 - 1)
    T.index = T.index.map(LINES)
    print(T.round(5).to_string())
    if len(T) == len(LINES):
        print(f"common optimum vs the model x{T.b.sum() / T.c.sum() * a.base_level:.4f}")
    print(f"predicted public score with the 'used' levels: {a.base_score + T.public_gain.sum():.5f}")
    print('CFG["level"] =', {l: float(v) for l, v in T.used.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("overall")
    o.add_argument("file", help="the x1.00 file the multipliers were applied to")
    o.add_argument("probes", nargs="+", help="multiplier:public_score")
    li = sp.add_parser("lines")
    li.add_argument("file", help="base file the line probes were made from")
    li.add_argument("base_score", type=float, help="public score of the base file")
    li.add_argument("probes", nargs="+", help="row_id prefix (M / B / C):multiplier:public_score")
    li.add_argument("--base-level", type=float, default=1.0, help="multiplier of the base file vs the model")
    li.add_argument("--ref", required=True, help="file `overall` was fitted on")
    li.add_argument("--c", type=float, required=True, help="c printed by `overall`")
    a = ap.parse_args()
    {"overall": cmd_overall, "lines": cmd_lines}[a.cmd](a)


if __name__ == "__main__":
    main()
