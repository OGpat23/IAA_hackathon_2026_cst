"""Local check of the peril-weighted large-loss loading: one data preparation, validation at several
strengths of CFG["xs_peril"] (0 = flat loading per line as in v2/v3, 1 = full peril profile).

    IAA_OUT=outputs_week2_v4 python xs_compare.py

Only the *_raw scores can move: the *_c scores are on capped claims and do not see the loading.
"""
import copy
import os
from pathlib import Path

os.environ.setdefault("IAA_OUT", str(Path(__file__).resolve().parent / "outputs_week2_v4"))
import pandas as pd  # noqa: E402

import week2_v2 as W  # noqa: E402

ctx = W.prepare()
rows = {}
for s in (0.0, 0.5, 1.0):
    cfg = copy.deepcopy(W.CFG)
    cfg["xs_peril"] = s
    W.attach_priors(ctx, W.fit_priors(ctx, cfg))
    v = W.validate(ctx, cfg)
    rows[f"xs_peril={s}"] = {k: v[k] for k in ["score", "score_raw", "fwd_raw", "bwd_raw", "cv_raw"]}
    W.log(f"xs_peril={s}: {rows[f'xs_peril={s}']}")
out = pd.DataFrame(rows).T.round(5)
print(out.to_string())
out.to_csv(Path(os.environ["IAA_OUT"]) / "xs_compare.csv")
