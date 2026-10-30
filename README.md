# IAA AI-Assisted Hackathon 2026 - team CST

Predicts the expected ultimate claim cost of every 2025 policy row for three lines: private motor, home
buildings and home contents. Part 1 is a developed market with four years of history. Part 2 is an emerging
market with two years, where most of the 2025 book is a broker portfolio taken on at the start of the year.
Submissions are scored with the Tweedie D² (power 1.5).

## Results (Part 2, public leaderboard)

| Version | What changed | Local validation | Public score |
|---|---|---|---|
| v1 | `week2_emerging.py`: emerging-only GBM blended with a developed-prior transfer model | 0.1585 | 0.15406 |
| v2 | `week2_v2.py`: tuned transfer pipeline; overall level x1.134 from probes | 0.1635 | 0.15852 at x1.00, 0.16058 at x1.134 |
| v3 | development of open claims (IBNER), mainly motor bodily injury | 0.1676 | 0.16177 at x1.06 |
| v4 | large-loss loading follows each row's peril mix | 0.1676 (uncapped 0.1346) | 0.16210 |
| v5 | per-line levels from leaderboard probes, credibility-weighted | 0.1676 | 0.16223 |
| v6 | pooled developed + emerging model blended in (motor 20%, buildings 35%); 10 prior seeds | 0.1690 (uncapped 0.1361) | 0.16143, rejected |
| v7 | affluence (from `marketing_segment`) as a contents layer term; motor and buildings unchanged (**final**) | **0.1686** | 0.16201 |

The public leaderboard is 33% of the test rows. For a change of v7's size, bootstrapping the validation rows shows the
public score difference has a sampling spread of about 0.001, and 30% of such samples would show v7 below v5 even
with its local gain of +0.001. The public score therefore cannot separate v7 from v5; local validation, which
prefers v7 on all three schemes, decides. A probe of the new-business rows (x1.10, public 0.16229) implied a
new-business / renewal split 20% higher than the model's, but that is exactly one standard deviation of the
sampling noise of that split, so the model's split is kept.

v6 is the one change where local validation and the leaderboard clearly disagreed. Most of its local gain came from the
backward scheme (buildings in the heavy 2023 year), and the flexible pooled model is more exposed to the 2025
shift (a broker book starting in January-February) than the GLM layer. It stays in the code, switched off.

Local validation is the Tweedie D² on capped claims, level-neutral per line, averaged over three schemes:
forward (2023 -> 2024), backward (2024 -> 2023) and 5-fold CV grouped by customer, all weighted to the 2025
new-business mix. Every submission, with its file hash and local scores, is in `experiments/ledger.csv`.

## Results (Part 1)

7th of 51 on the private leaderboard with 0.13766 (leader 0.13923), up from 20th on the public board. The ranked
entry is v7 (public 0.12811). `kaggle_v8.py` is v7 plus the five fixes listed in its header: 0.1380 on a 2024 holdout
against developed claims (v7 0.1374), the same private 0.13766, but submitted just after the deadline. The script is
annotated section by section; every setting is a constant in its configuration block.

## Repository

| Path | Purpose |
|---|---|
| `week2_v2.py` | Part 2 pipeline (v2 to v7; defaults = final v7). Settings are in `CFG`; the header lists every change and every rejected test |
| `week2_v2_kaggle.py`, `week2_v2_kaggle.ipynb` | Kaggle copies with the final configuration embedded, written by `IAA_MODE=export` |
| `week2_emerging.py` | Part 2 v1 |
| `kaggle_v8.py` | Part 1 model, annotated: targets developed to ultimate, LightGBM Tweedie + frequency × severity blend, guardrails |
| `load_part2.py`, `kaggle_part2.py` | first Part 2 loader and pipeline, kept for history |
| `kaggle_lb.py` | leaderboard ledger: checks a file, submits it, records local and public scores |
| `level_probe.py` | level calibration from probe submissions with the exact Tweedie curve |
| `xs_compare.py` | local check of the peril-weighted large-loss loading |
| `tests/` | pytest suite: submission format and values, levels, guardrails, Kaggle copy in sync with the run |
| `experiments/ledger.csv` | every leaderboard submission |
| `docs/IAA_2026_CST_writeup.pdf` | the write-up (5 pages, both parts); rebuilt from `docs/writeup.md` by `python3 docs/build_writeup_pdf.py` |
| `docs/video_script.md` | narration for the 5-minute video, slide by slide |
| `outputs_week2*/` | per run: manifest (configuration, data hashes, validation, levels, coefficients), QA log, run and tuning logs |

## Setup

Python 3.13 with pandas 2.2, numpy 2.2, lightgbm 4.6, scikit-learn 1.7, scipy 1.16, pyarrow and pytest. The
Kaggle CLI (2.2) is only needed to submit.

The competition data is not in the repository. Download it from Kaggle into `data/emerging/` and
`data/developping/` (`developed/` and `developing/` are also accepted), or set `IAA_DATA` to the folder that
holds both.

## Run

```bash
IAA_OUT=outputs_week2_v7 python3 week2_v2.py                  # final submission (v7)
IAA_OUT=outputs_week2_v7 python3 -m pytest tests -q             # tests against that run
IAA_OUT=outputs_week2_v7 IAA_MODE=export python3 week2_v2.py    # refresh the Kaggle script and notebook
IAA_SMOKE=1 IAA_OUT=outputs_smoke python3 week2_v2.py           # fast dry run on a sample
IAA_DATA=data/developping IAA_OUT=outputs_part1 python3 kaggle_v8.py   # Part 1 (writes submission.csv)
```

A run writes `submission_week2_v2.csv`, `run_manifest_v2.json`, `qa_log.csv` and
`reference_policies_technical_premium.csv` to `IAA_OUT`. The first run trains the developed-market models and
caches them in `IAA_OUT/cache` (about 10 minutes on a laptop); later runs take about a minute.

Other submissions re-run from the configuration recorded in their manifest:

```bash
IAA_CONFIG=outputs_week2_v5/run_manifest_v2.json IAA_OUT=rerun_v5 python3 week2_v2.py
IAA_CONFIG=outputs_week2_v6/run_manifest_v2.json IAA_OUT=rerun_v6 python3 week2_v2.py
IAA_CONFIG=outputs_week2_v4/run_manifest_v2.json IAA_LEVEL_SCALE=1.06 IAA_OUT=rerun_v4 python3 week2_v2.py
IAA_CONFIG=outputs_week2_v3/run_manifest_v2.json IAA_LEVEL_SCALE=1.06 IAA_OUT=rerun_v3 python3 week2_v2.py
IAA_CONFIG=outputs_week2_v2/run_manifest_v2.json IAA_SEV_DEV=0 IAA_LEVEL_SCALE=1.134 IAA_OUT=rerun_v2 python3 week2_v2.py
```

Leaderboard work:

```bash
python3 kaggle_lb.py status                                     # rank, leader, ledger
python3 kaggle_lb.py check FILE                                 # format checks only
python3 kaggle_lb.py submit FILE -m "message" --manifest outputs_week2_v7/run_manifest_v2.json
python3 level_probe.py --help                                   # overall and per-line level calibration
```

## Method (Part 2)

1. **Cleaning.** Both markets go through the same rules: exposure recorded in days, sum insured 1000x too
   large, construction codes, free-text occupation, and claims re-linked to the nearest policy row within 31
   days. Every fix is counted in `qa_log.csv`.
2. **Targets.** Ultimate cost per row, net of excess and recoveries, excluding repudiated claims. Claim counts
   are grossed up for late reporting (IBNR) with developed-market delays. Open claims in the immature 2023-24
   loss years are developed where the shortfall is significant (IBNER: motor bodily injury x1.35 and x1.40).
   Severity is trended to 2025, with the emerging trend credibility-weighted towards the developed one. Large
   losses are capped at the 99.5th percentile and added back as a loading.
3. **Developed-market priors.** LightGBM Tweedie models per line and per peril, trained on the features both
   markets share (frequency and severity models for contents), averaged over 3 seeds.
4. **Emerging layer.** Tweedie GLMs on the emerging data re-base the priors: per peril and per line for motor
   and buildings, frequency and severity slopes for contents, plus a gender term for motor and an affluence term
   (budget / mainstream / affluent, from the marketing segment) for contents, where it stands in for the credit
   band the emerging market does not record. Motor also blends in 15% of an emerging-only GBM. A test requires
   the GLM layer to keep at least half of each line's weight.
5. **New business.** 76% of the 2025 book is new business against 13% in training. New business and renewals
   get credibility-weighted A/E factors from out-of-fold predictions.
6. **Large losses.** The loading per line is spread by each row's predicted peril mix, because the cost above
   the cap comes almost entirely from motor bodily injury, buildings fire and contents flood.
7. **Levels.** The overall and per-line levels come from leaderboard probes, credibility-weighted towards the
   model (`level_probe.py`).

## Validation and guardrails

- A change is kept only if the level-neutral validation score improves and the uncapped score does not fall by
  more than 0.001. Rejected changes are listed in the header of `week2_v2.py`.
- Every submission goes through `kaggle_lb.py`. It checks the format and values against the sample submission,
  blocks resubmitting an identical file, asks for confirmation, and writes a ledger row that links the public
  score to the local validation and the run manifest.
- Run manifests record the configuration, data file hashes, validation scores, levels and coefficients. The
  model cache is keyed on data, configuration and code.
- Assertions stop a run on changed row counts, row ids that do not match the test files, non-finite
  predictions, or a train/test difference in policy timing. Levels outside 0.80-1.35 of history are flagged
  for review.
- The tests in `tests/` check the submission, the levels, the guardrails and that the Kaggle copy carries the
  configuration of the run.

## Limitations

- Only two emerging years. Line levels move a lot between years (buildings came in at 1.26x the model's level
  in 2023 and 0.73x in 2024), so the 2025 level is uncertain, and the leaderboard calibration saw only the
  public rows.
- Most of 2025 is new business, but training holds few new-business claims (77 for buildings), so effects
  specific to new business rest on the developed market.
- 2025 weather is unknown; weather enters as long-run averages per address.
