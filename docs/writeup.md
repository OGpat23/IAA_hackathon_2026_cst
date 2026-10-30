# IAA AI-Assisted Hackathon 2026: pricing across two markets

<p class="byline">Team CST · write-up · 30 October 2026</p>

## Summary

We priced private motor, home buildings and home contents in a developed market with four years of data (Part 1), then carried that pricing into an emerging market with two years of data and a 2025 book that is 76% new business (Part 2). Part 1 finished 7th of 51 teams on the private leaderboard, up from 20th on the public one. Part 2 is decided on its private leaderboard; the public board scores only 33% of the test rows.

| Part | Final model | Local validation (Tweedie D², p = 1.5) | Kaggle |
| --- | --- | --- | --- |
| 1 developed | `kaggle_v8.py`: two LightGBM Tweedie models and a frequency × severity pair per line, on claims developed to ultimate | 0.1380 (2024 holdout) | private 0.13766, 7th of 51 (leader 0.13923) |
| 2 emerging | `week2_v2.py` v7: developed priors re-based by emerging GLMs per peril, calibrated to the 2025 mix | 0.1686 (three schemes) | public 0.16201 |

Both parts build the same target (claims developed to ultimate, trended and capped) and keep a change only if held-out data improves. The emerging model keeps the developed market's relativities and re-estimates only what two years can support: levels, per-peril slopes, gender and affluence terms, the new-business loading and large losses. Claude Code ran the analysis, code and tests; the actuary set the questions, reviewed each result and made every decision.

## Data and data quality

Both markets were cleaned by the same rules and every fix is counted in a data-quality log. Developed: 2.41 million policy rows and 163,150 claims (2021–2024). Emerging: 109,296 rows and 9,258 claims (2023–2024), with 209,063 rows to price for 2025.

| Issue | Treatment | Developed | Emerging |
| --- | --- | --- | --- |
| Exposure recorded in days | recomputed from the period dates | 10,123 rows | 452 rows |
| Sum insured about 1000× too large | divided by 1000 | 3,527 rows | 349 rows |
| Construction codes; free-text occupation | mapped to the developed names; 80 → 40 occupations | — | all rows |
| Repudiated claims | excluded from the target | 1,711 | 120 |
| Claims outside their policy period | re-linked within 31 days, else dropped | 245 / 165 | 2 / 34 |
| `customer_service_contacts` | leak: 3.4 contacts on claim rows, 0.4 on claim-free and test rows | excluded | excluded |
| Marketing fields | noise in a one-way test, except affluence from `marketing_segment` | excluded | affluence kept |

Two facts about 2025 shape the model. A broker portfolio taken on at the start of the year makes 76% of the rows new business (13% in training); they match training new business on every rating field but start in January–February. The weather table also holds 2025 values (a cold, dry year); year weather is worth about +0.002 on buildings in a developed holdout and nothing elsewhere.

## Part 1: the developed-market model

Four years and 163,150 claims support a full rating model, so the effort went into the target and into validation. Part 1 finished 7th of 51 with a private score of 0.13766 (leader 0.13923), up from 20th on the public board. The ranked entry is v7. The repository's `kaggle_v8.py` is v7 plus the five fixes listed in its header: it scores 0.1380 on a 2024 holdout against developed claims (v7 0.1374) and the same 0.13766 privately, but reached Kaggle just after the deadline.

1. **Target: expected ultimate cost.** Claims are valued at 31 December 2024, so recent years understate what 2025 policies will cost. Reported claims are grossed up for late reporting per line, peril and loss year from the reporting delays of the mature years 2021–22; where more than half of a cell is still open, its severity is raised to the fitted trend; total development is capped at 3×. Motor bodily-injury severity falls about 25% at every quantile from 2023 at unchanged maturity, so earlier years are restated to the new basis. Severity is trended to 2025 per peril, credibility-weighted towards the line trend (Z = τ² / (τ² + se²), τ = 3% a year) and bounded to −2% to +10% a year.
2. **Large losses.** Cost is capped at the 99.5th percentile of its line and peril and the excess returns as a loading per line. A claim above 10× its 99.9th percentile is treated as a possible error: capped, given no excess credit and listed for review.
3. **Features.** Customer, vehicle and property fields; weather as each year's level and its anomaly from the address average; prior-year developed claim counts by policy, customer and address; premium and sum insured relative to their line-year median, so the trended target does not count inflation twice; excess ratios; an out-of-fold, credibility-weighted make-model frequency; and interactions with a physical cause (clay soil × dry summers × trees, hail × flat roofs, frost × property age, elevation over distance to a river).
4. **Models.** Per line, two LightGBM Tweedie models (power 1.5 as in the metric, exposure offset; one with 7 leaves of at least 3,000 rows, one deeper) and a Poisson frequency × Gamma severity pair kept within 0.5–2× of them. Trees stop early on held-out policies; the final fit averages three seeds.
5. **Validation and calibration.** Rolling origin, 2021–22 → 2023 and 2021–23 → 2024, with trends fitted on the earlier years only, scored against recorded, suspect-capped and developed claims. Blend weights maximise the level-neutral D² over both folds, so a level error cannot pick the model. A line is recalibrated only when both folds miss in the same direction, by three quarters of the smaller miss and at most ±10%; a 2025 level outside 0.75–1.30× the trended 2024 experience is flagged for review.

| Part 1 version | Main change | Public | Private |
| --- | --- | --- | --- |
| v1 | one regularised LightGBM Tweedie model per line | 0.12521 | 0.13386 |
| v2 | simpler trees; premium and sum insured made inflation-neutral | 0.12484 | 0.13530 |
| v7, ranked entry | two-year validation; frequency × severity blend; revised targets | 0.12811 | 0.13766 |
| v8, `kaggle_v8.py`, after the deadline | end-of-year flag fixed; per-peril trends; bodily-injury break | 0.12791 | 0.13766 |

The 2024 holdout rose from 0.130 (v1) to 0.138 (v8). v2 lost on the public board but gained 0.0014 on the private one, an early sign that a third of the rows is too noisy to steer by; Part 2 therefore decides on local validation. The targets, the validation design and the guardrails carry into Part 2 unchanged; levels, trends, development and the rating factors that differ by market are re-estimated or borrowed there.

## Part 2: adapting to the emerging market

Two years and 9,258 claims (22× fewer than the developed market) cannot support a full rating model, so the emerging model borrows the shape of risk and fits locally only what the data can bear. The schema differs too: ages come as bands, gender is recorded, and there is no credit band, telematics or mileage, so the priors use shared fields only. `week2_v2.py` runs in about two minutes from cached priors.

<figure class="flow"><div class="grid">
<div class="box" style="grid-column:1;grid-row:1"><b>Developed market 2021–24</b>2.41M policy rows, 163k claims, every peril</div>
<div class="arr" style="grid-column:2;grid-row:1">→</div>
<div class="box" style="grid-column:3;grid-row:1"><b>Targets to ultimate</b>IBNR, reserve development, trend to 2025, cap at 99.5%</div>
<div class="arr" style="grid-column:4;grid-row:1">→</div>
<div class="box" style="grid-column:5;grid-row:1"><b>Priors: LightGBM Tweedie</b>per line and per peril, 3 seeds, shared fields only</div>
<div class="box" style="grid-column:7;grid-row:1"><b>Validation and tests</b>forward, backward, 5-fold CV by customer, weighted to the 2025 mix</div>
<div class="down" style="grid-column:5;grid-row:2">↓ log relativities</div>
<div class="down" style="grid-column:7;grid-row:2">↓ fixes every setting</div>
<div class="box" style="grid-column:1;grid-row:3"><b>Emerging market 2023–24</b>109k policy rows, 9.3k claims; gender, age bands</div>
<div class="arr" style="grid-column:2;grid-row:3">→</div>
<div class="box" style="grid-column:3;grid-row:3"><b>Same targets, same rules</b>development borrowed, trend credibility-blended</div>
<div class="arr" style="grid-column:4;grid-row:3">→</div>
<div class="box hl" style="grid-column:5;grid-row:3"><b>Emerging layer: GLMs</b>priors as offsets, free slopes, gender and affluence terms</div>
<div class="arr" style="grid-column:6;grid-row:3">→</div>
<div class="box" style="grid-column:7;grid-row:3"><b>Calibration to 2025</b>new-business factors, large-loss loading by peril mix, line levels</div>
<div class="arr" style="grid-column:8;grid-row:3">→</div>
<div class="box" style="grid-column:9;grid-row:3"><b>2025 predictions</b>209,063 rows in one file: checked, ledgered, submitted</div>
</div><figcaption>Developed data gives the shape of risk, emerging data sets the levels: both markets get the same targets, emerging rows re-fit the developed relativities, and calibration sets the 2025 loadings and levels.</figcaption></figure>

1. **Targets.** Two emerging years show no reporting or reserve pattern of their own, so both come from the developed market. Developed 2024 bodily-injury claims sit 40% below mature years after trend, so 2023 and 2024 bodily injury are developed by 1.35 and 1.40 (buildings accidental damage 2023 by 1.09). Trends blend the single emerging step with the developed trend by credibility: motor 7.0% a year, buildings 6.7%, contents 1.9%. Claims are capped at the 99.5th percentile.
2. **Priors.** LightGBM Tweedie models per line and per peril on shared fields, averaged over three seeds; motor adds the premium relativity, buildings the claims history, contents separate frequency and severity priors.
3. **Emerging layer.** Tweedie GLMs take the priors as offsets. Motor gets free levels and slopes per peril, where costs differ most (bodily injury near half the developed level; theft and glass slopes 1.35 and 1.52); buildings gets levels per peril; both blend the peril sum 50/50 with the line total. Contents re-fits frequency and severity slopes (0.89, 1.16). Gender lowers motor cost for women by 22–34% on bodily injury, own damage and third-party property; affluence stands in for the missing credit band (contents +39% for budget, +28% for mainstream customers). Motor blends in 15% of an emerging-only GBM.
4. **New business.** Credibility-weighted out-of-fold A/E (Z = n/(n+300)) gives factors of 1.05 for motor, 1.02 for buildings and 1.08 for contents, with renewals just under 1; a leaderboard probe found this split right within noise.
5. **Large losses.** The cost above the cap (7.9% of motor, 5.0% of buildings, 2.2% of contents) comes almost entirely from bodily injury, fire and contents flood, so each row's loading follows its predicted peril mix.
6. **Levels.** For Tweedie p = 1.5 the public score is an exact function of a multiplier on any group of rows, so one probe per line gave its public-optimal level (1.08, 1.02, 0.98); credibility pulls these to 1.05, 1.02 and 1.01. Buildings, which swings ±25% between years on a handful of fire claims, moves most of the way.

## Part 2: what each version changed

A change was kept only if local validation improved and the uncapped score fell by no more than 0.001; the leaderboard was the second check.

| Change | Local | Public | Decision |
| --- | --- | --- | --- |
| v2 tuned transfer settings (per-line priors, 50/50 peril re-basing, gender term, 15% emerging GBM) | 0.1635 | 0.1585; 0.1606 at the probed level | kept |
| v3 reserve development of open claims | 0.1676 | 0.1615 (+0.0009 at the same level) | kept |
| v4 large-loss loading by peril mix | 0.1676 (uncapped +0.0006) | 0.1621 | kept |
| v5 credibility-weighted line levels | 0.1676 | 0.1622 | kept |
| v7 affluence term, contents | 0.1686 | 0.1620, within sampling noise | kept, final |
| v6 pooled developed + emerging GBM blended at 20–35% | 0.1690 | 0.1614 | rejected: gain from one scheme and one year |
| discount or channel terms; 2025-mix-weighted fit; weather, region, history terms; residual GBM; larger prior trees | 0.1645–0.1669; trees worse on a developed holdout | not submitted | rejected |

## Validation, guardrails and governance

The checks are built into the code, so a run stops rather than write a doubtful file. In Part 2 no change reached Kaggle without local validation.

- **Validation.** Part 1: rolling origin on 2023 and 2024. Part 2: forward (2023 → 2024), backward (2024 → 2023) and 5-fold cross-validation grouped by customer, all weighted to the 2025 new-business mix. Both select on the Tweedie D² made level-neutral per line; Part 2 also guards the uncapped score.
- **Leakage and shift.** Part 1 compares every numeric policy field in the 2024 training rows with the 2025 test rows, and the claim-free 2024 rows with the test rows (population stability index). A field filled in after a claim makes test rows look claim-free: the run stops if the known leak is not found and drops any other field with that signature for review. Both parts also stop on a train/test gap in policy timing.
- **Pipeline checks.** Part 1 asserts that each claim links to one policy row, merges keep row counts, no target enters the features, peril targets add up and the submission is complete, finite and positive. Part 2 stops on changed row counts, row ids that differ from the test files, or missing or infinite predictions. Both flag a 2025 line level outside a band of trended history (0.75–1.30 in Part 1, 0.80–1.35 in Part 2) for review.
- **Reproducibility.** Each part is one script in the git repository, with its settings in one configuration block and fixed seeds. Part 2 also writes a run manifest (configuration, data hashes, validation scores, levels, coefficients), and its final file reproduces byte for byte.
- **Submission control.** `kaggle_lb.py` validates each Part 2 file against the sample submission, blocks duplicates, asks for confirmation and writes a ledger row linking the public score to the local scores and the run manifest (configuration, data hashes, factors, coefficients).
- **Tests.** 33 pytest checks per Part 2 run, from file format and levels to the Kaggle copy matching the run's configuration.
- **Local against public.** The public board scores 33% of the test rows: in Part 1 we stood 20th on it and finished 7th. In Part 2 a bootstrap of the validation rows puts the sampling spread of a v7-sized difference at about 0.001, with the better model losing in 30% of samples, so v7's −0.0002 could not overturn a gain seen on all three schemes. v6's gain came from one scheme and lost 0.0008 publicly, so it was reverted. Probing beyond line levels was ruled out: it would fit public noise for roughly 0.01 publicly and nothing privately.

## AI-assisted workflow and how its output was verified

Claude Code did the exploration, wrote and ran every script, experiment and test in both parts, and drafted the documentation; the actuary framed each question, read every diagnostic, approved every submission and took the decisions. An AI proposal counted for nothing until a number from held-out data agreed with it.

| Stage | What AI did | How it was checked |
| --- | --- | --- |
| Data quality | found the exposure-in-days rows, the 1000× sums insured, the coded construction types, the free-text occupations, the claims outside their policy period and the leaking contacts field | each finding counted per market in the data-quality log; the leak confirmed on claim, claim-free and test rows |
| Targets | Part 1: development to ultimate, the bodily-injury break, per-peril trends. Part 2: reserve development instead of a flat level multiplier | Part 1: both holdouts, against recorded and developed claims. Part 2: the leaderboard gain at the same level (+0.0009) matched the local gain; the flat ×1.13 of v2 fell to ×1.05 |
| Modelling | built the Part 1 ensemble and the Part 2 transfer pipeline, searched their settings, proposed each later change | every setting fixed on held-out validation; changes accepted there first, then on the leaderboard |
| Diagnostics | A/E by line and segment, train/test shift checks, residual scans over every field including the dropped ones, the exact Tweedie level curve, the public-sample bootstrap | the scans surfaced the end-of-year flag, the discount artefact and the affluence term; the curve predicted v5's public score to five decimals |
| Tooling | 33 tests, the submission ledger, the probe calculator, the Kaggle export | tests run on every run; the ledger reconciled against Kaggle's submission list |

The gains are measured: Part 1 v1 → v8 raised the 2024 holdout from 0.130 to 0.138 and the private score from 0.13386 to 0.13766; Part 2 v1 → v7 raised local validation from 0.1585 to 0.1686 and the public score from 0.15406 to 0.16201. Three AI proposals were rejected by the evidence: a discount term worth +0.003 on validation came from 187 new-business rows with a 10% discount and no buildings claims; the pooled developed + emerging model gained 0.0015 locally and lost 0.0008 publicly; weighting the fit to the 2025 mix lowered the score at every weight. Two AI bugs were caught the same way: an experiment reusing predictions across folds (15 fits logged instead of 21) and the Part 1 end-of-year flag set on every 2024 row (train/test shift check; fixed in v8). Human decision points were fixed in advance: the API token was never read by the assistant, each submission needed explicit approval, and the final choice, the decision to stop and the refusal to probe the public board by group were the actuary's.

## Assumptions, limitations and remaining uncertainties

- **Development (both parts).** Late-reporting and reserve factors rest on judgement and on the two mature developed years; 2024 is immature, which also weakens the 2024 holdout. Emerging bodily-injury factors are borrowed: the 1,056 open emerging claims give no triangle, and undeveloped property reserves may understate 2024 targets by a few percent.
- **Trend and large losses (both).** Trends are bounded and the excess is a loading (flat per line in Part 1, by peril mix in Part 2), so a sharply different 2025 trend or an unusual fire or flood year shows as a level miss.
- **Selection (Part 1).** Blend weights and calibration were chosen on the same two holdouts, and the premium relativity ties the model to current pricing. Neither part produces an uncertainty range per policy.
- **Transfer (Part 2).** Developed relativities, reporting and reserve development hold up to the re-fitted levels, slopes and terms; a stronger market-specific effect would be missed.
- **Levels (Part 2).** Set on the public third of the test rows and shrunk towards the model. Buildings ran at 1.26× the model in 2023 and 0.73× in 2024 on five fire claims; a similar 2025 would move the private score by about ±0.003.
- **New business (Part 2).** Most of 2025 is a bulk transfer, but training holds only 77 buildings and 102 contents new-business claims.

The public score cannot separate models within about 0.001, so the private ranking may reorder teams in that band.

## Office premium for the reference policies

On the emerging expense basis the technical premium of the 20 reference policies averages 1.9–4.0× their written premium by line, so current prices do not cover the technical cost (full table: `outputs_week2_v7/reference_policies_technical_premium.csv`). The risk premium is the model's expected ultimate cost for the row:

<div class="formula">technical premium = <span class="frac"><span class="num">risk premium × (1 + cat reinsurance) + fixed expense × exposure</span><span class="den">1 − variable expense − commission − premium tax − profit margin</span></span></div>

Emerging basis: fixed expense 85 CU per policy, variable expense 15%, commission 20% broker and 0% direct, premium tax 8%, target profit 8%, catastrophe reinsurance 12% of the property risk premium (none on motor); the denominator is 0.49 for a broker policy and 0.69 for a direct one.

| Line | Policies | Mean risk premium (CU) | Mean technical premium (CU) | Mean written premium (CU) | Technical / written |
| --- | --- | --- | --- | --- | --- |
| Motor | 7 | 763 | 1,523 | 682 | 2.3 |
| Buildings | 7 | 417 | 1,050 | 248 | 4.0 |
| Contents | 6 | 52 | 264 | 138 | 1.9 |

Motor and buildings risk premiums alone exceed the written premiums; for contents the 85 CU fixed expense dominates and makes short policies expensive. A 10% change in the buildings level moves its technical premiums by about 8%.
