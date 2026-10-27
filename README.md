# F1 Race Outcome Prediction

Predicts the finishing order of the next Formula 1 race with three models —
Ridge, Random Forest and XGBoost — trained on 2018–2026 results held in
PostgreSQL.

Built to be read end to end. Every module is under ~300 lines, every tunable
value lives in `config.yaml`, and every claim about accuracy is printed next
to a naive baseline so you can tell whether the model is actually doing
anything.

```bash
python race.py                  # one file, all the knobs at the top
```

or the individual stages:

```bash
python scripts/predict_next.py --mode post_quali --explain
```

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                  # edit DATABASE_URL if needed
createdb f1_outcome_pred
git config core.hooksPath .githooks   # conventional-commit validation
```

Then build the data (first run takes a few minutes; it is cached and
incremental afterwards):

```bash
python scripts/init_db.py          # create schemas and tables
python scripts/ingest.py           # FastF1/Jolpica results + OpenF1 pace
python scripts/build_features.py   # compute core.entry_features
```

---

## The pipeline

```
FastF1 / Jolpica ──┐
                   ├──> raw.*  ──> core.entry_features ──> models ──> ml.*
OpenF1 ────────────┘
   (2023+)
```

| Stage | Module | What it does |
|---|---|---|
| ingest | `f1pred/sources/`, `f1pred/ingest.py` | Season-at-a-time pulls into `raw.*` |
| features | `f1pred/features.py` | Leak-safe rolling stats → `core.entry_features` |
| models | `f1pred/models.py` | Ridge / RF / XGBoost behind one interface |
| evaluate | `f1pred/backtest.py`, `metrics.py`, `baselines.py` | Walk-forward scoring vs baselines |
| tune | `f1pred/tune.py` | Two-stage search → `ml.tuned_params` |
| predict | `f1pred/predict.py` | Next-race order + P(DNF) |

### Two prediction modes

| Mode | Knows the grid? | Use it |
|---|---|---|
| `pre_weekend` | No | Monday, for the coming race |
| `post_quali` | Yes | Saturday evening, once the grid is set |

Grid position is the single strongest predictor, so `post_quali` is far more
accurate — but it is also partly reading the answer. `pre_weekend` is the
honest test of whether the rest of the features carry any signal.

---

## `race.py` — the single-file control surface

Every knob worth turning sits at the top of [race.py](race.py): mode, which
models to fit, the recency half-life, each model's hyperparameters, and a
`GRID_OVERRIDE` for what-if questions ("what if Verstappen starts P10?").
Edit, run, read the backtest underneath the prediction. Nothing it sets
touches `config.yaml`, so experiments cannot corrupt the defaults.

**Why the modelling is not also in that file.** Single-file F1 predictors are
common and they share a failure mode: with no walk-forward split and no
baseline in the output, there is nothing to catch a broken target. One
widely-shared example sets `target = qualifying_time * 1.055` — the label is
a feature times a constant, so the model learns only that multiplier and its
output is just qualifying order wearing a hat. It looks like it works,
because nothing in the script is capable of telling it that it doesn't.

So the knobs live in one editable file and the evaluation stays in modules
that always print the baselines next to the models.

### Beware the inert knob

That same example exposes `rain_probability` and `temperature`, assigned as
one constant to every row. A column with no variance carries no information:
the model cannot learn from it and changing the number cannot change the
predicted order. It is a dial wired to nothing.

This project has no weather knob for that reason. Making one real needs
per-race historical weather in the training set, which FastF1 moved behind
an F1 TV subscription.

---

## Data constraints worth knowing

**FastF1 3.9 moved lap timing behind an F1 TV subscription.** Race results and
qualifying times still come through free via Jolpica, but lap data, weather
and track-status flags do not. Consequences:

- Practice long-run pace is sourced from **OpenF1 instead** (free, no auth,
  but 2023 onwards only — earlier seasons get imputed values).
- Safety-car rate per circuit is not available; **circuit attrition rate**
  stands in for it.
- If you do have an F1 TV subscription, set `F1_USERNAME` / `F1_PASSWORD` in
  `.env` and flip `data.load_practice_pace` in `config.yaml`.

**Rate limits.** Jolpica allows 500 calls/hour, which is why ingestion pulls a
whole season per batch rather than a session per race. OpenF1's free tier is
3 requests/second and 30/minute, handled by a limiter in
`openf1_source.py`. Its data is licensed for personal, non-commercial use.

**2026 is year one of new technical regulations.** The pre-2026 pecking order
does not carry over, and only ~18 races exist under the current rules. This
is why recency weighting is the most important knob in the project.

---

## Tuning

Everything lives in `config.yaml`. The three that matter most:

```yaml
training:
  recency_half_life_races: 40      # weight halves every N races back
  season_weight_multiplier:
    2026: 3.0                      # extra weight on the current regulations
features:
  base: [...]                      # comment a line out to ablate a feature
```

`recency_half_life_races` is the big one. Lower values discount pre-2026 races
harder. Change it, re-run the backtest, and compare:

```bash
python scripts/backtest.py --mode post_quali
```

Or let the search do it, which also grid-searches each model and stores the
winners in `ml.tuned_params` (picked up automatically by `predict_next.py`):

```bash
python scripts/tune.py --mode post_quali
```

The search is two-stage: recency first (scored on Ridge, which fits in
milliseconds), then each model's own hyperparameters at that setting. A full
joint grid would take hours for a worse answer, since the two barely
interact.

**The tuner validates on an earlier block of races than the backtest
reports on** — it skips the most recent `backtest.n_test_races` and selects
on the 20 before them. Choosing hyperparameters on the same races you then
quote a score from is the quiet way to fool yourself, and the gap between
the two windows is what keeps the headline number honest.

### Comparing experiments in SQL

Every run is recorded, so you compare with a query rather than scrollback:

```sql
SELECT r.created_at, r.mode, m.model_name, m.mae, m.spearman,
       r.config_json->'training'->>'recency_half_life_races' AS half_life
FROM ml.backtest_metrics m
JOIN ml.runs r USING (run_id)
WHERE r.kind = 'backtest'
ORDER BY m.mae ASC
LIMIT 20;
```

---

## Measured results

Walk-forward over the **last 30 races** (late 2025 through the 2026 season),
using the tuned settings. These are the real numbers from this repository,
not illustrations.

### `post_quali` — the grid is known

| model | MAE | Spearman | Podium | Winner |
|---|---|---|---|---|
| **rf** | **3.36** | **0.673** | 1.93/3 | 63% |
| ridge *(off by default)* | 3.38 | 0.669 | 1.90/3 | 50% |
| `baseline_grid_order` | 3.41 | 0.647 | 2.03/3 | **70%** |
| xgb | 3.45 | 0.662 | 1.93/3 | 60% |
| `baseline_championship` | 3.75 | 0.615 | 1.77/3 | 40% |

The models beat "everyone finishes where they started" by 0.05 MAE. That is
a win, but a thin one. The clearer gain is in Spearman — the models order the
*whole field* better — while the baseline still calls the **winner** more
often, because pole position converted to victory in 70% of these races.

### `pre_weekend` — no grid

| model | MAE | Spearman | Podium | Winner |
|---|---|---|---|---|
| `baseline_championship` | **3.75** | **0.615** | 1.77/3 | 40% |
| rf | 4.04 | 0.593 | 1.60/3 | 37% |
| ridge *(off by default)* | 4.05 | 0.583 | 1.63/3 | 20% |
| xgb | 4.06 | 0.584 | 1.60/3 | 27% |

**The models lose to championship order here.** Ranking drivers by the points
they have already scored this season is a better pre-weekend predictor than
anything trained on 2018–2026. That is a genuine result, not a bug, and it is
the most useful thing this project has to teach: a well-chosen single feature
can beat a model, and you only find out because the baseline is in the table.

### Why 2026 is hard

2026 is year one of new technical regulations. Only ~18 races exist under the
current cars, so most of the training set describes a pecking order that no
longer applies. The recency search is the lever, and it finds something
awkward: tuning validates on the races *before* the backtest window, which
in this season are mostly pre-regulation-change — so the half-life it picks
is tuned for a world that has since changed. Tuning helps `post_quali`
slightly and currently *hurts* `pre_weekend` (4.05 against 3.94 untuned).

With more 2026 races this should resolve. It is a real limitation of the
current data, not something to tune away.

---

## Reading the metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average positions wrong. Lower is better. |
| **Spearman** | Rank correlation with the real order, −1 to 1. |
| **Podium** | How many of the top 3 were correctly identified, of 3. |
| **Winner** | Share of races where the winner was called correctly. |
| **Top10** | How many points scorers were identified, of 10. |

All are averaged **per race**, so a 22-car grand prix does not outweigh a
shortened one.

### The baselines are the point

`baseline_grid_order` predicts that everyone finishes where they started.
It is a genuinely strong predictor, and a model that cannot beat it is not
contributing anything. Two others are printed alongside it:
`baseline_championship` (current standings order) and `baseline_last_race`.

**Treat a large win over the baselines as a bug, not a triumph.** Beating
grid-order by 0.3–0.7 MAE is a good result. Beating it by 2 means something
has leaked.

### Leakage is the failure mode

The one rule in `features.py`: a feature may only use information that
existed before lights out. Every rolling statistic shifts before it rolls,
and team and circuit aggregates are reduced to one row per race first — so
the sister car in the same event cannot leak in.

Two guards:

1. `check_shift_discipline()` asserts that a driver's debut race has no
   prior-form values, and runs on every feature build.
2. A shuffled-target probe. Scramble the finishing order within each race and
   the models must collapse. Measured here: MAE 7.5–8.4 and Spearman −0.11 to
   −0.31, i.e. *worse* than guessing, which is what fitting pure noise should
   produce. Anything near the real scores would mean leakage.

```bash
python scripts/backtest.py --mode post_quali --shuffle-target
```

---

## Layout

```
race.py              single-file control surface - start here
config.yaml          committed defaults for every tunable knob
db/schema.sql        raw / core / ml schemas, idempotent
f1pred/
  config.py          loads config.yaml + .env
  db.py              engine, upsert, run tracking
  sources/           fastf1_source.py, openf1_source.py
  ingest.py          sources -> raw.*
  features.py        leak-safe feature engineering
  models.py          ridge / rf / xgb + recency weighting
  baselines.py       the bar to beat
  metrics.py         MAE, Spearman, podium, winner, top10
  backtest.py        walk-forward evaluation
  tune.py            two-stage search
  predict.py         next-race prediction
scripts/             one entry point per stage
```

### Database

Three schemas, mirroring the pipeline: `raw` (as fetched), `core` (one
modelling row per driver-race), `ml` (every run, metric and prediction kept
for comparison). `event_id` is `season * 100 + round`, so 2026 round 19 is
`202619`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Commits follow Conventional Commits
and are validated by `.githooks/commit-msg`.
