# REGLAT — AlexandriaX-2026 Subtask 3

Two-stage system for character-level error-span detection and LQM (Linguistically
Motivated Multidimensional Quality Metrics) classification in dialectal-Arabic MT
hypotheses, submitted to AlexandriaX-2026 Subtask 3.

- **Stage 1** — a 4-member BIO-tagging ensemble (CAMeLBERT-DA + MARBERTv2, 2 seeds
  each) detects error spans by projecting each member's token-level probabilities
  onto a shared character grid, then decoding via a tunable threshold, gap-bridging,
  dilation, and optional boundary refinement.
- **Stage 2** — a 4-member classification ensemble (same two encoders/seeds) assigns
  one of six LQM categories to each detected span, conditioned on the marked
  hypothesis, the translation direction, and the English source, with class
  reweighting and a per-class logit bias fitted by coordinate ascent.

Official test result: Overall Score 0.4021 (Overlap F1 0.4272, Error-Class Macro-F1
0.3770, Exact Match F1 0.0041) — rank 4 of 6.

## Repository contents

This repo currently contains a single file:

```
AlexandriaX_Subtask3_v3 (2).ipynb
```

Everything — data loading, the local scorer, Stage 1, Stage 2, nested validation,
and submission-file generation — lives in this one notebook, organized into
numbered sections (1–7) that are meant to be run **top to bottom, in order**.

## Requirements

- **Platform: Google Colab.** The notebook uses `google.colab.files.download(...)`
  to hand you the finished submission zip, and assumes a Colab GPU runtime. It will
  not run unmodified as a plain local script — swap that one cell for a normal
  file write if you need to run it outside Colab.
- **GPU:** developed and timed on a single NVIDIA T4. Expect ~90–120 minutes for
  the full 4-member/2-stage run; if you hit Colab's session limit, reduce
  `ENCODERS` to one model or `SEEDS_PER` to 1 (see Configuration below).
- **Python packages** (installed by the first code cell):
  ```
  transformers>=4.44
  datasets>=2.20
  accelerate>=0.33
  seqeval
  ```
  (`torch`, `numpy`, and `scikit-learn` are assumed pre-installed, as they are on
  the default Colab image.)
- **Data access:** the notebook pulls data directly from the Hugging Face Hub:
  - `UBC-NLP/AlexandriaX_Subtask_3` (train + dev)
  - `UBC-NLP/AlexandriaX_Subtask_3_Test` (test)

  If either dataset is gated, accept the terms on its Hugging Face page and
  uncomment the `huggingface_hub.notebook_login()` line near the top of Section 2
  before running.

## How to run

1. Open the notebook in Google Colab with a GPU runtime (T4 or better).
2. In the **Config** cell (Section 1), set:
   - `TEAM_NAME` — used to name the output submission file.
   - `ENCODERS` — comment one out to train a single-encoder system instead of the
     dual-encoder ensemble (faster, but not the submitted configuration).
   - `SEEDS_PER` — seeds per encoder; total ensemble size is `len(ENCODERS) * SEEDS_PER`
     members per stage. The submitted system used 2 encoders × 2 seeds = 4 members.
3. Runtime → **Run all**. Sections execute in this order:

   | Section | What it does |
   |---|---|
   | 1. Environment | Installs dependencies, checks GPU. |
   | 2. Load data | Pulls train/dev/test from the HF Hub, resolves column names, and builds internal records. |
   | 3. Scorer | A local metric (greedy overlap-pairing; per-category F1 averaged only over categories present in gold/predictions) used **only** for tuning — see caveat below. |
   | 4. Stage 1 | Trains the BIO-tagger ensemble, aggregates probabilities onto a shared character grid, and defines span decoding + boundary refinement. |
   | 5. Stage 2 | Trains the span classifier ensemble on gold training spans. |
   | 6. Nested validation | Two-fold tuning of (threshold, dilation, gap, refinement mode, class bias) to get an honest held-out estimate, then refits the same procedure on all of dev for the final configuration. |
   | 7. Predict + write submission | Runs the validated configuration on the test set, writes `submission.jsonl`, zips it as `test_{TEAM_NAME}_{SUB_NUMBER}.zip`, validates the file, and downloads it. |

4. The final zip is validated in-notebook (line count, required fields, offset
   bounds, span text matching its own offsets, id multiset matching the test file)
   before download — if it prints `PROBLEMS FOUND`, do not submit the file as-is.

## Important caveats (read before trusting any printed number)

- **The notebook's own scorer is a development-time proxy, not the official
  metric.** It pairs predicted/gold spans greedily on any positive overlap and
  averages per-category F1 only over categories that actually occur in that
  batch's gold/predictions. The official CodaBench scorer instead uses
  character-coverage masks for overlap and globally pooled true/false
  positive/negative counts across all categories. **Do not compare the
  notebook's printed `overall`/`overlap_f1`/`class_macro_f1` numbers directly
  against official leaderboard scores** — they are related but not numerically
  equivalent.
- **Dataset IDs are not unique.** The notebook explicitly checks for and works
  around this by indexing everything by row position, never by `id`. If you
  modify the data pipeline, preserve this — merging rows by `id` will corrupt
  offsets.
- **`graphetics` has 2 training examples and 0 dev examples.** Any per-class
  metric for this category on dev is meaningless; on the official (larger,
  balanced) test distribution it may still register near-zero.
- Held-out score differences under ~0.02 are within noise at only 138 dev
  instances (per the notebook's own author note).

## Checkpoints used

- `CAMeL-Lab/bert-base-arabic-camelbert-da`
- `UBC-NLP/MARBERTv2`

## Outputs

- `submission.jsonl` — one JSON object per test row: `id`,
  `LQM_prediction_tagged_errors` (list of `{text, start, end, category}`),
  `direction`, `source`, `model_prediction`.
- `test_{TEAM_NAME}_{SUB_NUMBER}.zip` — the above file zipped, ready for
  CodaBench submission.
