# Twitter Sentiment & Topic Project - Results Report (label mode: `four`)

_Generated automatically from files in `results/`; every number below was produced by the pipeline run that created them._

## 1. Data and evaluation protocol
- Training file rows read: **74,682**; validation file rows read: **1,000** (header detected: False).
- Unmapped-label rows (training/validation): 0 / 0 (dropped, never mapped to Neutral).
- Training cleaning: 2643 rows dropped for conflicting labels, 4523 duplicate texts removed, 704 rows removed for overlap with the official validation file.
- Label set: Negative, Neutral, Positive, Irrelevant.
- Protocol: official training file -> stratified 85/15 train/val (model selection); official validation file = held-out test (never used for selection).
- Rows - train 56,060, val 9,894, test 1,000.

## 2. Model comparison
Selection criterion: highest **validation** macro-F1. Test columns are the single held-out evaluation (official validation file).

| model         |   val_macro_f1 |   test_accuracy |   test_macro_f1 |   test_weighted_f1 |   train_seconds |   inference_ms_per_1k |
|:--------------|---------------:|----------------:|----------------:|-------------------:|----------------:|----------------------:|
| linear_svm    |         0.9521 |           0.98  |          0.9795 |             0.98   |             1.6 |                 155.3 |
| logreg        |         0.9429 |           0.981 |          0.98   |             0.981  |            31.1 |                 153.4 |
| naive_bayes   |         0.9411 |           0.966 |          0.9651 |             0.966  |             0.1 |                 154.7 |
| random_forest |         0.9103 |           0.947 |          0.9462 |             0.947  |            62.5 |                 238.2 |
| distilbert    |         0.891  |           0.933 |          0.9318 |             0.9331 |          7632.1 |               13617.5 |

**Selected model:** `linear_svm` (val macro-F1 0.9521; test macro-F1 0.9795).

## 2b. Near-duplicate / leakage diagnostic
- 39.9% of held-out test tweets have a training tweet with cosine similarity >= 0.9 (validation sample: 15.4%).
- `linear_svm` accuracy on those near-duplicates: 0.9924812030075187; on novel tweets: 0.9717138103161398.
- A large gap means headline scores partly measure memorisation of repeated tweets; report the *novel* figure as the conservative estimate.

## 2c. Verified leaderboard (saved models re-evaluated from disk)
val_* = internal validation split (selection); test_* = official validation file (held out). CI = bootstrap 95% over test tweets; inference time is end-to-end from raw text on the listed device.

| model         |   val_macro_f1 |   test_accuracy |   test_macro_f1 |   test_macro_f1_ci95_low |   test_macro_f1_ci95_high |   test_weighted_f1 |   train_seconds |   inference_ms_per_1k_tweets |   model_size_mb | device_for_inference         | all_checks_passed   |
|:--------------|---------------:|----------------:|----------------:|-------------------------:|--------------------------:|-------------------:|----------------:|-----------------------------:|----------------:|:-----------------------------|:--------------------|
| linear_svm    |         0.9521 |           0.98  |          0.9795 |                   0.9702 |                    0.9878 |             0.98   |             1.6 |                        168.6 |            6.21 | cpu                          | True                |
| logreg        |         0.9429 |           0.981 |          0.98   |                   0.9708 |                    0.9885 |             0.981  |            31.1 |                        168   |            6.44 | cpu                          | True                |
| naive_bayes   |         0.9411 |           0.966 |          0.9651 |                   0.9536 |                    0.9756 |             0.966  |             0.1 |                        168.9 |            8.82 | cpu                          | True                |
| random_forest |         0.9103 |           0.947 |          0.9462 |                   0.9316 |                    0.9596 |             0.947  |            62.5 |                        239.1 |          195.05 | cpu                          | True                |
| distilbert    |         0.8908 |           0.934 |          0.9329 |                   0.9154 |                    0.9481 |             0.9341 |          7632.1 |                       4962.2 |          268.56 | cuda:NVIDIA GeForce GTX 1650 | False               |

Test accuracy by maximum TF-IDF cosine similarity to any training tweet:

| model         |   0.5-0.7 |   0.7-0.9 |   <0.5 (novel) |   >=0.9 (near-duplicate) |
|:--------------|----------:|----------:|---------------:|-------------------------:|
| distilbert    |    0.8545 |    0.9526 |         0.5946 |                   0.9527 |
| linear_svm    |    0.9455 |    0.9941 |         0.7027 |                   0.9925 |
| logreg        |    0.9636 |    0.9862 |         0.7568 |                   0.9975 |
| naive_bayes   |    0.9273 |    0.9802 |         0.6216 |                   0.9851 |
| random_forest |    0.7818 |    0.9625 |         0.7297 |                   0.9701 |

## 3. Preprocessing alternatives (LogReg, validation split)
| preprocessing                                    |   val_accuracy |   val_macro_f1 |   n_features |   seconds |
|:-------------------------------------------------|---------------:|---------------:|-------------:|----------:|
| light (no contraction/repeat handling, keep #/@) |         0.9335 |         0.9322 |       155256 |      38.2 |
| default (negation-preserving)                    |         0.9262 |         0.9245 |       150896 |      47.9 |
| default + stopword removal                       |         0.9227 |         0.9213 |       120336 |      44.2 |
| default + stemming                               |         0.9239 |         0.9222 |       141099 |      56.5 |
| default + POS-aware lemmatisation                |         0.9247 |         0.9228 |       138533 |      85.5 |

## 4. Topic discovery (fit on training text only)
### LDA - k = 12
_max mean UMass coherence over the grid (override: TSP_LDA_K)_

- **lda 0** (6,731 tweets, 25% negative): amazon, com, watch, youtube, google, facebook, verizon, thanks
- **lda 1** (13,051 tweets, 43% negative): game, play, player, battlefield, games, fix, time, legends
- **lda 2** (4,719 tweets, 31% negative): com, pic, twitter, rhandlerr, facebook, google, org, beautiful
- **lda 3** (6,218 tweets, 40% negative): fuck, shit, black, fucking, game, looks, war, don
- **lda 4** (6,573 tweets, 20% negative): twitch, people, stream, thank, fun, live, borderlands, overwatch
- **lda 5** (3,537 tweets, 28% negative): johnson, com, baby, powder, company, vaccine, stop, covid
- **lda 6** (4,036 tweets, 27% negative): unk, hearthstone, level, love, hell, fans, yes, playing
- **lda 7** (6,492 tweets, 19% negative): xbox, com, series, best, news, nvidia, playstation, looking
- **lda 8** (4,046 tweets, 30% negative): dead, red, gta, redemption, online, order, service, work
- **lda 9** (4,726 tweets, 20% negative): world, love, creed, warcraft, assassin, amazing, don, assassins
- **lda 10** (2,881 tweets, 32% negative): pubg, com, microsoft, banned, mobile, change, ban, app
- **lda 11** (3,944 tweets, 40% negative): home, depot, fifa, people, microsoft, problems, windows, sucks

### NMF - k = 8
_max mean UMass coherence over the grid (override: TSP_NMF_K)_

- **nmf 0** (8,211 tweets, 25% negative): pic, twitter, com, rhandlerr, facebook, playstation, nice, thank
- **nmf 1** (41,598 tweets, 33% negative): game, play, best, shit, time, playing, games, fix
- **nmf 2** (2,004 tweets, 26% negative): player, ban, battlefield, details, occurred, com, imposed, pubg
- **nmf 3** (2,140 tweets, 19% negative): dead, red, redemption, redeeming, twitch, online, games, time
- **nmf 4** (2,724 tweets, 9% negative): love, world, playing, people, home, warcraft, legends, god
- **nmf 5** (3,054 tweets, 26% negative): unk, happy, wtf, com, home, youtu, playstation, depot
- **nmf 6** (3,429 tweets, 53% negative): fuck, shit, fucking, don, home, depot, ass, say
- **nmf 7** (3,794 tweets, 30% negative): johnson, powder, baby, talc, stop, selling, based, vaccine

### KMEANS - k = 10
_documented default 10 (override: TSP_KMEANS_K); silhouette reported, low silhouette is typical for short texts and is NOT proof of semantic quality_

- **kmeans 0** (484 tweets, 1% negative): player, ban, details, occurred, battlefield, com, spell, imposed
- **kmeans 1** (5,488 tweets, 20% negative): love, play, xbox, games, series, don, video, playing
- **kmeans 2** (1,527 tweets, 17% negative): dead, red, redemption, redeeming, game, com, pic, games
- **kmeans 3** (1,221 tweets, 60% negative): fuck, game, don, fucking, shit, play, ass, fifa
- **kmeans 4** (47,083 tweets, 30% negative): unk, com, johnson, shit, best, people, facebook, great
- **kmeans 5** (6 tweets, 0% negative): zelda, breath, newest, legend, wild, assassin, creed, wait
- **kmeans 6** (3,279 tweets, 35% negative): pic, twitter, com, rhandlerr, facebook, game, wikipedia, org
- **kmeans 7** (1,343 tweets, 41% negative): home, depot, damn, com, lowe, walmart, people, love
- **kmeans 8** (5,795 tweets, 49% negative): game, rainbow, fix, play, played, playing, best, fucking
- **kmeans 9** (728 tweets, 5% negative): excited, game, creed, assassin, playing, finally, super, hearthstone

Topic labels are machine-generated term lists. No human validation is claimed unless `results/<mode>/topics/topic_labels.json` has been filled in by a reviewer.

## 5. Environment
```json
{
  "python": "3.12.3",
  "platform": "Linux-7.0.0-38-generic-x86_64-with-glibc2.39",
  "cpu_count": 12,
  "packages": {
    "numpy": "2.5.2",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1",
    "nltk": "3.10.4",
    "torch": "2.11.0+cu128",
    "transformers": "5.20.2",
    "streamlit": "1.66.1",
    "plotly": "7.1.0"
  },
  "stage_seconds": {
    "data": 2.3,
    "eda": 12.0,
    "classical": 554.4,
    "topics": 158.4,
    "transformer": 7800.1
  },
  "label_mode": "four"
}
```


