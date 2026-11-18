# Predict Podcast Listening Time (Kaggle Playground S5E4)

This is the code for my submission to the Kaggle Playground Series S5E4 competition, which ranked **24 / 3300**.
The goal is to predict `Listening_Time_minutes` for a podcast episode.

## Pipeline

The work is split into **three stages that run as separate processes**. The stages pass data to each other through
parquet files on disk. This is deliberate: the full feature set is roughly 860 columns × 800k rows, too big to build
and train in one process on a machine with limited RAM.

```
data/*.csv ──► pre_preprocess.py ──► output/df_X, df_test ──► preprocess.py ──► output/X_{train,valid,test}_k ──► model.py ──► output/submission.csv
              (no target used)                              (per fold, target-based)                          (per fold LightGBM)
```

| Stage | Script | What it does |
|---|---|---|
| 1 | `pre_preprocess.py` | Merges the competition train data with the [original podcast dataset](https://www.kaggle.com/datasets/ysthehurricane/podcast-listening-time-prediction-dataset). Adds features that don't use the target: decimal-digit count, NaN bitmask, rounding and digit features of episode length, label codes, pairwise `+ − × ÷` of 10 base columns, and 375 string-combination categories (every 2-, 3- and 4-column combination). |
| 2 | `preprocess.py` | Uses a 5-fold `KFold` split (seed 42). For each fold it fits on the train rows only and applies to valid and test: 7 target stats × 16 grouping keys, `TargetEncoder` on all 375 combinations, episode length ÷ the encoded 2- and 3-way combinations, and sin/cos/int/dec of episode length. |
| 3 | `model.py` | For each fold it adds quantile bins of episode length with the target mean and median per bin, then trains LightGBM (1024 leaves, lr 0.005, early stopping). The final prediction is the mean of the 5 fold test predictions. |

`common.py` holds the shared paths, column lists and group-stat helpers.

## Usage

```bash
pip install -r requirements.txt
unzip Data.zip -x '__MACOSX/*' -d data      # train.csv, test.csv, sample_submission.csv, podcast_dataset.csv

python pre_preprocess.py
python preprocess.py          # or one fold per process: python preprocess.py --fold 1 ... --fold 5
python model.py               # or: python model.py --fold 1 ... then python model.py --submit
```

If RAM is tight, run each fold as its own process with `--fold k`.
`model.py --submit` only averages the existing `pred_test_*.csv` files into `output/submission.csv`.

You can override the input and output folders with `PODCAST_DATA_DIR` and `PODCAST_WORK_DIR` (defaults: `data/` and `output/`).

### Outputs (`output/`)

- `df_X.parquet.gzip`, `df_test.parquet.gzip`: stage 1 features
- `X_{train,valid,test}_k.parquet.gzip`, `y_{train,valid}_k.csv`: stage 2 data for fold k
- `pred_valid_k.csv`, `pred_test_k.csv`, `importance_k.csv`: stage 3 results for fold k
- `submission.csv`

## Notes

- The `*_round0`, `*_digitK` and `*_nan_wc` features named after Host/Guest popularity are all computed from
  `Episode_Length_minutes`.
- Each fold reloads a fresh test set, so test features always come from that fold's training rows only.
