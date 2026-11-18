"""Shared configuration and helpers for the 3-stage pipeline.

The pipeline is split into three scripts that hand data to each other through
parquet files on disk, so no single process ever holds every intermediate
DataFrame in RAM:

    pre_preprocess.py  ->  preprocess.py  ->  model.py
"""

import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Paths (override with environment variables)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PODCAST_DATA_DIR", ROOT / "data"))
WORK_DIR = Path(os.environ.get("PODCAST_WORK_DIR", ROOT / "output"))

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
ORIGINAL_CSV = DATA_DIR / "podcast_dataset.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

FULL_TRAIN_PARQUET = WORK_DIR / "df_X.parquet.gzip"
FULL_TEST_PARQUET = WORK_DIR / "df_test.parquet.gzip"
SUBMISSION_CSV = WORK_DIR / "submission.csv"


def fold_path(name: str, fold: int) -> Path:
    """Per-fold file, e.g. fold_path('X_train', 1) -> output/X_train_1.parquet.gzip."""
    ext = ".csv" if name.startswith(("y_", "pred_", "importance")) else ".parquet.gzip"
    return WORK_DIR / f"{name}_{fold}{ext}"


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
TARGET = "Listening_Time_minutes"
EP_LEN = "Episode_Length_minutes"

CATEGORICAL_COLS = [
    "Genre", "Podcast_Name", "Publication_Day",
    "Publication_Time", "Episode_Sentiment", "Episode_Title",
]

# Base columns used for arithmetic pairs and for the string-combination features.
BASE_COLS = [
    "Episode_Length_minutes", "Episode_Title", "Host_Popularity_percentage",
    "Number_of_Ads", "Episode_Sentiment", "Publication_Day", "Publication_Time",
    "Podcast_Name", "Guest_Popularity_percentage", "Genre",
]

COMBO_SIZES = (2, 3, 4)


def combo_name(cols) -> str:
    return "_".join(cols)


def combo_columns(r: int) -> list[str]:
    """Names of the r-way string-combination features of BASE_COLS."""
    return [combo_name(c) for c in combinations(BASE_COLS, r)]


# All combination columns (2-, 3- and 4-way): these get target-encoded.
TE_COLS = [name for r in COMBO_SIZES for name in combo_columns(r)]

N_FOLDS = 5
SEED = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cast_categoricals(df: pd.DataFrame, cols=CATEGORICAL_COLS) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].astype("category")
    return df


def group_stat(train: pd.DataFrame, key: str, value: str, stat: str) -> pd.DataFrame:
    """`stat` of `value` grouped by `key` on the training rows.

    'quantile' with no argument is the median.
    """
    grouped = train.groupby(key, observed=False)[value]
    return getattr(grouped, stat)().rename("stat").reset_index()


def lookup(df: pd.DataFrame, key: str, table: pd.DataFrame) -> np.ndarray:
    """Left-join `table` on `key`. Only the key column is copied, not the whole
    frame, which keeps memory low. Row order is preserved."""
    return df[[key]].merge(table, on=key, how="left")["stat"].to_numpy()


def add_group_stats(frames, train, keys, value, stats, name_fn):
    """Compute stats on `train`, then add them as new columns to every frame in `frames`."""
    for stat in stats:
        for key in keys:
            table = group_stat(train, key, value, stat)
            name = name_fn(stat, key)
            for df in frames:
                df[name] = lookup(df, key, table)
