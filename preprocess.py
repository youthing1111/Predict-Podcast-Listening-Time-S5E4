"""Stage 2: per-fold, target-dependent feature engineering.

For each of the 5 KFold splits, every target statistic and target encoding is
fitted on that fold's training rows only, then applied to the train, valid and
test rows. Writes, for each fold k:
    output/X_train_k.parquet.gzip, X_valid_k.parquet.gzip, X_test_k.parquet.gzip
    output/y_train_k.csv, y_valid_k.csv

Usage:
    python preprocess.py              # all folds
    python preprocess.py --fold 3     # a single fold (lowest RAM: one process per fold)
"""

import argparse
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

from common import (
    EP_LEN, FULL_TEST_PARQUET, FULL_TRAIN_PARQUET, N_FOLDS, SEED, TARGET,
    TE_COLS, add_group_stats, combo_columns, fold_path,
)

STATS = ["mean", "max", "min", "std", "quantile", "skew", "sum"]
STAT_KEYS = [
    "NaNs",
    "Episode_Length_minutes_round0", "Host_Popularity_percentage_round0",
    "Guest_Popularity_percentage_round0",
    "Episode_Length_minutes_digit0", "Episode_Length_minutes_digit1",
    "Host_Popularity_percentage_digit0", "Host_Popularity_percentage_digit1",
    "Guest_Popularity_percentage_digit0", "Guest_Popularity_percentage_digit1",
    "Publication_Day", "Podcast_Name", "Publication_Time", "Episode_Title",
    "Genre", "Number_of_Ads",
]
# Episode length divided by the target encoding of these combinations.
RATIO_COLS = combo_columns(2) + combo_columns(3)


def build_fold(X_train, y_train, X_valid, X_test):
    frames = (X_train, X_valid, X_test)

    # 1. Target stats grouped by each key (7 stats x 16 keys).
    add_group_stats(frames, X_train, STAT_KEYS, TARGET, STATS,
                    name_fn=lambda stat, key: f"{TARGET}{stat}{key}")

    # 2. Target encoding of the 2/3/4-way string combinations.
    encoder = TargetEncoder(random_state=SEED)
    X_train[TE_COLS] = encoder.fit_transform(X_train[TE_COLS], y_train)
    X_valid[TE_COLS] = encoder.transform(X_valid[TE_COLS])
    X_test[TE_COLS] = encoder.transform(X_test[TE_COLS])

    # 3. Episode length relative to the encoded combinations, plus cyclic and
    #    integer/decimal parts of the length.
    for df in frames:
        for c in RATIO_COLS:
            df[f"{EP_LEN}_div_{c}"] = df[EP_LEN] / df[c]
        df["SinEpLen"] = np.sin(2 * np.pi * df[EP_LEN] / 60)
        df["CosEpLen"] = np.cos(2 * np.pi * df[EP_LEN] / 60)
        df["ELen_Int"] = np.floor(df[EP_LEN])
        df["ELen_Dec"] = df[EP_LEN] - df["ELen_Int"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1),
                        help="process only this fold (1-based); default: all")
    args = parser.parse_args()

    X = pd.read_parquet(FULL_TRAIN_PARQUET)
    y = X[TARGET]
    splits = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    folds = [args.fold] if args.fold else range(1, N_FOLDS + 1)

    for fold in folds:
        idx_train, idx_valid = splits[fold - 1]
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
        # Reload the test set fresh for every fold, so features from one fold
        # never leak into another.
        X_test = pd.read_parquet(FULL_TEST_PARQUET)

        build_fold(X_train, y_train, X_valid, X_test)
        print(f"fold {fold}: train {X_train.shape}, valid {X_valid.shape}, test {X_test.shape}")

        X_train.to_parquet(fold_path("X_train", fold), compression="gzip")
        X_valid.to_parquet(fold_path("X_valid", fold), compression="gzip")
        X_test.to_parquet(fold_path("X_test", fold), compression="gzip")
        y_train.to_csv(fold_path("y_train", fold), index=False)
        y_valid.to_csv(fold_path("y_valid", fold), index=False)

        del X_train, X_valid, X_test, y_train, y_valid
        gc.collect()


if __name__ == "__main__":
    main()
