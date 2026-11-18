"""Stage 3: train one LightGBM per fold and average the 5 test predictions.

For each fold k, reads the stage-2 files, adds the length-bin target features,
trains, and writes:
    output/pred_valid_k.csv   predictions on the fold's validation rows
    output/pred_test_k.csv    predictions on the test set
    output/importance_k.csv   gain feature importance
Once all 5 test prediction files exist, it writes output/submission.csv.

Usage:
    python model.py               # train all folds, then write the submission
    python model.py --fold 2      # train a single fold
    python model.py --submit      # only blend existing pred_test_*.csv files
"""

import argparse
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    EP_LEN, N_FOLDS, SAMPLE_SUBMISSION_CSV, SEED, SUBMISSION_CSV, TARGET,
    add_group_stats, cast_categoricals, fold_path,
)

N_BINS = [5, 10, 15, 20, 25, 50, 100]

LGBM_PARAMS = dict(
    n_estimators=50_000,
    max_depth=-1,
    num_leaves=1024,
    colsample_bytree=0.2,
    subsample=0.7,
    learning_rate=0.005,
    objective="l2",
    metric="rmse",
    verbosity=-1,
    max_bin=1024,
    random_state=SEED,
)
EARLY_STOPPING_ROUNDS = 200
LOG_EVERY = 100


def load_fold(fold: int):
    X_train = pd.read_parquet(fold_path("X_train", fold))
    X_valid = pd.read_parquet(fold_path("X_valid", fold))
    X_test = pd.read_parquet(fold_path("X_test", fold))
    y_train = pd.read_csv(fold_path("y_train", fold))[TARGET]
    y_valid = pd.read_csv(fold_path("y_valid", fold))[TARGET]
    return X_train, y_train, X_valid, y_valid, X_test


def add_length_bin_features(X_train, X_valid, X_test) -> None:
    """Put episode length into quantile bins (edges taken from the train rows),
    then add the target mean and median of each bin."""
    frames = (X_train, X_valid, X_test)
    bin_cols = []
    for n in N_BINS:
        name = f"{EP_LEN}{n}"
        _, edges = pd.qcut(X_train[EP_LEN], n, retbins=True, labels=False)
        for df in frames:
            df[name] = pd.cut(df[EP_LEN], bins=edges, labels=False,
                              include_lowest=True).astype("category")
        bin_cols.append(name)

    add_group_stats(frames, X_train, bin_cols, TARGET, ["mean", "quantile"],
                    name_fn=lambda stat, key: f"{TARGET}{stat}{key}")

    for df in frames:
        df["Long_Episode"] = (df[EP_LEN] > 75).astype(int)


def train_fold(fold: int) -> float:
    X_train, y_train, X_valid, y_valid, X_test = load_fold(fold)
    for df in (X_train, X_valid, X_test):
        cast_categoricals(df)
    add_length_bin_features(X_train, X_valid, X_test)

    X_train = X_train.drop(columns=[TARGET])
    X_valid = X_valid.drop(columns=[TARGET])
    X_test = X_test[X_train.columns]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(LOG_EVERY),
                   lgb.early_stopping(EARLY_STOPPING_ROUNDS)],
    )

    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred_valid - y_valid.to_numpy()) ** 2)))
    print(f"fold {fold}: valid RMSE {rmse:.5f}")

    pd.DataFrame({TARGET: pred_valid}).to_csv(fold_path("pred_valid", fold), index=False)
    pd.DataFrame({TARGET: pred_test}).to_csv(fold_path("pred_test", fold), index=False)
    (pd.Series(model.booster_.feature_importance("gain"), index=X_train.columns, name="gain")
       .sort_values(ascending=False)
       .to_csv(fold_path("importance", fold)))
    return rmse


def write_submission() -> None:
    paths = [fold_path("pred_test", k) for k in range(1, N_FOLDS + 1)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"Submission not written; missing {missing}")
        return
    preds = np.mean([pd.read_csv(p)[TARGET].to_numpy() for p in paths], axis=0)
    subm = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    subm[TARGET] = preds
    subm.to_csv(SUBMISSION_CSV, index=False)
    print(f"Saved {SUBMISSION_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1),
                        help="train only this fold (1-based); default: all")
    parser.add_argument("--submit", action="store_true",
                        help="skip training, only blend existing fold predictions")
    args = parser.parse_args()

    if not args.submit:
        folds = [args.fold] if args.fold else range(1, N_FOLDS + 1)
        for fold in folds:
            train_fold(fold)
            gc.collect()
    write_submission()


if __name__ == "__main__":
    main()
