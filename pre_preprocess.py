"""Stage 1: fold-independent feature engineering.

Reads the raw Kaggle CSVs plus the original podcast dataset and builds every
feature that does not use the target. Writes:
    output/df_X.parquet.gzip     train + original data (includes the target)
    output/df_test.parquet.gzip  test data

Usage:  python pre_preprocess.py
"""

from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    BASE_COLS, COMBO_SIZES, EP_LEN, FULL_TEST_PARQUET, FULL_TRAIN_PARQUET,
    ORIGINAL_CSV, TARGET, TEST_CSV, TRAIN_CSV, WORK_DIR, cast_categoricals,
    combo_name,
)

POPULARITY_COLS = [EP_LEN, "Host_Popularity_percentage", "Guest_Popularity_percentage"]

PODCAST_CODES = {
    "Mystery Matters": 48, "Joke Junction": 1, "Study Sessions": 2, "Digital Digest": 3,
    "Mind & Body": 4, "Fitness First": 5, "Criminal Minds": 6, "News Roundup": 7,
    "Daily Digest": 8, "Music Matters": 9, "Sports Central": 10, "Melody Mix": 11,
    "Game Day": 12, "Gadget Geek": 13, "Global News": 14, "Tech Talks": 15,
    "Sport Spot": 16, "Funny Folks": 17, "Sports Weekly": 18, "Business Briefs": 19,
    "Tech Trends": 20, "Innovators": 21, "Health Hour": 22, "Comedy Corner": 23,
    "Sound Waves": 24, "Brain Boost": 25, "Athlete's Arena": 26, "Wellness Wave": 27,
    "Style Guide": 28, "World Watch": 29, "Humor Hub": 30, "Money Matters": 31,
    "Healthy Living": 32, "Home & Living": 33, "Educational Nuggets": 34,
    "Market Masters": 35, "Learning Lab": 36, "Lifestyle Lounge": 37,
    "Crime Chronicles": 38, "Detective Diaries": 39, "Life Lessons": 40,
    "Current Affairs": 41, "Finance Focus": 42, "Laugh Line": 43,
    "True Crime Stories": 44, "Business Insights": 45, "Fashion Forward": 46,
    "Tune Time": 47,
}
GENRE_CODES = {
    "True Crime": 10, "Comedy": 1, "Education": 2, "Technology": 3, "Health": 4,
    "News": 5, "Music": 6, "Sports": 7, "Business": 8, "Lifestyle": 9,
}
DAY_CODES = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
             "Friday": 5, "Saturday": 6, "Sunday": 7}
TIME_CODES = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}
SENTIMENT_CODES = {"Negative": -1, "Neutral": 1, "Positive": 2}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Episode length is read as a string first, so we can count how many
    # decimal digits the raw value had before converting it to float.
    str_len = {EP_LEN: str}
    train = pd.read_csv(TRAIN_CSV, index_col="id", dtype=str_len)
    test = pd.read_csv(TEST_CSV, index_col="id", dtype=str_len)
    original = pd.read_csv(ORIGINAL_CSV, dtype=str_len)

    for df in (train, test, original):
        df["decimal_digits_e"] = df[EP_LEN].map(
            lambda s: None if pd.isna(s) else (len(s.split(".")[1]) if "." in s else 0)
        )
        df[EP_LEN] = df[EP_LEN].astype(float)

    train = train[train["Number_of_Ads"].notna()]
    original = original[original[TARGET].notna()].drop_duplicates()
    train = pd.concat([train, original])
    return train, test


def add_nan_features(df: pd.DataFrame) -> None:
    # Bit mask of which columns are missing: bit 0 = episode length, bit 1 = guest popularity.
    df["NaNs"] = np.float32(0)
    for i, c in enumerate([EP_LEN, "Guest_Popularity_percentage"]):
        df["NaNs"] += df[c].isna() * 2**i
        df[f"{c}_nan_wc"] = df[c].isna() * 100 + df[EP_LEN]


def add_digit_features(df: pd.DataFrame) -> None:
    # NOTE: every one of these features is computed from Episode_Length_minutes,
    # including the ones named after Host/Guest popularity.
    for c in POPULARITY_COLS:
        df[f"{c}_round0"] = df[EP_LEN].round(0)
    for c in POPULARITY_COLS:
        for k in (0, 1):
            df[f"{c}_digit{k}"] = ((df[EP_LEN] * 10**k) % 10).fillna(-1).astype("int8")
    for c in POPULARITY_COLS:
        df[f"{c}_digit0_1"] = ((df[f"{c}_digit0"] + 1) * 11 + df[f"{c}_digit1"] + 1).astype("int8")


def encode_labels(df: pd.DataFrame) -> None:
    df["Episode_Title"] = df["Episode_Title"].str[len("Episode "):].astype(int)
    df["Genre"] = df["Genre"].replace(GENRE_CODES)
    df["Podcast_Name"] = df["Podcast_Name"].replace(PODCAST_CODES)
    df["Publication_Day"] = df["Publication_Day"].replace(DAY_CODES)
    df["Publication_Time"] = df["Publication_Time"].replace(TIME_CODES)
    df["Episode_Sentiment"] = df["Episode_Sentiment"].replace(SENTIMENT_CODES)


def add_pair_arithmetic(df: pd.DataFrame) -> None:
    for a, b in combinations(BASE_COLS, 2):
        df[f"{a}_sum_{b}"] = df[a] + df[b]
        df[f"{a}_divide_{b}"] = df[a] / (df[b] + 0.1)
        df[f"{a}_mul_{b}"] = df[a] * df[b]
        df[f"{a}_sub_{b}"] = df[a] - df[b]


def add_string_combos(df: pd.DataFrame) -> None:
    """Join every 2-, 3- and 4-column combination of BASE_COLS into one string
    category (target-encoded in stage 2). Each column is converted to string once
    and the parts are joined with vectorised '+'."""
    as_str = {c: df[c].astype(str) for c in BASE_COLS}
    for r in COMBO_SIZES:
        for cols in tqdm(list(combinations(BASE_COLS, r)), desc=f"{r}-way combos"):
            joined = as_str[cols[0]]
            for c in cols[1:]:
                joined = joined + "_" + as_str[c]
            df[combo_name(cols)] = joined.astype("category")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    add_nan_features(df)
    add_digit_features(df)
    encode_labels(df)
    add_pair_arithmetic(df)
    cast_categoricals(df)
    add_string_combos(df)
    return df


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()

    # Build and save one frame at a time to limit peak memory.
    build_features(test).to_parquet(FULL_TEST_PARQUET, compression="gzip")
    del test
    build_features(train).to_parquet(FULL_TRAIN_PARQUET, compression="gzip")
    print(f"Saved {FULL_TRAIN_PARQUET} and {FULL_TEST_PARQUET}")


if __name__ == "__main__":
    main()
