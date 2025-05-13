import logging
import os
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def read_raw_data(train_path, test_path):
    """Read raw data files using Polars"""
    logger.info(f"Reading train data from {train_path}")
    train_df = pl.read_csv(train_path)

    logger.info(f"Reading test data from {test_path}")
    test_df = pl.read_csv(test_path)

    return train_df, test_df


def apply_dtypes(df, is_train=True):
    """Apply the correct dtypes to the dataframe"""
    dtypes = {
        # IDs and integers
        "srch_id": pl.Int64,
        "site_id": pl.Int8,  # max 34
        "visitor_location_country_id": pl.Int32,
        "prop_country_id": pl.Int32,
        "prop_id": pl.Int64,
        "prop_starrating": pl.Int8,  # 0-5 star rating
        "srch_destination_id": pl.Int32,
        "srch_length_of_stay": pl.Int16,
        "srch_booking_window": pl.Int32,
        "srch_adults_count": pl.Int8,
        "srch_children_count": pl.Int8,
        "srch_room_count": pl.Int8,
        # Floats
        "visitor_hist_starrating": pl.Float32,  # Can be null
        "visitor_hist_adr_usd": pl.Float32,  # Can be null
        "prop_review_score": pl.Float32,  # Can be null
        "prop_location_score1": pl.Float32,
        "prop_location_score2": pl.Float32,
        "prop_log_historical_price": pl.Float32,
        "price_usd": pl.Float32,
        "srch_query_affinity_score": pl.Float32,  # Can be null
        "orig_destination_distance": pl.Float32,  # Can be null
        # Boolean fields stored as int8
        "prop_brand_bool": pl.Int8,  # +1 or 0
        "promotion_flag": pl.Int8,  # +1 or 0
        "srch_saturday_night_bool": pl.Int8,  # +1 or 0
        "random_bool": pl.Int8,  # +1 or 0
        # Competitor data (can be null)
        "comp1_rate": pl.Int8,  # +1, 0, -1, or null
        "comp1_inv": pl.Int8,  # +1, 0, or null
        "comp1_rate_percent_diff": pl.Float32,  # Can be null
        "comp2_rate": pl.Int8,
        "comp2_inv": pl.Int8,
        "comp2_rate_percent_diff": pl.Float32,
        "comp3_rate": pl.Int8,
        "comp3_inv": pl.Int8,
        "comp3_rate_percent_diff": pl.Float32,
        "comp4_rate": pl.Int8,
        "comp4_inv": pl.Int8,
        "comp4_rate_percent_diff": pl.Float32,
        "comp5_rate": pl.Int8,
        "comp5_inv": pl.Int8,
        "comp5_rate_percent_diff": pl.Float32,
        "comp6_rate": pl.Int8,
        "comp6_inv": pl.Int8,
        "comp6_rate_percent_diff": pl.Float32,
        "comp7_rate": pl.Int8,
        "comp7_inv": pl.Int8,
        "comp7_rate_percent_diff": pl.Float32,
        "comp8_rate": pl.Int8,
        "comp8_inv": pl.Int8,
        "comp8_rate_percent_diff": pl.Float32,
    }

    # Add train-specific dtypes
    if is_train:
        train_specific = {
            "position": pl.Int32,
            "click_bool": pl.Int8,  # 1 or 0
            "booking_bool": pl.Int8,  # 1 or 0
            "gross_bookings_usd": pl.Float32,  # Can be null
        }
        dtypes.update(train_specific)

    # Date column will be handled separately
    date_columns = ["date_time"]

    # Convert columns to their proper data types
    for col in df.columns:
        if col in dtypes and col in df.columns:
            try:
                df = df.with_columns([pl.col(col).cast(dtypes[col], strict=False)])
            except Exception as e:
                logger.warning(f"Failed to cast {col} to {dtypes[col]}: {e}")

    # Handle date columns
    for col in date_columns:
        if col in df.columns:
            try:
                df = df.with_columns([pl.col(col).str.to_datetime()])
            except Exception as e:
                logger.warning(f"Failed to cast {col} to datetime: {e}")

    return df


def feature_engineering(
    train_df, test_df, fill_means=True, fill_medians=False, add_mean=True, add_median=True, add_std=True, save_path=None
):
    """Perform imputing & feature engineering on Expedia dataset:

      IMPUTATION
    - Optionally fill missing values with per-property means
    - Optionally fill missing values with per-property medians

      FEATURE ENGINEERING
    - Optionally add per-property mean features
    - Optionally add per-property median features
    - Optionally add per-property standard deviation features
    """
    # Apply correct dtypes
    logger.info("Applying correct data types")
    train_df = apply_dtypes(train_df, is_train=True)
    test_df = apply_dtypes(test_df, is_train=False)

    # Get all prop columns
    prop_columns = [col for col in train_df.columns if col.startswith("prop")]
    logger.info(f"Property columns for aggregation: {prop_columns}")

    # Exclude identifiers
    agg_columns = [col for col in prop_columns if col not in ["prop_id", "prop_country_id"]]

    # Imputation
    if fill_means:
        logger.info("Filling missing values with property means")
        train_df = mean_fill(train_df, agg_columns)
        test_df = mean_fill(test_df, agg_columns)

    if fill_medians:
        logger.info("Filling missing values with property medians")
        train_df = median_fill(train_df, agg_columns)
        test_df = median_fill(test_df, agg_columns)

    # Align columns for combined operations
    train_df, test_df = align_columns(train_df, test_df)
    combined_df = pl.concat([train_df, test_df])

    # Calculate aggregations on combined dataset
    if add_mean or add_median or add_std:
        logger.info("Calculating aggregated features on combined dataset")

        agg_expressions = []

        if add_mean:
            agg_expressions.extend([pl.col(col).mean().alias(f"{col}_mean") for col in agg_columns])

        if add_median:
            agg_expressions.extend([pl.col(col).median().alias(f"{col}_median") for col in agg_columns])

        if add_std:
            agg_expressions.extend([pl.col(col).std().alias(f"{col}_std") for col in agg_columns])

        agg_df = combined_df.group_by("prop_id").agg(agg_expressions)

        # Join aggregations back to train and test
        logger.info("Joining aggregated features to train data")
        train_df = train_df.join(agg_df, on="prop_id", how="left")

        logger.info("Joining aggregated features to test data")
        test_df = test_df.join(agg_df, on="prop_id", how="left")

    # Save processed data if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train_processed.feather")
        test_path = os.path.join(save_path, "test_processed.feather")

        logger.info(f"Saving processed training data to {train_path}")
        train_df.write_ipc(train_path)

        logger.info(f"Saving processed test data to {test_path}")
        test_df.write_ipc(test_path)

    return train_df, test_df


def mean_fill(df, agg_columns):
    # Compute mean per prop_id for each relevant column
    agg_df = df.group_by("prop_id").agg([pl.col(col).mean().alias(f"{col}_mean") for col in agg_columns])

    # Join means back to df
    df = df.join(agg_df, on="prop_id", how="left")

    # Fill nulls with correct means
    for col in agg_columns:
        if col in df.columns:
            global_mean = df.select(pl.col(col).mean()).item()

            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(pl.when(pl.col(f"{col}_mean").is_not_null()).then(pl.col(f"{col}_mean")).otherwise(global_mean))
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Drop helper column
    df = df.drop([f"{col}_mean" for col in agg_columns if f"{col}_mean" in df.columns])

    return df


def median_fill(df, agg_columns):
    # Compute medians per id
    agg_df = df.group_by("prop_id").agg([pl.col(col).median().alias(f"{col}_median") for col in agg_columns])

    # Join medians to df
    df = df.join(agg_df, on="prop_id", how="left")

    # Fill nulls with medians
    # Fallback: global median
    for col in agg_columns:
        if col in df.columns:
            global_median = df.select(pl.col(col).median()).item()

            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(
                    pl.when(pl.col(f"{col}_median").is_not_null())
                    .then(pl.col(f"{col}_median"))
                    .otherwise(global_median)
                )
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Drop helper columns
    df = df.drop([f"{col}_median" for col in agg_columns if f"{col}_median" in df.columns])

    return df


def add_mean_features(train_df, test_df, agg_columns):
    # Align train and test
    train_df, test_df = align_columns(train_df, test_df)
    # Compute mean per prop_id
    combined_df = pl.concat([train_df, test_df])
    mean_df = combined_df.group_by("prop_id").agg([pl.col(col).mean().alias(f"{col}_mean") for col in agg_columns])

    # Join mean features to df
    train_df = train_df.join(mean_df, on="prop_id", how="left")

    return train_df


def add_median_features(train_df, test_df, agg_columns):
    # Align train and test
    train_df, test_df = align_columns(train_df, test_df)

    # Compute median per prop_id
    combined_df = pl.concat([train_df, test_df])
    median_df = combined_df.group_by("prop_id").agg(
        [pl.col(col).median().alias(f"{col}_median") for col in agg_columns]
    )

    # Join median features to df
    train_df = train_df.join(median_df, on="prop_id", how="left")

    return train_df


def add_std_features(train_df, test_df, agg_columns):
    # Align train and test
    train_df, test_df = align_columns(train_df, test_df)

    # Compute standard deviation per prop_id
    combined_df = pl.concat([train_df, test_df])
    std_df = combined_df.group_by("prop_id").agg([pl.col(col).std().alias(f"{col}_std") for col in agg_columns])

    # Join stddevs back to df
    train_df = train_df.join(std_df, on="prop_id", how="left")

    return train_df


def align_columns(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_cols = sorted(set(df1.columns) | set(df2.columns))

    def is_numeric(dtype):
        return dtype in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        )

    def cast_column(colname, dtype1, dtype2):
        if dtype1 == dtype2:
            return dtype1
        elif is_numeric(dtype1) and is_numeric(dtype2):
            return pl.Float64
        else:
            return pl.Utf8

    schema1 = df1.schema
    schema2 = df2.schema
    new_schema = {}

    for col in all_cols:
        dtype1 = schema1.get(col)
        dtype2 = schema2.get(col)

        if dtype1 and dtype2:
            new_schema[col] = cast_column(col, dtype1, dtype2)
        elif dtype1:
            new_schema[col] = dtype1
        elif dtype2:
            new_schema[col] = dtype2

    def prepare(df, schema):
        return df.select(
            [
                (pl.col(c).cast(schema[c]) if c in df.columns else pl.lit(None).cast(schema[c])).alias(c)
                for c in all_cols
            ]
        )

    return prepare(df1, new_schema), prepare(df2, new_schema)


def data_cleaning(train_df):
    # Remove outliers in price_usd
    quarter1 = train_df.select(pl.col("price_usd").quantile(0.25, "nearest")).item()
    quarter3 = train_df.select(pl.col("price_usd").quantile(0.75, "nearest")).item()  # Fixed: changed from 0.25 to 0.75

    iqr = quarter3 - quarter1

    upper_bound = quarter3 + 1.5 * iqr

    train_df = train_df.filter(pl.col("price_usd") <= upper_bound)

    # Fill small amount of values with 0 as 5% is 0 already
    train_df = train_df.with_columns(pl.col("prop_review_score").fill_null(0))

    return train_df


def main():
    """Main data processing function"""
    # Set up logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Get project directory
    project_dir = Path(__file__).resolve().parents[3]

    # Define paths
    raw_dir = os.path.join(project_dir, "data", "raw")
    processed_dir = os.path.join(project_dir, "data", "processed")

    train_path = os.path.join(raw_dir, "training_set_VU_DM.csv")
    test_path = os.path.join(raw_dir, "test_set_VU_DM.csv")

    train_df, test_df = read_raw_data(train_path, test_path)

    # -- Data cleaning --
    # train_df = data_cleaning(train_df)

    # -- Feature engineering --
    train_df, test_df = feature_engineering(
        train_df,
        test_df,
        fill_means=True,
        fill_medians=False,
        add_mean=True,
        add_median=True,
        add_std=True,
        save_path=processed_dir,
    )

    logger.info("Data processing completed.")


if __name__ == "__main__":
    main()
