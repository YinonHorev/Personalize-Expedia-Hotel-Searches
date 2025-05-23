import logging
import os
from pathlib import Path

import numpy as np  # Added for cyclical features
import polars as pl

logger = logging.getLogger(__name__)

EPSILON = 1e-6  # Epsilon for safe division


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

    if is_train:
        train_specific = {
            "position": pl.Int32,
            "click_bool": pl.Int8,
            "booking_bool": pl.Int8,
            "gross_booking_usd": pl.Float32,
        }
        dtypes.update(train_specific)

    date_columns = ["date_time"]

    for col_name, dtype in dtypes.items():
        if col_name in df.columns:
            try:
                df = df.with_columns(pl.col(col_name).cast(dtype, strict=False))
            except Exception as e:
                logger.warning(f"Failed to cast {col_name} to {dtype}: {e}")

    for col in date_columns:
        if col in df.columns:
            try:
                df = df.with_columns(pl.col(col).str.to_datetime(strict=False))  # Add strict=False for robustness
            except Exception as e:
                logger.warning(f"Failed to cast {col} to datetime: {e}")
    return df


def _create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Creating 'rating' column and dropping 'click_bool', 'booking_bool'.")
    if "click_bool" in df.columns and "booking_bool" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("booking_bool") == 1)
            .then(pl.lit(5).cast(pl.Int8))
            .when(pl.col("click_bool") == 1)
            .then(pl.lit(1).cast(pl.Int8))
            .otherwise(pl.lit(0).cast(pl.Int8))
            .alias("rating")
        ).drop(["click_bool", "booking_bool"])
    else:
        logger.warning("'click_bool' or 'booking_bool' not found. 'rating' column not created.")
    return df


def _initial_imputations(train_df: pl.DataFrame, test_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("Performing initial specific imputations.")
    # Specific imputation for prop_review_score
    train_df = train_df.with_columns(pl.col("prop_review_score").fill_null(0))
    test_df = test_df.with_columns(pl.col("prop_review_score").fill_null(0))

    # Competitor data imputation
    for i in range(1, 9):
        for col_suffix in ["rate", "inv", "rate_percent_diff"]:
            col_name = f"comp{i}_{col_suffix}"
            if col_name in train_df.columns:
                train_df = train_df.with_columns(pl.col(col_name).fill_null(0))
            if col_name in test_df.columns:
                test_df = test_df.with_columns(pl.col(col_name).fill_null(0))

    # General imputation for specified columns using train stats
    cols_to_impute_globally = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "srch_query_affinity_score",
        "orig_destination_distance",
    ]
    for col in cols_to_impute_globally:
        if col in train_df.columns:
            mean_val = train_df.select(pl.col(col).mean()).item()
            if mean_val is not None:  # Check if mean_val could be calculated (e.g. not all nulls)
                train_df = train_df.with_columns(pl.col(col).fill_null(mean_val))
                if col in test_df.columns:
                    test_df = test_df.with_columns(pl.col(col).fill_null(mean_val))
            else:
                logger.warning(f"Global mean for {col} is None. Skipping imputation for this column.")

    return train_df, test_df


def mean_fill(df: pl.DataFrame, columns_to_fill: list[str], stats_df: pl.DataFrame) -> pl.DataFrame:
    logger.debug(f"Mean filling columns: {columns_to_fill}")
    prop_means_df = stats_df.group_by("prop_id").agg(
        [pl.col(col).mean().alias(f"{col}_prop_mean") for col in columns_to_fill if col in stats_df.columns]
    )
    df = df.join(prop_means_df, on="prop_id", how="left")

    for col in columns_to_fill:
        if col in df.columns and col in stats_df.columns:  # Ensure col exists in both df and stats_df
            global_mean = stats_df.select(pl.col(col).mean()).item()
            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(
                    pl.col(f"{col}_prop_mean").fill_null(global_mean if global_mean is not None else 0)
                )  # Fallback to 0 if global_mean is None
                .otherwise(pl.col(col))
                .alias(col)
            )
            if f"{col}_prop_mean" in df.columns:
                df = df.drop(f"{col}_prop_mean")
        elif col not in stats_df.columns:
            logger.warning(f"Column {col} not in stats_df for mean_fill, cannot calculate global_mean.")
        # If col not in df.columns, it's skipped, which is fine.

    # Cleanup any remaining helper columns if they exist
    for col in columns_to_fill:
        if f"{col}_prop_mean" in df.columns:
            df = df.drop(f"{col}_prop_mean")
    return df


def median_fill(df: pl.DataFrame, columns_to_fill: list[str], stats_df: pl.DataFrame) -> pl.DataFrame:
    logger.debug(f"Median filling columns: {columns_to_fill}")
    prop_medians_df = stats_df.group_by("prop_id").agg(
        [pl.col(col).median().alias(f"{col}_prop_median") for col in columns_to_fill if col in stats_df.columns]
    )
    df = df.join(prop_medians_df, on="prop_id", how="left")

    for col in columns_to_fill:
        if col in df.columns and col in stats_df.columns:
            global_median = stats_df.select(pl.col(col).median()).item()
            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(
                    pl.col(f"{col}_prop_median").fill_null(global_median if global_median is not None else 0)
                )  # Fallback to 0 if global_median is None
                .otherwise(pl.col(col))
                .alias(col)
            )
            if f"{col}_prop_median" in df.columns:
                df = df.drop(f"{col}_prop_median")
        elif col not in stats_df.columns:
            logger.warning(f"Column {col} not in stats_df for median_fill, cannot calculate global_median.")

    for col in columns_to_fill:
        if f"{col}_prop_median" in df.columns:
            df = df.drop(f"{col}_prop_median")
    return df


def _add_property_aggregated_features(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    prop_numeric_features: list[str],
    add_mean: bool,
    add_median: bool,
    add_std: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not (add_mean or add_median or add_std):
        return train_df, test_df

    logger.info("Calculating and adding property aggregated features.")
    # Align before combining for aggregation
    aligned_train_df, aligned_test_df = align_columns(train_df, test_df)
    combined_df_for_aggregation = pl.concat([aligned_train_df, aligned_test_df], how="diagonal")

    agg_expressions = []
    for col in prop_numeric_features:
        if col not in combined_df_for_aggregation.columns:
            logger.warning(f"Column {col} for property aggregation not found in combined_df. Skipping.")
            continue
        if add_mean:
            agg_expressions.append(pl.col(col).mean().alias(f"{col}_mean_by_prop_id"))
        if add_median:
            agg_expressions.append(pl.col(col).median().alias(f"{col}_median_by_prop_id"))
        if add_std:
            agg_expressions.append(pl.col(col).std().alias(f"{col}_std_by_prop_id"))

    if not agg_expressions:
        logger.warning("No aggregation expressions generated for property features.")
        return train_df, test_df

    aggregated_prop_features_df = combined_df_for_aggregation.group_by("prop_id").agg(agg_expressions)

    train_df = train_df.join(aggregated_prop_features_df, on="prop_id", how="left")
    test_df = test_df.join(aggregated_prop_features_df, on="prop_id", how="left")

    return train_df, test_df


def _engineer_datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Engineering datetime features.")
    if "date_time" in df.columns and df["date_time"].dtype == pl.Datetime:
        df = df.with_columns(
            [
                pl.col("date_time").dt.hour().alias("dt_hour"),
                pl.col("date_time").dt.day().alias("dt_day"),
                pl.col("date_time").dt.month().alias("dt_month"),
                pl.col("date_time").dt.year().alias("dt_year"),
                pl.col("date_time").dt.weekday().alias("dt_weekday"),  # Monday=1, Sunday=7
                pl.col("date_time").dt.ordinal_day().alias("dt_dayofyear"),
                pl.col("date_time").dt.week().alias("dt_weekofyear"),
            ]
        )
        df = df.with_columns(
            pl.col("dt_weekday").is_in([6, 7]).cast(pl.Int8).alias("dt_is_weekend")  # Saturday=6, Sunday=7
        )
        # Cyclical features
        df = df.with_columns(
            [
                (np.sin(2 * np.pi * pl.col("dt_hour") / 24)).alias("dt_hour_sin"),
                (np.cos(2 * np.pi * pl.col("dt_hour") / 24)).alias("dt_hour_cos"),
                (np.sin(2 * np.pi * pl.col("dt_weekday") / 7)).alias("dt_weekday_sin"),
                (np.cos(2 * np.pi * pl.col("dt_weekday") / 7)).alias("dt_weekday_cos"),
                (np.sin(2 * np.pi * pl.col("dt_month") / 12)).alias("dt_month_sin"),
                (np.cos(2 * np.pi * pl.col("dt_month") / 12)).alias("dt_month_cos"),
            ]
        )
    else:
        logger.warning("'date_time' column not found or not datetime type. Skipping datetime features.")
    return df


def _engineer_historical_contextual_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("Engineering historical/contextual features.")
    # Average position by prop_id (from training data)
    if "position" in train_df.columns and "prop_id" in train_df.columns:
        avg_pos_by_prop = train_df.group_by("prop_id").agg(pl.col("position").mean().alias("avg_pos_by_prop_id"))
        global_avg_pos = train_df.select(pl.col("position").mean()).item()

        train_df = train_df.join(avg_pos_by_prop, on="prop_id", how="left")
        train_df = train_df.with_columns(pl.col("avg_pos_by_prop_id").fill_null(global_avg_pos))

        test_df = test_df.join(avg_pos_by_prop, on="prop_id", how="left")
        test_df = test_df.with_columns(pl.col("avg_pos_by_prop_id").fill_null(global_avg_pos))

    # User historical features
    for hist_col, prop_col, prefix in [
        ("visitor_hist_starrating", "prop_starrating", "star_rating"),
        ("visitor_hist_adr_usd", "price_usd", "price_usd"),
    ]:
        if hist_col in train_df.columns and prop_col in train_df.columns:
            train_df = train_df.with_columns(
                [
                    (pl.col(prop_col) - pl.col(hist_col)).alias(f"diff_{prefix}_prop_visitor_hist"),
                    (pl.col(prop_col) / (pl.col(hist_col) + EPSILON)).alias(f"ratio_{prefix}_prop_visitor_hist"),
                ]
            )
        if hist_col in test_df.columns and prop_col in test_df.columns:
            test_df = test_df.with_columns(
                [
                    (pl.col(prop_col) - pl.col(hist_col)).alias(f"diff_{prefix}_prop_visitor_hist"),
                    (pl.col(prop_col) / (pl.col(hist_col) + EPSILON)).alias(f"ratio_{prefix}_prop_visitor_hist"),
                ]
            )
    return train_df, test_df


def _engineer_search_context_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Engineering search context features.")
    if "srch_id" not in df.columns:
        logger.warning("'srch_id' not found. Skipping search context features.")
        return df

    # Price-based features
    if "price_usd" in df.columns:
        df = df.with_columns(
            [
                pl.col("price_usd").rank("ordinal", descending=False).over("srch_id").alias("price_rank_in_srch"),
                (pl.col("price_usd") - pl.col("price_usd").mean().over("srch_id")).alias("diff_from_avg_price_in_srch"),
                (
                    (pl.col("price_usd") - pl.col("price_usd").min().over("srch_id"))
                    / (pl.col("price_usd").max().over("srch_id") - pl.col("price_usd").min().over("srch_id") + EPSILON)
                ).alias("norm_price_in_srch"),
            ]
        )

    # Star rating features
    if "prop_starrating" in df.columns:
        df = df.with_columns(
            [
                pl.col("prop_starrating")
                .rank("ordinal", descending=True)
                .over("srch_id")
                .alias("starrating_rank_in_srch"),
                (pl.col("prop_starrating") - pl.col("prop_starrating").mean().over("srch_id")).alias(
                    "diff_from_avg_starrating_in_srch"
                ),
            ]
        )

    # Distance features
    if "orig_destination_distance" in df.columns:
        df = df.with_columns(
            [
                pl.col("orig_destination_distance")
                .rank("ordinal", descending=False)
                .over("srch_id")
                .alias("distance_rank_in_srch"),
                (
                    pl.col("orig_destination_distance") - pl.col("orig_destination_distance").mean().over("srch_id")
                ).alias("diff_from_avg_distance_in_srch"),
            ]
        )

    # Number of properties in search
    df = df.with_columns(pl.col("prop_id").count().over("srch_id").alias("n_properties_in_srch"))
    return df


def _engineer_competitor_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Engineering competitor interaction features.")
    comp_cheaper_cols = []
    comp_more_expensive_cols = []
    comp_inv_available_cols = []
    comp_rate_percent_diff_cols = []

    for i in range(1, 9):
        rate_col = f"comp{i}_rate"
        inv_col = f"comp{i}_inv"
        diff_col = f"comp{i}_rate_percent_diff"

        if rate_col in df.columns:
            df = df.with_columns((pl.col(rate_col) == -1).cast(pl.Int8).alias(f"is_comp{i}_cheaper"))
            comp_cheaper_cols.append(f"is_comp{i}_cheaper")
            df = df.with_columns((pl.col(rate_col) == 1).cast(pl.Int8).alias(f"is_comp{i}_more_expensive"))
            comp_more_expensive_cols.append(f"is_comp{i}_more_expensive")

        if inv_col in df.columns:
            # compX_inv: +1 if no availability, 0 if availability
            df = df.with_columns((pl.col(inv_col) == 0).cast(pl.Int8).alias(f"is_comp{i}_inv_available"))
            comp_inv_available_cols.append(f"is_comp{i}_inv_available")

        if diff_col in df.columns:
            comp_rate_percent_diff_cols.append(diff_col)

    if comp_cheaper_cols:
        df = df.with_columns(pl.sum_horizontal(comp_cheaper_cols).alias("num_comps_cheaper"))
    if comp_more_expensive_cols:
        df = df.with_columns(pl.sum_horizontal(comp_more_expensive_cols).alias("num_comps_more_expensive"))
    if comp_inv_available_cols:
        df = df.with_columns(pl.sum_horizontal(comp_inv_available_cols).alias("num_comps_available"))

    if comp_rate_percent_diff_cols:  # Ensure list is not empty
        # For mean/min/max, handle cases where all might be null for a row
        df = df.with_columns(
            pl.mean_horizontal([pl.col(c) for c in comp_rate_percent_diff_cols if c in df.columns]).alias(
                "avg_comp_rate_percent_diff"
            ),
            pl.min_horizontal([pl.col(c) for c in comp_rate_percent_diff_cols if c in df.columns]).alias(
                "min_comp_rate_percent_diff"
            ),
            pl.max_horizontal([pl.col(c) for c in comp_rate_percent_diff_cols if c in df.columns]).alias(
                "max_comp_rate_percent_diff"
            ),
        )
    return df


def _engineer_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Engineering interaction features.")
    if "price_usd" in df.columns and "prop_starrating" in df.columns:
        df = df.with_columns((pl.col("price_usd") / (pl.col("prop_starrating") + EPSILON)).alias("price_per_star"))
    if "price_usd" in df.columns and "prop_location_score1" in df.columns:
        df = df.with_columns(
            (pl.col("price_usd") * pl.col("prop_location_score1")).alias("price_usd_x_prop_location_score1")
        )
    if "prop_review_score" in df.columns and "prop_starrating" in df.columns:
        df = df.with_columns(
            (pl.col("prop_review_score") * pl.col("prop_starrating")).alias("prop_review_score_x_prop_starrating")
        )

    # diff_current_price_from_prop_avg
    # This requires 'price_usd_mean_by_prop_id' to be created in _add_property_aggregated_features
    if "price_usd" in df.columns and "price_usd_mean_by_prop_id" in df.columns:
        df = df.with_columns(
            (pl.col("price_usd") - pl.col("price_usd_mean_by_prop_id")).alias("diff_current_price_from_prop_avg")
        )
    return df


def _engineer_likelihood_features(
    train_df: pl.DataFrame, test_df: pl.DataFrame, target_column: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("Engineering likelihood features (Target Encoding).")
    # TODO: Implement out-of-fold target encoding for training data and apply to test data.
    # Plan details:
    # Target: rating > 0 (or click_bool, booking_bool). Use 'rating' > 0.
    # Categorical Features: prop_id, srch_destination_id, site_id, visitor_location_country_id, prop_country_id.
    # Smoothing: likelihood = (count * mean_in_cat + global_mean * alpha) / (count + alpha).
    # Use GroupKFold on srch_id for training data.
    logger.warning("Likelihood feature engineering is a placeholder and not fully implemented.")

    # Example for one column, without OOF, just for structure:
    # categorical_cols_for_encoding = ["prop_id", "srch_destination_id"] # Add more as per plan
    # alpha = 20 # Smoothing factor

    # if target_column in train_df.columns:
    #     train_df_with_target = train_df.with_columns((pl.col(target_column) > 0).cast(pl.UInt8).alias("_target_binary"))
    #     global_mean = train_df_with_target["_target_binary"].mean()

    #     for col_to_encode in categorical_cols_for_encoding:
    #         if col_to_encode in train_df.columns:
    #             # Calculate likelihoods on full training data (for applying to test & as simple train version)
    #             agg = train_df_with_target.group_by(col_to_encode).agg(
    #                 pl.col("_target_binary").mean().alias(f"{col_to_encode}_likelihood_mean"),
    #                 pl.col("_target_binary").count().alias(f"{col_to_encode}_likelihood_count")
    #             )
    #             agg = agg.with_columns(
    #                 ((pl.col(f"{col_to_encode}_likelihood_count") * pl.col(f"{col_to_encode}_likelihood_mean") + global_mean * alpha) / \
    #                  (pl.col(f"{col_to_encode}_likelihood_count") + alpha)).alias(f"{col_to_encode}_likelihood")
    #             )

    #             train_df = train_df.join(agg.select([col_to_encode, f"{col_to_encode}_likelihood"]), on=col_to_encode, how="left")
    #             train_df = train_df.with_columns(pl.col(f"{col_to_encode}_likelihood").fill_null(global_mean))

    #             if col_to_encode in test_df.columns:
    #                 test_df = test_df.join(agg.select([col_to_encode, f"{col_to_encode}_likelihood"]), on=col_to_encode, how="left")
    #                 test_df = test_df.with_columns(pl.col(f"{col_to_encode}_likelihood").fill_null(global_mean))

    #     if "_target_binary" in train_df.columns: # drop helper
    #        train_df = train_df.drop("_target_binary")

    return train_df, test_df


def feature_engineering(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    fill_means_prop: bool = True,
    fill_medians_prop: bool = False,  # Renamed for clarity
    add_mean_prop: bool = True,
    add_median_prop: bool = True,
    add_std_prop: bool = True,  # Renamed
    save_path: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Perform imputing & feature engineering on Expedia dataset."""
    logger.info("Starting feature engineering process.")

    # I. Initial Data Preparation & Target Variable Creation
    logger.info("Applying data types.")
    train_df = apply_dtypes(train_df, is_train=True)
    test_df = apply_dtypes(test_df, is_train=False)

    if "rating" not in train_df.columns:
        train_df = _create_target_variable(train_df)

    # II. Imputation of Missing Values
    train_df, test_df = _initial_imputations(train_df, test_df)

    prop_numeric_features = [  # Define features for prop-based imputation and aggregation
        "prop_starrating",
        "prop_review_score",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
    ]

    # For prop-specific imputations, use combined stats
    # Align columns before creating stats_source_df
    aligned_train_for_stats, aligned_test_for_stats = align_columns(
        train_df.clone(), test_df.clone()
    )  # Clone to avoid modifying original before all steps
    stats_source_df = pl.concat([aligned_train_for_stats, aligned_test_for_stats], how="diagonal")

    if fill_means_prop:
        logger.info("Filling missing property numeric features with property means.")
        train_df = mean_fill(train_df, prop_numeric_features, stats_df=stats_source_df)
        test_df = mean_fill(test_df, prop_numeric_features, stats_df=stats_source_df)

    if fill_medians_prop:
        logger.info("Filling missing property numeric features with property medians.")
        train_df = median_fill(train_df, prop_numeric_features, stats_df=stats_source_df)
        test_df = median_fill(test_df, prop_numeric_features, stats_df=stats_source_df)

    # III. Aggregated Features per prop_id
    train_df, test_df = _add_property_aggregated_features(
        train_df, test_df, prop_numeric_features, add_mean_prop, add_median_prop, add_std_prop
    )

    # IV. Likelihood Features (Target Encoding) - Placeholder
    if "rating" in train_df.columns:  # Only if target is present
        train_df, test_df = _engineer_likelihood_features(train_df, test_df, target_column="rating")

    # V. Date and Time Features
    train_df = _engineer_datetime_features(train_df)
    test_df = _engineer_datetime_features(test_df)

    # VI. Historical/Contextual User and Property Features
    train_df, test_df = _engineer_historical_contextual_features(train_df, test_df)

    # VII. Search Context Features
    train_df = _engineer_search_context_features(train_df)
    test_df = _engineer_search_context_features(test_df)

    # VIII. Competitor Interaction Features
    train_df = _engineer_competitor_features(train_df)
    test_df = _engineer_competitor_features(test_df)

    # IX. Interaction Features
    train_df = _engineer_interaction_features(train_df)
    test_df = _engineer_interaction_features(test_df)

    # X. Data Cleaning (Outlier removal) - Skipped as per plan's "Optional - to be reviewed"
    # train_df = data_cleaning(train_df) # If re-enabled, ensure it's done at appropriate stage

    # XI. Final Column Alignment
    logger.info("Aligning columns before saving.")
    train_df, test_df = align_columns(train_df, test_df)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        train_output_path = os.path.join(save_path, "train_processed.feather")
        test_output_path = os.path.join(save_path, "test_processed.feather")
        logger.info(f"Saving processed training data to {train_output_path}")
        train_df.write_ipc(train_output_path)
        logger.info(f"Saving processed test data to {test_output_path}")
        test_df.write_ipc(test_output_path)

    logger.info("Feature engineering process completed.")
    return train_df, test_df


def align_columns(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # This function might need to handle completely new columns created in one df but not other
    # The current logic ensures all columns from either df are present in both, filled with nulls
    # if missing, and casts to a common type if types differ.

    all_cols_set = set(df1.columns) | set(df2.columns)  # Using set for efficiency

    # Determine common schema, preferring more general types if conflicts
    final_schema = {}
    schema1 = df1.schema
    schema2 = df2.schema

    for col_name in all_cols_set:
        dtype1 = schema1.get(col_name)
        dtype2 = schema2.get(col_name)

        if dtype1 and dtype2:
            if dtype1 == dtype2:
                final_schema[col_name] = dtype1
            # Handle type conflicts, e.g. by upcasting or choosing one.
            # For simplicity, if numeric, prefer Float64. Otherwise, Utf8.
            # This could be made more sophisticated.
            elif dtype1.is_numeric() and dtype2.is_numeric():
                final_schema[col_name] = pl.Float64
            elif dtype1.is_temporal() and dtype2.is_temporal():  # Datetime/Date
                final_schema[col_name] = dtype1  # or choose larger if applicable
            else:  # Default to Utf8 for mixed non-numeric types or complex cases
                final_schema[col_name] = pl.String
        elif dtype1:
            final_schema[col_name] = dtype1
        elif dtype2:
            final_schema[col_name] = dtype2

    def conform_df(df: pl.DataFrame, target_schema: dict[str, pl.DataType]) -> pl.DataFrame:
        select_exprs = []
        for col_name, target_type in target_schema.items():
            if col_name in df.columns:
                if df[col_name].dtype != target_type:
                    select_exprs.append(pl.col(col_name).cast(target_type, strict=False).alias(col_name))
                else:
                    select_exprs.append(pl.col(col_name))
            else:  # Column is missing, add it as nulls of target type
                select_exprs.append(pl.lit(None, dtype=target_type).alias(col_name))
        return df.select(select_exprs)

    # Ensure final order is consistent, e.g., sorted alphabetically
    sorted_all_cols = sorted(list(all_cols_set))

    df1_aligned = conform_df(df1, final_schema)
    df2_aligned = conform_df(df2, final_schema)

    return df1_aligned.select(sorted_all_cols), df2_aligned.select(sorted_all_cols)


def data_cleaning(train_df):  # This function is currently not used in main flow
    logger.info("Performing data cleaning (outlier removal for price_usd).")
    # Remove outliers in price_usd
    q1 = train_df.select(pl.col("price_usd").quantile(0.25)).item()
    q3 = train_df.select(pl.col("price_usd").quantile(0.75)).item()
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr  # Also consider lower bound if appropriate

    original_rows = train_df.height
    train_df = train_df.filter((pl.col("price_usd") <= upper_bound) & (pl.col("price_usd") >= lower_bound))
    logger.info(f"Removed {original_rows - train_df.height} rows due to price_usd outliers.")

    # Fill small amount of values with 0 as 5% is 0 already (This was moved to _initial_imputations)
    # train_df = train_df.with_columns(pl.col("prop_review_score").fill_null(0))
    return train_df


def main():
    """Main data processing function"""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[3]
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"

    train_path = raw_dir / "training_set_VU_DM.csv"
    test_path = raw_dir / "test_set_VU_DM.csv"

    train_df, test_df = read_raw_data(train_path, test_path)

    # -- Feature engineering --
    # Parameters match the plan's intentions for default run
    train_df, test_df = feature_engineering(
        train_df,
        test_df,
        fill_means_prop=True,  # Corresponds to Plan II's Strategy 1 (mean fill for prop features)
        fill_medians_prop=False,  # Median fill is an option, but plan focuses on mean
        add_mean_prop=True,  # Corresponds to Plan III (add mean features by prop_id)
        add_median_prop=True,  # Corresponds to Plan III (add median features by prop_id)
        add_std_prop=True,  # Corresponds to Plan III (add std features by prop_id)
        save_path=str(processed_dir),  # Ensure path is string
    )

    logger.info("Data processing script completed.")


if __name__ == "__main__":
    main()
