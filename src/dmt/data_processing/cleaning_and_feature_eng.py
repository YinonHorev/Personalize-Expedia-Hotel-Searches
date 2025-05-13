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

def feature_engineering(train_df, test_df, fill_means = True, fill_medians = False, add_mean = True, add_median = True, add_std = True, save_path = None):
    """Perform imputing & feature engineering on Expedia dataset:
    
      IMPUTATION
    - Optionally fill missing values with per-property means
    - Optionally fill missing values with per-property medians

      FEATURE ENGINEERING
    - Optionally add per-property mean features
    - Optionally add per-property median features
    - Optionally add per-property standard deviation features
    """
    # get all prop collumns
    prop_columns = [col for col in train_df.columns if col.startswith("prop")]
    print(prop_columns)
    # exclude identifiers
    agg_columns = [col for col in prop_columns if col not in ["prop_id", "prop_country_id"]]

    # make sure all entries are floats
    train_df = clean_prop_location_score2(train_df)

    # imputation 
    if fill_means == True:
        train_df = mean_fill(train_df, agg_columns)
    if fill_medians == True:
        train_df = median_fill(train_df, agg_columns)

    # add features
    if add_mean == True:
        train_df = add_mean_features(train_df, test_df, agg_columns)
    if add_median == True:
        train_df = add_median_features(train_df, test_df, agg_columns)
    if add_std == True:
        train_df = add_std_features(train_df, test_df, agg_columns)

    # Save processed data if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        train_path = os.path.join(save_path, "train_processed.feather")
        test_path = os.path.join(save_path, "test_processed.feather")

        logger.info(f"Saving processed training data to {train_path}")
        train_df.write_ipc(train_path)

        logger.info(f"Saving test data (unchanged) to {test_path}")
        test_df.write_ipc(test_path)

def mean_fill(df, agg_columns):
    ####################
    # print for checks #
    ####################
    # print(df.select("prop_location_score2").unique().sort("prop_location_score2").head(50)
    # )
    # print(df.group_by("prop_location_score2").len().sort("len", descending=True))
    # nulls_before = df.select([pl.col(c).is_null().sum().alias(c) for c in agg_columns])
    # means_before = df.select([pl.col(c).mean().alias(c) for c in agg_columns])

    # Compute mean per prop_id for each relevant column
    agg_df = df.group_by("prop_id").agg([
        pl.col(col).mean().alias(f"{col}_mean") for col in agg_columns
    ])

    # join means back to df
    df = df.join(agg_df, on="prop_id", how="left")

    # fill nulls with correct means
    for col in agg_columns:
        global_mean = df.select(pl.col(col).mean()).item()
        
        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(
                pl.when(pl.col(f"{col}_mean").is_not_null())
                .then(pl.col(f"{col}_mean"))
                .otherwise(global_mean)
            )
            .otherwise(pl.col(col))
            .alias(col)
        )

    # drop helper column
    df = df.drop([f"{col}_mean" for col in agg_columns])

    ####################
    # print for checks #
    ####################
    # nulls_after = df.select([pl.col(c).is_null().sum().alias(c) for c in agg_columns])
    # means_after = df.select([pl.col(c).mean().alias(c) for c in agg_columns])
    # print("Nulls before filling:")
    # print(nulls_before)
    # print("Nulls after filling:")
    # print(nulls_after)
    # print("Means before filling:")
    # print(means_before)
    # print("Means after filling:")
    # print(means_after)
    ###################
    
    return df

def median_fill(df, agg_columns):
    # compute medians per id 
    agg_df = df.group_by("prop_id").agg([
        pl.col(col).median().alias(f"{col}_median") for col in agg_columns
    ])

    # join medians to df  
    df = df.join(agg_df, on="prop_id", how="left")

    # fill nulls with medians    
    # fallback: global median    
    for col in agg_columns:
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

    # drop helper columns  
    df = df.drop([f"{col}_median" for col in agg_columns])
    
    return df

def add_mean_features(train_df, test_df, agg_columns):
    # align train and test
    train_df, test_df = align_columns(train_df, test_df)
    # compute mean per prop_id
    combined_df = pl.concat([train_df, test_df])
    mean_df = combined_df.group_by("prop_id").agg([
        pl.col(col).mean().alias(f"{col}_mean") for col in agg_columns
    ])

    # join mean features to df
    train_df = train_df.join(mean_df, on="prop_id", how="left")

    return train_df

def add_median_features(train_df, test_df, agg_columns):
    # align train and test
    train_df, test_df = align_columns(train_df, test_df)

    # compute median per prop_id
    combined_df = pl.concat([train_df, test_df])
    median_df = combined_df.group_by("prop_id").agg([
        pl.col(col).median().alias(f"{col}_median") for col in agg_columns
    ])

    # join median features to df
    train_df = train_df.join(median_df, on="prop_id", how="left")

    return train_df


def add_std_features(train_df, test_df, agg_columns):
    # align train and test
    train_df, test_df = align_columns(train_df, test_df)

    # compute standard deviation per prop_id 
    combined_df = pl.concat([train_df, test_df])
    std_df = combined_df.group_by("prop_id").agg([
        pl.col(col).std().alias(f"{col}_std") for col in agg_columns
    ])

    # join stddevs back to df
    train_df = train_df.join(std_df, on="prop_id", how="left")

    return train_df
    
def align_columns(df1: pl.DataFrame, df2: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    all_cols = sorted(set(df1.columns) | set(df2.columns))
    
    def is_numeric(dtype):
        return dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                         pl.Float32, pl.Float64)

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
        return df.select([
            (pl.col(c).cast(schema[c]) if c in df.columns else pl.lit(None).cast(schema[c])).alias(c)
            for c in all_cols
        ])

    return prepare(df1, new_schema), prepare(df2, new_schema)

def clean_prop_location_score2(df):
    return df.with_columns([
        pl.when(pl.col("prop_location_score2") == "NULL")
        .then(None)
        .otherwise(pl.col("prop_location_score2"))
        .alias("prop_location_score2")
    ]).with_columns([
        pl.col("prop_location_score2").cast(pl.Float64)
    ])

def data_cleaning(train_df):
    # remove outliers in price_usd
    quarter1 = train_df.select(pl.col("price_usd").quantile(0.25, "nearest")).item()
    quarter3 = train_df.select(pl.col("price_usd").quantile(0.25, "nearest")).item()

    iqr = quarter3 - quarter1

    upper_bound = quarter3 + 1.5 * iqr 

    train_df = train_df.filter(pl.col("price_usd") <= upper_bound)

    # fill small amount of values with 0 as 5% is 0 already
    train_df = train_df.with_columns(
        pl.col("prop_review_score").fill_null(0)
    )

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
    train_df = data_cleaning(train_df)

    # -- Feature engineering -- 
    feature_engineering(train_df, test_df, fill_means = True, fill_medians = False, add_mean = True, add_median = True, add_std = True, save_path=processed_dir)

    logger.info("Data processing completed.")

if __name__ == "__main__":
    main()

