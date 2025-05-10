import polars as pl

def feature_engineering(df, fill_means = True, fill_medians = False, add_mean = True, add_median = True, add_std = True):
    """
    Perform imputing & feature engineering on Expedia dataset: 
    
      IMPUTATION
    - Optionally fill missing values with per-property means
    - Optionally fill missing values with per-property medians

      FEATURE ENGINEERING
    - Optionally add per-property mean features
    - Optionally add per-property median features
    - Optionally add per-property standard deviation features
    """
    # get all prop collumns
    prop_columns = [col for col in df.columns if col.startswith("prop")]
    # exclude identifiers
    agg_columns = [col for col in prop_columns if col not in ["prop_id", "prop_country_id"]]

    # make sure all entries are floats
    df = clean_prop_location_score2(df)

    # imputation 
    if fill_means == True:
        df = mean_fill(df, agg_columns)
    if fill_medians == True:
        df = median_fill(df, agg_columns)

    # add features
    if add_mean == True:
        df = add_mean_features(df, agg_columns)
    if add_median == True:
        df = add_median_features(df, agg_columns)
    if add_std == True:
        df = add_std_features(df, agg_columns)

    return df

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

def add_mean_features(df, agg_columns):
    # compute mean per prop_id
    mean_df = df.group_by("prop_id").agg([
        pl.col(col).mean().alias(f"{col}_mean") for col in agg_columns
    ])

    # join mean features to df
    df = df.join(mean_df, on="prop_id", how="left")

    return df

def add_median_features(df, agg_columns):
    # compute median per prop_id
    median_df = df.group_by("prop_id").agg([
        pl.col(col).median().alias(f"{col}_median") for col in agg_columns
    ])

    # join median features to df
    df = df.join(median_df, on="prop_id", how="left")

    return df

def add_std_features(df, agg_columns):
    # compute standard deviation per prop_id 
    std_df = df.group_by("prop_id").agg([
        pl.col(col).std().alias(f"{col}_std") for col in agg_columns
    ])

    # join stddevs back to df
    df = df.join(std_df, on="prop_id", how="left")

    return df

def clean_prop_location_score2(df):
    return df.with_columns([
        pl.when(pl.col("prop_location_score2") == "NULL")
        .then(None)
        .otherwise(pl.col("prop_location_score2"))
        .alias("prop_location_score2")
    ]).with_columns([
        pl.col("prop_location_score2").cast(pl.Float64)
    ])


if __name__ == "__main__":
    # load in dataset
    df = pl.read_csv("../../data/training_set_VU_DM.csv")
    df = feature_engineering(df, fill_means = True, fill_medians = False, add_mean = True, add_median = True, add_std = True)
