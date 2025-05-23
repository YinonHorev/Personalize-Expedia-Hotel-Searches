import datetime
import json
import logging
import os

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "rating"
GROUP_COL = "srch_id"
# Based on the dataset description, 'rating' can be 0, 1, or 5.
# Ensure these are sorted if you use label_gain: e.g., [0, 1, 5]
# if your actual unique values are sorted like that.
# Example: unique_ratings = sorted(train_df[TARGET_COL].unique().to_list())
# label_gain = unique_ratings
# For this dataset, it's [0, 1, 5]
LABEL_GAIN = [0, 1, 5]
RATING_MAP = {0: 0, 1: 1, 5: 2}  # Maps original ratings to 0-indexed contiguous integers


def preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """Minimal date features, similar to lambdaMART.py."""
    df = df.with_columns([pl.col("date_time").dt.hour().alias("hour"), pl.col("date_time").dt.weekday().alias("dow")])
    return df.drop("date_time")


def load_data_and_define_features(data_path="data/processed/train_processed.feather"):
    """Loads training data and defines feature columns."""
    logger.info(f"Loading training data from {data_path}...")
    train_df = pl.read_ipc(data_path)
    train_df = preprocess_data(train_df)

    # Define features_list: Exclude IDs, target, and other non-feature columns
    excluded_cols = [
        TARGET_COL,
        GROUP_COL,
        "prop_id",
        "click_bool",
        "booking_bool",
        "position",
        "gross_bookings_usd",  # "gross_bookings_usd" might be a typo in guidance, using "gross_booking_usd"
    ]
    # Correcting potential typo from guidance for "gross_bookings_usd"
    if "gross_bookings_usd" in train_df.columns and "gross_booking_usd" not in train_df.columns:
        pass  # Keep as is
    elif "gross_booking_usd" in train_df.columns and "gross_bookings_usd" not in excluded_cols:
        excluded_cols.append("gross_booking_usd")

    features_list = [
        col for col in train_df.columns if col not in excluded_cols and col != "date_time"
    ]  # date_time is dropped by preprocess
    logger.info(f"Number of features: {len(features_list)}")
    logger.debug(f"Features: {features_list}")

    return train_df, features_list


def objective(trial, train_df: pl.DataFrame, features_list: list[str], target_col: str, group_col: str):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "label_gain": LABEL_GAIN,  # Using the defined label_gain
    }

    gkf = GroupKFold(n_splits=5)
    fold_ndcg_scores = []

    # Using .to_pandas() once outside the loop if memory allows, as per guidance.
    # If dataset is too large, move this inside the loop for each fold.
    train_pdf = train_df.to_pandas()

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_pdf, train_pdf[target_col], groups=train_pdf[group_col])
    ):
        logger.debug(f"Starting Fold {fold + 1}/5")
        X_trn_pd, y_trn_pd = train_pdf.loc[train_idx, features_list], train_pdf.loc[train_idx, target_col]
        X_val_pd, y_val_pd = train_pdf.loc[val_idx, features_list], train_pdf.loc[val_idx, target_col]

        # Map ratings to 0, 1, 2 for LightGBM
        y_trn_mapped_pd = y_trn_pd.map(RATING_MAP).astype(int)
        y_val_mapped_pd = y_val_pd.map(RATING_MAP).astype(int)

        # Calculate group sizes for LGBMRanker
        # Robust way: group by group_col and get size()
        grp_trn = train_pdf.iloc[train_idx].groupby(group_col).size().to_numpy()
        grp_val = train_pdf.iloc[val_idx].groupby(group_col).size().to_numpy()

        model = lgb.LGBMRanker(**params)
        model.fit(
            X_trn_pd,
            y_trn_mapped_pd,  # Use mapped labels for training
            group=grp_trn,
            eval_set=[(X_val_pd, y_val_mapped_pd)],  # Use mapped labels for eval set
            eval_group=[grp_val],
            eval_at=[5],  # for NDCG@5
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],  # Corrected verbose
        )

        preds_val = model.predict(X_val_pd)

        # Calculate NDCG@5 for the current fold (group-wise)
        current_fold_group_ndcg_scores = []
        # Get unique groups in the validation set for this fold
        # Ensure we're using the correct group IDs from the validation set indices
        val_groups_pd = train_pdf.iloc[val_idx]
        unique_val_groups = val_groups_pd[group_col].unique()

        for search_id_val in unique_val_groups:
            current_search_mask_val = val_groups_pd[group_col] == search_id_val
            y_true_group = y_val_pd[
                current_search_mask_val
            ].to_numpy()  # y_val_pd is already pandas Series with original ratings
            y_pred_group = preds_val[
                current_search_mask_val.to_numpy(dtype=bool)
            ]  # preds_val is numpy, mask needs to be numpy bool

            if np.sum(y_true_group) == 0:  # Skip if no relevant items in this group
                continue

            # Reshape for ndcg_score: (n_samples, n_labels) -> (1, n_labels_in_group)
            # Ensure k is not greater than the number of items in the group
            k_val = min(5, len(y_true_group))
            if k_val > 0:
                current_fold_group_ndcg_scores.append(ndcg_score([y_true_group], [y_pred_group], k=k_val))

        if current_fold_group_ndcg_scores:
            fold_ndcg_scores.append(np.mean(current_fold_group_ndcg_scores))
        else:  # Handle case where a fold might have no valid groups with relevant items
            fold_ndcg_scores.append(0.0)
            logger.warning(f"Fold {fold + 1} had no groups with relevant items for NDCG calculation.")

        logger.debug(f"Fold {fold + 1} NDCG@5: {fold_ndcg_scores[-1]:.4f}")

    avg_ndcg = np.mean(fold_ndcg_scores) if fold_ndcg_scores else 0.0
    logger.info(f"Trial {trial.number} - Avg NDCG@5: {avg_ndcg:.4f} with params: {trial.params}")
    return avg_ndcg


def main():
    logger.info("Starting hyperparameter tuning for LGBMRanker...")
    train_df, features_list = load_data_and_define_features()

    # Ensure the 'models' directory exists for saving parameters
    os.makedirs("models", exist_ok=True)
    # Ensure the 'submissions' directory (or wherever study.db is stored) exists
    # Optuna will create the .db file, but parent dir might be needed if nested
    # For sqlite:///study_name.db, it creates in the current working directory (project root).

    study_name = "lgbmranker-tuning-expedia-v1"
    storage_name = f"sqlite:///{study_name}.db"

    logger.info(f"Creating/loading Optuna study: {study_name} with storage: {storage_name}")
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")

    # Wrap objective to pass additional arguments
    # Using a lambda function for this purpose
    func = lambda trial: objective(trial, train_df, features_list, TARGET_COL, GROUP_COL)

    n_trials = 100  # As per guidance, adjust as needed
    logger.info(f"Optimizing with Optuna for {n_trials} trials...")
    study.optimize(func, n_trials=n_trials, timeout=None)  # Adjust timeout if needed, e.g., 600 seconds for 10 mins

    logger.info("Hyperparameter tuning completed.")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"  Value (NDCG@5): {best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best parameters to a JSON file
    best_params_file = f"models/best_lgbm_params_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(best_params_file, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    logger.info(f"Best parameters saved to {best_params_file}")

    # Output all trials to a CSV for detailed analysis if desired
    try:
        trials_df = study.trials_dataframe()
        trials_csv_path = f"reports/tuning_trials_{study_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs("reports", exist_ok=True)
        trials_df.to_csv(trials_csv_path, index=False)
        logger.info(f"All trials saved to {trials_csv_path}")
    except Exception as e:
        logger.error(f"Could not save trials dataframe: {e}")


if __name__ == "__main__":
    main()
