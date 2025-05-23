import logging
import os
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

# Assuming generate_submission is in dmt.utils.helpers or dmt.models.lambdaMART
# We will try to import it from lambdaMART as per the search results.
# If it's in helpers, this might need adjustment.
from dmt.models.lambdaMART import generate_submission  # Or adjust path as needed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Define globals (features_list, target_col, group_col) ---
TRAIN_DF_PATH = "data/processed/train_processed.feather"
TEST_DF_PATH = "data/processed/test_processed.feather"
TARGET_COL = "rating"  # As per dataset description for training
GROUP_COL = "srch_id"

# Define features_list globally after loading data
features_list = []
train_df_full_pd = None  # Will be loaded once


def load_and_prepare_data():
    """Loads and prepares data, defines global features_list and train_df_full_pd."""
    global features_list, train_df_full_pd

    logger.info(f"Loading training data from {TRAIN_DF_PATH}")
    train_df_polars = pl.read_ipc(TRAIN_DF_PATH)
    train_df_full_pd = train_df_polars.to_pandas()  # Convert to Pandas

    # Define features_list, excluding IDs, target, etc.
    # These are typical columns to exclude. Adjust if your 'rating' or other target-like cols are named differently.
    excluded_cols = [
        TARGET_COL,
        GROUP_COL,
        "prop_id",
        "click_bool",
        "booking_bool",
        "position",
        "gross_bookings_usd",
        "date_time",
        # Columns that might be derived from date_time and already handled if present
        # "hour",
        # "dow",
    ]
    # Filter out excluded columns that might not exist to prevent errors
    potential_cols = train_df_polars.columns
    features_list = [col for col in potential_cols if col not in excluded_cols]

    # Ensure all features are numeric and handle NaNs if necessary (XGBoost can handle NaNs internally)
    # Forcing dtypes or more complex imputation could be added here if needed.
    logger.info(f"Features for training: {features_list}")


def objective_xgb(trial):
    """Optuna objective function for XGBoost Ranker."""
    # Ensure data is loaded
    if train_df_full_pd is None or not features_list:
        logger.error("Training data not loaded. Call load_and_prepare_data() first.")
        raise ValueError("Training data not loaded.")

    logger.info(f"Starting Trial {trial.number}...")

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0, step=0.1),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
        "random_state": 42,
        "tree_method": "hist",
    }

    gkf = GroupKFold(n_splits=3)  # Reduced splits for faster tuning
    ndcg_scores_fold = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(train_df_full_pd, train_df_full_pd[TARGET_COL], groups=train_df_full_pd[GROUP_COL])
    ):
        logger.info(f"Trial {trial.number}, Fold {fold + 1}/{gkf.get_n_splits()}: Preparing data...")
        X_trn_fold, y_trn_fold = (
            train_df_full_pd.loc[train_idx, features_list],
            train_df_full_pd.loc[train_idx, TARGET_COL],
        )
        X_val_fold, y_val_fold = train_df_full_pd.loc[val_idx, features_list], train_df_full_pd.loc[val_idx, TARGET_COL]

        # Calculate group sizes for XGBRanker
        # Ensure train_df_full_pd is sorted by GROUP_COL if not done globally for safety,
        # though GroupKFold should maintain group integrity.
        # groups are calculated on the subset of data for the fold.
        grp_trn_fold = train_df_full_pd.loc[train_idx].groupby(GROUP_COL).size().to_numpy()
        grp_val_fold = train_df_full_pd.loc[val_idx].groupby(GROUP_COL).size().to_numpy()

        model_xgb = xgb.XGBRanker(**params, early_stopping_rounds=30)
        logger.info(f"Trial {trial.number}, Fold {fold + 1}: Fitting XGBRanker with params: {params}")
        model_xgb.fit(
            X_trn_fold,
            y_trn_fold,
            group=grp_trn_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_group=[grp_val_fold],
            verbose=True,  # Changed from False to True for XGBoost internal logging
        )
        logger.info(f"Trial {trial.number}, Fold {fold + 1}: XGBRanker fitting complete.")

        predict_start_time = time.time()
        preds_val_fold = model_xgb.predict(X_val_fold)
        predict_duration = time.time() - predict_start_time
        logger.info(
            f"Trial {trial.number}, Fold {fold + 1}: Prediction on X_val_fold (shape {X_val_fold.shape}) took {predict_duration:.2f} seconds."
        )

        ndcg_calc_start_time = time.time()

        # Prepare data for groupby
        # y_val_fold is a Pandas Series, preds_val_fold is a NumPy array
        # Group identifiers are from train_df_full_pd.loc[val_idx, GROUP_COL]
        df_val_fold_eval = pd.DataFrame(
            {
                "y_true": y_val_fold.to_numpy(),
                "y_pred": preds_val_fold,
                GROUP_COL: train_df_full_pd.loc[val_idx, GROUP_COL].to_numpy(),
            }
        )

        num_unique_val_groups = df_val_fold_eval[GROUP_COL].nunique()
        logger.info(
            f"Trial {trial.number}, Fold {fold + 1}: Calculating NDCG for {num_unique_val_groups} unique groups in validation set using groupby.apply."
        )

        def ndcg_per_group(df_group):
            true_scores = df_group["y_true"].to_numpy()
            pred_scores = df_group["y_pred"].to_numpy()
            if np.sum(true_scores) == 0:  # All true labels are 0, NDCG is undefined or 0
                return np.nan  # np.nanmean will ignore this
            return ndcg_score([true_scores], [pred_scores], k=5)

        # Calculate NDCG for all groups using groupby.apply
        all_group_ndcgs = df_val_fold_eval.groupby(GROUP_COL).apply(ndcg_per_group)

        # Average the NDCG scores
        fold_ndcg_mean = np.nanmean(all_group_ndcgs.to_numpy())

        if np.isnan(fold_ndcg_mean):  # Handle case where all groups might have been NaN
            logger.warning(
                f"Fold {fold + 1} resulted in NaN average NDCG (all groups may have had no relevant items or were empty). Appending NDCG of 0."
            )
            ndcg_scores_fold.append(0.0)
        else:
            ndcg_scores_fold.append(fold_ndcg_mean)

        ndcg_calc_duration = time.time() - ndcg_calc_start_time
        logger.info(
            f"Trial {trial.number}, Fold {fold + 1}: Grouped NDCG calculation took {ndcg_calc_duration:.2f} seconds. Avg NDCG for fold: {fold_ndcg_mean:.4f}"
        )

    avg_ndcg = np.mean(ndcg_scores_fold)  # This remains the average over folds
    logger.info(
        f"Finished Trial {trial.number} - Avg NDCG@5 over {gkf.get_n_splits()} folds: {avg_ndcg:.4f} with final params: {trial.params}"
    )
    return avg_ndcg


def tune_xgboost():
    """Runs the Optuna study for XGBoost Ranker."""
    load_and_prepare_data()  # Ensure data is loaded

    study_name_xgb = "xgbranker-tuning-m3-v1"  # As per guide
    # Ensure the directory for the database exists if it's not project root
    os.makedirs("models", exist_ok=True)  # Assuming study db goes into models or project root
    storage_name_xgb = f"sqlite:///models/{study_name_xgb}.db"  # Store in models directory

    study_xgb = optuna.create_study(
        study_name=study_name_xgb, storage=storage_name_xgb, load_if_exists=True, direction="maximize"
    )

    # Log information about the existing study
    logger.info(f"Study '{study_name_xgb}' details:")
    logger.info(f"  Number of trials in DB: {len(study_xgb.trials)}")
    completed_trials = [t for t in study_xgb.trials if t.state == optuna.trial.TrialState.COMPLETE]
    running_trials = [t for t in study_xgb.trials if t.state == optuna.trial.TrialState.RUNNING]
    failed_trials = [t for t in study_xgb.trials if t.state == optuna.trial.TrialState.FAIL]
    logger.info(f"  Completed trials: {len(completed_trials)}")
    if completed_trials:
        logger.info(f"  Best trial so far: {study_xgb.best_trial.number} with value {study_xgb.best_trial.value:.4f}")
        logger.info(f"    Params: {study_xgb.best_trial.params}")
    logger.info(f"  Currently running trials (in DB): {len(running_trials)}")
    logger.info(f"  Failed trials (in DB): {len(failed_trials)}")

    logger.info(f"Starting/Continuing Optuna optimization for {study_name_xgb} with n_trials={15}.")
    study_xgb.optimize(objective_xgb, n_trials=15)

    logger.info("Best XGBoost trial from current Optuna run:")
    best_trial_xgb = study_xgb.best_trial
    logger.info(f"  Value (NDCG@5): {best_trial_xgb.value:.4f}")
    logger.info("  Params: ")
    for key, value in best_trial_xgb.params.items():
        logger.info(f"    {key}: {value}")

    return best_trial_xgb.params


def train_final_xgboost_model(best_xgb_params):
    """Trains the final XGBoost model on the full training data and saves it."""
    if train_df_full_pd is None or not features_list:
        logger.error("Training data not loaded for final model training.")
        raise ValueError("Training data not loaded.")

    X_train_full = train_df_full_pd[features_list]
    y_train_full = train_df_full_pd[TARGET_COL]
    groups_train_full = train_df_full_pd.groupby(GROUP_COL).size().to_numpy()

    # Ensure n_estimators is in the params for the final model
    # Optuna might not always include it if it was part of the objective's direct definition
    # but here it's part of `params` dict passed to XGBRanker constructor
    final_xgb_model = xgb.XGBRanker(**best_xgb_params)

    logger.info("Fitting final XGBoost model...")
    final_xgb_model.fit(X_train_full, y_train_full, group=groups_train_full, verbose=True)

    model_xgb_path = "models/xgb_ranker_model.json"
    os.makedirs("models", exist_ok=True)
    final_xgb_model.save_model(model_xgb_path)
    logger.info(f"Final XGBoost model saved to {model_xgb_path}")
    return model_xgb_path, final_xgb_model


def ensemble_and_predict(xgb_model_path_or_obj, lgbm_model_path):
    """Loads models, generates predictions, blends them, and creates a submission file."""
    global features_list  # Ensure features_list is available

    if not features_list:  # If tune_xgboost wasn't run in the same session, features_list might be empty
        logger.info("features_list is empty. Attempting to load data to define it.")
        load_and_prepare_data()  # This will define features_list from train data.

    logger.info(f"Loading original test data from {TEST_DF_PATH}")
    test_df_polars_orig = pl.read_ipc(TEST_DF_PATH)

    # --- XGBoost Prediction ---
    # features_list is derived from train_processed.feather columns minus excluded_cols.
    # It should not contain original 'date_time', 'hour', or 'dow'.
    logger.info(f"Preparing test data for XGBoost using features_list ({len(features_list)} features).")
    # Ensure all features in features_list are present in the loaded test data
    missing_xgb_feats = [f for f in features_list if f not in test_df_polars_orig.columns]
    if missing_xgb_feats:
        logger.error(f"Missing features for XGBoost in loaded test data: {missing_xgb_feats}")
        raise ValueError(f"Cannot prepare data for XGBoost, missing features: {missing_xgb_feats}")

    X_test_final_xgb = test_df_polars_orig.select(features_list).to_pandas()
    logger.info(f"Shape of X_test_final_xgb for XGBoost: {X_test_final_xgb.shape}")

    if isinstance(xgb_model_path_or_obj, str):
        logger.info(f"Loading XGBoost model from {xgb_model_path_or_obj}")
        xgb_model = xgb.XGBRanker()
        xgb_model.load_model(xgb_model_path_or_obj)
    else:
        xgb_model = xgb_model_path_or_obj
        logger.info("Using provided XGBoost model object.")

    logger.info("Generating predictions from XGBoost model...")
    preds_xgb = xgb_model.predict(X_test_final_xgb)

    # --- LightGBM Prediction ---
    logger.info("Preparing test data for LightGBM.")
    test_df_polars_lgbm = test_df_polars_orig.clone()

    if "date_time" in test_df_polars_lgbm.columns:
        logger.info("Creating 'hour' and 'dow' columns from 'date_time' for LightGBM.")
        test_df_polars_lgbm = test_df_polars_lgbm.with_columns(
            [pl.col("date_time").dt.hour().alias("hour"), pl.col("date_time").dt.weekday().alias("dow")]
        )
    else:
        logger.error(
            "'date_time' column not found in loaded test data. This is required to create 'hour' and 'dow' features for the LightGBM model as done in lambdaMART.py."
        )
        raise ValueError(
            "'date_time' column missing from test_processed.feather, cannot ensure LightGBM feature compatibility."
        )

    # Define the feature set for LightGBM: features_list + "hour" + "dow"
    # This assumes features_list (126 features) + "hour" + "dow" = 128 features for LGBM
    features_lgbm = features_list + ["hour", "dow"]

    # Sanity check for LightGBM features
    missing_lgbm_features = [f for f in features_lgbm if f not in test_df_polars_lgbm.columns]
    if missing_lgbm_features:
        logger.error(f"Features missing for LightGBM after attempting to create 'hour'/'dow': {missing_lgbm_features}")
        raise ValueError(f"Cannot form feature set for LightGBM. Missing: {missing_lgbm_features}")

    X_test_final_lgbm = test_df_polars_lgbm.select(features_lgbm).to_pandas()
    logger.info(f"Shape of X_test_final_lgbm for LightGBM: {X_test_final_lgbm.shape}")

    logger.info(f"Loading LGBM model from {lgbm_model_path}")
    try:
        lgbm_model = lgb.Booster(model_file=lgbm_model_path)
    except Exception as e:
        logger.error(f"Failed to load LGBM Booster model: {e}. Ensure it's a Booster model file.")
        raise

    logger.info("Generating predictions from LGBM model...")
    # Note: If LightGBM still complains about feature count, predict_disable_shape_check=True can be used
    # but it's better to ensure the feature set is perfectly matched.
    preds_lgbm = lgbm_model.predict(X_test_final_lgbm)

    # Blend predictions
    weight_lgbm = 0.6
    weight_xgb = 0.4
    blended_scores = (weight_lgbm * preds_lgbm) + (weight_xgb * preds_xgb)

    # Add blended scores to the Polars DataFrame for submission
    # Ensure that test_df_polars has the srch_id and prop_id for submission
    # The test_pdf was created with these, so we can use its index to align if needed,
    # but it's safer to add to the original polars df after ensuring row order.
    # For simplicity, assuming the order of X_test_final maps directly to test_df_polars.

    # Reconstruct the submission DataFrame with srch_id, prop_id, and blended_predictions
    # This needs to be careful about row order.
    # It's safer to use the original test_df_polars and add the column

    submission_df_prep = test_df_polars_orig.select(["srch_id", "prop_id"])
    submission_df_prep = submission_df_prep.with_columns(pl.lit(blended_scores).alias("blended_predictions"))

    # Retrieve the NDCG score from the XGBoost study for the filename
    # Fallback if study object is not available (e.g. if running prediction standalone)
    try:
        study_xgb_check = optuna.load_study(
            study_name="xgbranker-tuning-m3-v1", storage="sqlite:///models/xgbranker-tuning-m3-v1.db"
        )
        ndcg_for_filename = study_xgb_check.best_trial.value
    except Exception:
        logger.warning("Could not load study to get best NDCG for filename. Using placeholder 0.0")
        ndcg_for_filename = 0.0  # Placeholder

    # Generate submission file using the imported function
    logger.info("Generating submission file...")
    submission_file_path = generate_submission(
        submission_df_prep,  # DataFrame should contain srch_id, prop_id, and the prediction column
        pred_col="blended_predictions",
        # output_file can be None to auto-generate, or specify a name
        ndcg_score=ndcg_for_filename,  # Use a relevant NDCG score for filename
    )
    logger.info(f"Ensemble submission file generated at {submission_file_path}")


def main():
    # Phase 1: XGBoost Ranker Tuning and Training
    logger.info("--- Phase 1: XGBoost Ranker Tuning and Training ---")
    best_xgb_params_from_tuning = tune_xgboost()
    # Forcing dummy params to skip long tuning for now
    # logger.warning("Skipping Optuna tuning for XGBoost and using placeholder parameters for demonstration.")
    # logger.warning(
    #     "For a real run, uncomment 'best_xgb_params_from_tuning = tune_xgboost()' and ensure 'n_trials' is adequate."
    # )

    # Placeholder best_xgb_params if skipping tuning
    # These are just example values, replace with actual tuned ones or load from a previous study.
    # best_xgb_params_from_tuning = {
    #     "objective": "rank:ndcg",
    #     "eval_metric": "ndcg@5",
    #     "booster": "gbtree",
    #     "n_estimators": 100,  # Low for fast demo
    #     "learning_rate": 0.1,
    #     "max_depth": 5,
    #     "min_child_weight": 1,
    #     "subsample": 0.8,
    #     "colsample_bytree": 0.8,
    #     "gamma": 0.1,
    #     "reg_lambda": 1.0,
    #     "reg_alpha": 0.1,
    #     "random_state": 42,
    #     "tree_method": "hist",
    # }

    load_and_prepare_data()  # Ensure data is loaded before training
    final_xgb_model_path, _ = train_final_xgboost_model(best_xgb_params_from_tuning)

    # Phase 2: Ensembling with Pre-trained LGBMRanker
    logger.info("--- Phase 2: Ensembling XGBoost with LGBMRanker ---")
    # IMPORTANT: Replace with your actual LGBM model path
    lgbm_model_path = "models/lambdamart_model_20250518_1214.txt"  # Updated to latest model provided
    # Or "models/lambdamart_model_YOUR_TIMESTAMP.txt" from guide

    if not os.path.exists(lgbm_model_path):
        logger.error(f"LGBM model not found at {lgbm_model_path}. Please provide a valid path.")
        logger.error("Halting before ensembling. You might need to train an LGBM model first or correct the path.")
        return

    ensemble_and_predict(final_xgb_model_path, lgbm_model_path)

    logger.info("XGBoost tuning and ensembling process completed.")


if __name__ == "__main__":
    main()
