import logging
import os

import lightgbm as lgb
import polars as pl
import xgboost as xgb

# Attempt to import from dmt.models.lambdaMART, adjust if paths differ
# Assuming dmt is in PYTHONPATH or accessible relative to execution
try:
    from dmt.models.lambdaMART import generate_submission, preprocess
except ImportError:
    # Fallback for cases where the module might be run directly and path issues arise
    # This might require users to adjust PYTHONPATH or run from a specific directory
    logger = logging.getLogger(__name__)
    logger.warning(
        "Could not import from dmt.models.lambdaMART. Ensure PYTHONPATH is set correctly or script is run from project root."
    )

    # Define dummy functions if import fails, to allow script structure to be checked
    # This is not ideal for actual execution.
    def generate_submission(df, pred_col, output_file_name, ndcg_score=None):
        print(f"Dummy generate_submission called for {output_file_name}")
        pass

    def preprocess(df):
        print("Dummy preprocess called")
        if "date_time" in df.columns:
            # A very basic version of preprocess if the real one isn't available
            df = df.with_columns(
                [
                    pl.col("date_time").dt.hour().alias("hour"),
                    pl.col("date_time").dt.weekday().alias("dow"),
                ]
            ).drop("date_time")
        return df


# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- File Paths and Constants ---
TRAIN_PROCESSED_PATH = "data/processed/train_processed.feather"
TEST_PROCESSED_PATH = "data/processed/test_processed.feather"
TARGET_COL = "rating"
GROUP_COL = "srch_id"

LGBM_MODEL_OUTPUT_PATH = "models/final_lgbm_ranker_model.txt"
XGB_MODEL_OUTPUT_PATH = "models/final_xgb_ranker_model.json"

# --- Best LGBM Params (from train-full-data rule) ---
BEST_LGBM_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.03568617151380954,
    "num_leaves": 88,
    "max_depth": 13,
    "min_child_samples": 28,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "reg_alpha": 9.891854416407897,
    "reg_lambda": 0.0014933652147104117,
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    # As per analysis, 'label_gain' is commented out in the rule's param dict, so not included.
    # 'y_train_full' will be the raw 'rating' column.
}

# --- Best XGBoost Params (from train-full-data rule) ---
BEST_XGB_PARAMS = {
    "n_estimators": 550,
    "learning_rate": 0.06412038072139861,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.7999999999999999,
    "colsample_bytree": 0.8999999999999999,
    "gamma": 0.131873411179656,
    "reg_lambda": 1.1051835714463383,
    "reg_alpha": 0.09014861689127424,
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@5",
    "booster": "gbtree",
    "tree_method": "hist",
    "random_state": 42,
}


def main():
    logger.info("Starting final model training and ensembling script.")

    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # --- Phase 1: Data Preparation for Final Training ---
    logger.info("Phase 1: Data Preparation for Final Training")

    # Load Full Training Data
    logger.info(f"Loading full training data from {TRAIN_PROCESSED_PATH}...")
    train_df_polars_raw = pl.read_ipc(TRAIN_PROCESSED_PATH)

    # Preprocess training data (e.g., handle date_time to create hour, dow)
    logger.info("Preprocessing training data...")
    train_df_polars_processed = preprocess(
        train_df_polars_raw.clone()
    )  # Use clone to avoid modifying original if preprocess is in-place on some operations

    # Define features_list
    logger.info("Defining feature list...")
    base_excluded_cols = [
        TARGET_COL,
        GROUP_COL,
        "prop_id",
        "click_bool",
        "booking_bool",
        "position",
        "gross_bookings_usd",  # Original name from dataset description
        "gross_booking_usd",  # Potential variation in naming from files (dataset desc has 'gross_booking_usd')
    ]
    # Ensure all potential excluded columns are considered, even if not present
    actual_excluded_cols = [col for col in base_excluded_cols if col in train_df_polars_processed.columns]

    features_list = [col for col in train_df_polars_processed.columns if col not in actual_excluded_cols]
    logger.info(f"Number of features: {len(features_list)}")
    logger.debug(f"Features: {features_list}")

    # Convert to Pandas DataFrame for model training
    logger.info("Converting training data to Pandas DataFrame...")
    train_pdf_full = train_df_polars_processed.to_pandas()

    # Prepare Training Inputs
    logger.info("Preparing training inputs (X, y, groups)...")
    X_train_full = train_pdf_full[features_list]
    y_train_full = train_pdf_full[TARGET_COL]  # Using raw 'rating' as per decision

    # Calculate group sizes. Ensure data is sorted by group_col if necessary, though groupby().size() is robust.
    # For safety, ensure correct group calculation corresponding to X_train_full and y_train_full order
    # If train_pdf_full is not sorted by GROUP_COL, group sizes might mismatch if models expect sorted group data.
    # However, scikit-learn wrappers typically handle this. groupby().size() itself is fine.
    groups_train_full = train_pdf_full.groupby(GROUP_COL, sort=False).size().to_numpy()

    # --- Phase 2: Train Final LGBMRanker Model ---
    logger.info("Phase 2: Train Final LGBMRanker Model")
    logger.info("Initializing LGBMRanker with best hyperparameters...")
    final_lgbm_model = lgb.LGBMRanker(**BEST_LGBM_PARAMS)

    logger.info("Training final LGBMRanker model on the entire training set...")
    final_lgbm_model.fit(X_train_full, y_train_full, group=groups_train_full)
    logger.info("LGBMRanker training complete.")

    logger.info(f"Saving final LGBM model to {LGBM_MODEL_OUTPUT_PATH}...")
    final_lgbm_model.booster_.save_model(LGBM_MODEL_OUTPUT_PATH)
    logger.info(f"Final LGBM model saved to {LGBM_MODEL_OUTPUT_PATH}")

    # --- Phase 3: Train Final XGBRanker Model ---
    logger.info("Phase 3: Train Final XGBRanker Model")
    logger.info("Initializing XGBRanker with best hyperparameters...")
    final_xgb_model = xgb.XGBRanker(**BEST_XGB_PARAMS)

    logger.info("Training final XGBRanker model on the entire training set...")
    final_xgb_model.fit(X_train_full, y_train_full, group=groups_train_full, verbose=False)
    logger.info("XGBRanker training complete.")

    logger.info(f"Saving final XGBoost model to {XGB_MODEL_OUTPUT_PATH}...")
    final_xgb_model.save_model(XGB_MODEL_OUTPUT_PATH)
    logger.info(f"Final XGBoost model saved to {XGB_MODEL_OUTPUT_PATH}")

    # --- Phase 4: Ensemble Predictions and Submission ---
    logger.info("Phase 4: Ensemble Predictions and Submission")

    # Load Processed Test Data
    logger.info(f"Loading processed test data from {TEST_PROCESSED_PATH}...")
    test_df_polars_raw = pl.read_ipc(TEST_PROCESSED_PATH)

    # Preprocess test data consistently with training data
    logger.info("Preprocessing test data...")
    test_df_polars_processed = preprocess(test_df_polars_raw.clone())

    # Convert to Pandas DataFrame and extract X_test_final
    logger.info("Converting test data to Pandas DataFrame and extracting features...")
    test_pdf = test_df_polars_processed.to_pandas()
    X_test_final = test_pdf[features_list]

    # Generate Predictions
    logger.info("Generating predictions from final LGBM model...")
    preds_lgbm_final = final_lgbm_model.predict(X_test_final)

    logger.info("Generating predictions from final XGBoost model...")
    preds_xgb_final = final_xgb_model.predict(X_test_final)

    # Blend Predictions
    logger.info("Blending predictions...")
    weight_lgbm = 0.5  # As per rule example
    weight_xgb = 0.5  # As per rule example
    blended_scores_final = (weight_lgbm * preds_lgbm_final) + (weight_xgb * preds_xgb_final)

    # Add to the Polars DataFrame for submission generation
    # Ensure the original Polars DataFrame for test is used for srch_id, prop_id to maintain order
    test_df_polars_for_submission = test_df_polars_processed.with_columns(
        pl.lit(blended_scores_final).alias("ensemble_predictions")
    )

    # Create Submission File
    submission_ndcg_estimate = 0.4037  # From rule (XGBoost Optuna output placeholder)
    output_submission_filename = f"submission_ensemble_lgbm_xgb_{submission_ndcg_estimate:.4f}.csv"

    logger.info(f"Generating submission file: {output_submission_filename}...")
    generate_submission(
        df=test_df_polars_for_submission,
        pred_col="ensemble_predictions",
        output_file=output_submission_filename,  # generate_submission will place it in 'submissions/'
        ndcg_score=submission_ndcg_estimate,
    )
    logger.info(f"Submission file generation process called for: {output_submission_filename}")
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    # This allows running the script directly, e.g., `python src/dmt/models/train_final_ensemble.py`
    # Ensure that the working directory is the project root for relative paths to work correctly.
    # If using `uv run train_final_ensemble.py`, uv might handle paths from project root as well.
    main()
