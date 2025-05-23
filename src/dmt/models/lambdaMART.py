import datetime
import logging
import os

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

from dmt.utils.helpers import evaluate_model_ndcg, log_evaluation_summary

logger = logging.getLogger(__name__)

# Add the rating map
RATING_MAP = {0: 0, 1: 1, 5: 2}
LABEL_GAIN = [0, 1, 5]


def generate_submission(df, pred_col, output_file=None, ndcg_score=None):
    sub = df.select(["srch_id", "prop_id", pred_col]).clone()

    logger.info("Sorting results by prediction score...")
    # Sort by srch_id and predictions (descending)
    sub = sub.sort(by=["srch_id", pred_col], descending=[False, True])

    # Create the final submission without the prediction column
    final_sub = sub.select(["srch_id", "prop_id"])

    # Generate filename with timestamp if not provided
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        score_suffix = f"_{ndcg_score:.3f}" if ndcg_score is not None else ""
        output_file = f"submissions/submission_{timestamp}{score_suffix}.csv"
    else:
        # Ensure path is in submissions directory
        output_file = os.path.join("submissions", os.path.basename(output_file))

    # Create submissions directory if it doesn't exist
    os.makedirs("submissions", exist_ok=True)

    # Save to CSV
    final_sub.write_csv(output_file, include_header=True)
    logger.info(f"Submission saved to {output_file}")
    return final_sub


def preprocess(df):
    # minimal date features
    df = df.with_columns([pl.col("date_time").dt.hour().alias("hour"), pl.col("date_time").dt.weekday().alias("dow")])
    return df.drop("date_time")


def baseline_ndcg(df, sort_by, ascending=True, k=5):
    scores = []
    for search_id in df["srch_id"].unique():
        grp = df.filter(pl.col("srch_id") == search_id)
        y_true = grp["rating"].to_numpy()

        if sum(y_true) == 0:
            continue

        # sort by the baseline column (e.g. price or rating)
        y_score = -grp[sort_by].to_numpy() if not ascending else grp[sort_by].to_numpy()
        score = ndcg_score([y_true], [y_score], k=k)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def evaluate_ndcg(model, df, features, k=5):
    """Evaluate model using NDCG@k metric."""
    # Make predictions
    df = df.with_columns(pl.lit(model.predict(df.select(features).to_pandas())).alias("pred"))

    # Use scikit-learn's ndcg_score for consistency with training
    scores = []
    for search_id in df["srch_id"].unique():
        grp = df.filter(pl.col("srch_id") == search_id)
        y_true = grp["rating"].to_numpy()
        y_score = grp["pred"].to_numpy()

        # skip searches with no positive labels
        if sum(y_true) == 0:
            continue

        # ndcg_score expects 2D arrays: shape (1, n_candidates)
        score = ndcg_score(
            [y_true],  # outer list makes it shape (1, N)
            [y_score],  # same here
            k=k,
        )
        scores.append(score)

    # return the average across all searches
    return float(np.mean(scores)) if scores else 0.0


def evaluate_with_helpers(model, df, features, k=5):
    """Evaluate using the helper module's implementation of NDCG."""
    # Make predictions
    df_with_pred = df.with_columns(pl.lit(model.predict(df.select(features).to_pandas())).alias("pred"))

    # Convert to pandas for helpers evaluation
    df_pd = df_with_pred.to_pandas()

    # Prepare the dataframes expected by evaluate_model_ndcg
    predictions_df = df_pd[["srch_id", "prop_id", "pred"]].sort_values(["srch_id", "pred"], ascending=[True, False])
    ground_truth_df = df_pd[["srch_id", "prop_id", "rating"]].rename(columns={"rating": "rating"})

    # Calculate NDCG using helpers
    ndcg = evaluate_model_ndcg(predictions_df, ground_truth_df, k=k)

    return ndcg


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 1) Load full data
    logger.info("Loading training data...")
    df = pl.read_ipc("data/processed/train_processed.feather")

    # 2) Preprocess
    df = preprocess(df)

    # 3) Split into grouped train/val
    # Convert to pandas for GroupKFold
    df_pd = df.to_pandas()
    gkf = GroupKFold(n_splits=5)
    # just take the first split
    train_idx, val_idx = next(gkf.split(df_pd, groups=df_pd["srch_id"]))

    # Convert back to polars
    trn = pl.from_pandas(df_pd.iloc[train_idx].reset_index(drop=True))
    val = pl.from_pandas(df_pd.iloc[val_idx].reset_index(drop=True))

    # 4) Feature / target definition
    target = "rating"
    group_id = "srch_id"
    drop_cols = [group_id, "prop_id", "click_bool", "booking_bool", "position", "gross_bookings_usd", target]
    features = [c for c in trn.columns if c not in drop_cols]

    # Convert to pandas for LGBMRanker
    X_trn = trn.select(features).to_pandas()
    y_trn_original = trn.select(target).to_pandas().iloc[:, 0]
    grp_trn = trn.group_by(group_id).agg(pl.count()).select(pl.col("count")).to_numpy().flatten()

    X_val = val.select(features).to_pandas()
    y_val_original = val.select(target).to_pandas().iloc[:, 0]
    grp_val = val.group_by(group_id).agg(pl.count()).select(pl.col("count")).to_numpy().flatten()

    # Map ratings for LightGBM training as done during tuning
    y_trn_mapped = y_trn_original.map(RATING_MAP).astype(int)
    y_val_mapped = y_val_original.map(RATING_MAP).astype(int)

    # 5) Train on train‐fold
    logger.info("Training LambdaMART model with best parameters...")
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",  # LightGBM's internal NDCG will use label_gain with mapped labels
        boosting_type="gbdt",
        # Best parameters from Optuna trial 5:
        n_estimators=700,
        learning_rate=0.03568617151380954,
        num_leaves=88,
        max_depth=13,
        min_child_samples=28,  # Optuna param, LGBM alias for min_data_in_leaf
        subsample=0.7,
        colsample_bytree=0.8,
        reg_alpha=9.891854416407897,
        reg_lambda=0.0014933652147104117,
        label_gain=LABEL_GAIN,  # Crucial for consistency with tuning
        random_state=42,  # Keep for reproducibility
        n_jobs=-1,  # Use all available cores
        verbose=-1,
    )

    model.fit(
        X_trn,
        y_trn_mapped,  # Use mapped labels for training
        group=grp_trn,
        eval_set=[(X_val, y_val_mapped)],  # Use mapped labels for LightGBM's eval_set
        eval_group=[grp_val],
        callbacks=[lgb.log_evaluation(period=50), lgb.early_stopping(stopping_rounds=100)],
    )

    # 6) Evaluate on val‐fold
    # IMPORTANT: The evaluation functions evaluate_ndcg and evaluate_with_helpers
    # should use the ORIGINAL ratings from 'val' DataFrame for y_true.
    # The existing evaluate_ndcg and evaluate_with_helpers already select "rating" from the Polars DataFrame 'val',
    # which holds original ratings, so they should be correct.
    logger.info("Evaluating model...")
    ndcg5 = evaluate_ndcg(model, val, features, k=5)  # val DataFrame has original ratings
    logger.info(f"Validation NDCG@5 (model): {ndcg5:.4f}")

    # Also evaluate using the helpers implementation
    ndcg5_helpers = evaluate_with_helpers(model, val, features, k=5)  # val DataFrame has original ratings
    logger.info(f"Validation NDCG@5 (helpers): {ndcg5_helpers:.4f}")

    # 7) Baselines
    price_ndcg = baseline_ndcg(val, sort_by="price_usd", ascending=True, k=5)
    rating_ndcg = baseline_ndcg(val, sort_by="prop_starrating", ascending=False, k=5)

    logger.info(f"Baseline NDCG@5 (lowest price): {price_ndcg:.4f}")
    logger.info(f"Baseline NDCG@5 (highest rating): {rating_ndcg:.4f}")

    # Log evaluation metrics
    eval_metrics = {
        "ndcg@5": ndcg5,
        "ndcg@5_helpers": ndcg5_helpers,
        "price_baseline": price_ndcg,
        "rating_baseline": rating_ndcg,
    }
    log_evaluation_summary(eval_metrics)

    # 8) Load test data for the competition
    logger.info("Loading test data...")
    test_df = pl.read_ipc("data/processed/test_processed.feather")

    # 9) Preprocess test data
    test_df = preprocess(test_df)

    # 10) Make predictions on test data
    logger.info("Generating predictions...")
    predictions = model.predict(test_df.select(features).to_pandas())
    test_df = test_df.with_columns(pl.lit(predictions).alias("predictions"))

    # 11) Generate submission file with timestamp and NDCG score
    logger.info("Creating Kaggle submission file...")
    generate_submission(test_df, pred_col="predictions", ndcg_score=ndcg5)

    logger.info("Submission process completed!")

    # Also save the model for future use
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(model_dir, f"lambdamart_model_{timestamp}.txt")
    model.booster_.save_model(model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
