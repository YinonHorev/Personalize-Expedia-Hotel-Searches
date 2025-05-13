import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_ndcg(predictions, true_ratings, k=5):
    """Calculate Normalized Discounted Cumulative Gain (NDCG) at k.

    Parameters
    ----------
    predictions : list
        List of predicted items in order of relevance (highest to lowest)
    true_ratings : dict
        Dictionary mapping item_id to true relevance score
    k : int
        Number of top items to consider

    Returns
    -------
    float
        NDCG@k score

    """
    # Calculate DCG
    dcg = 0
    for i, item_id in enumerate(predictions[:k]):
        if item_id in true_ratings:
            # Position i is 0-based, but DCG formula uses 1-based positions
            dcg += (2 ** true_ratings[item_id] - 1) / np.log2(i + 2)

    # Calculate IDCG (Ideal DCG)
    ideal_order = sorted(true_ratings.items(), key=lambda x: x[1], reverse=True)
    idcg = 0
    for i, (item_id, rel) in enumerate(ideal_order[:k]):
        idcg += (2**rel - 1) / np.log2(i + 2)

    # Calculate NDCG
    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return ndcg


def evaluate_model_ndcg(predictions_df, ground_truth_df, k=5):
    """Evaluate model using NDCG@k metric for search recommendations.

    Parameters
    ----------
    predictions_df : pandas.DataFrame
        DataFrame with srch_id and prop_id columns, sorted by predicted relevance
    ground_truth_df : pandas.DataFrame
        DataFrame with srch_id, prop_id, and rating columns
    k : int
        Number of top items to consider

    Returns
    -------
    float
        Average NDCG@k score across all search queries

    """
    # Prepare ground truth data
    ground_truth = {}
    for _, row in ground_truth_df.iterrows():
        srch_id = row["srch_id"]
        prop_id = row["prop_id"]
        rating = row["rating"]

        if srch_id not in ground_truth:
            ground_truth[srch_id] = {}

        ground_truth[srch_id][prop_id] = rating

    # Calculate NDCG@k for each search query
    ndcg_scores = []
    for srch_id, group in predictions_df.groupby("srch_id"):
        if srch_id in ground_truth:
            # Get predicted properties for this search query
            predicted_props = group["prop_id"].tolist()

            # Get true relevance scores for this search query
            true_ratings = ground_truth[srch_id]

            # Calculate NDCG@k
            ndcg = calculate_ndcg(predicted_props, true_ratings, k=k)
            ndcg_scores.append(ndcg)

    # Calculate average NDCG@k
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return avg_ndcg


def log_evaluation_summary(eval_metrics, log_path=None):
    """Log evaluation metrics summary.

    Parameters
    ----------
    eval_metrics : dict
        Dictionary of evaluation metrics
    log_path : str, optional
        Path to save the metrics as a CSV file

    """
    logger.info("Evaluation Summary:")
    for metric, value in eval_metrics.items():
        logger.info(f"{metric}: {value}")

    if log_path:
        pd.DataFrame([eval_metrics]).to_csv(log_path, index=False)
        logger.info(f"Evaluation metrics saved to {log_path}")
