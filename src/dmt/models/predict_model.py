import logging
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_model(model_path):
    """Load the trained SVD model"""
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_test_data(test_processed_path):
    """Load the processed test data"""
    logger.info(f"Loading processed test data from {test_processed_path}")
    test_data = pd.read_feather(test_processed_path)
    return test_data


def generate_predictions(model, test_data):
    """Generate predictions for all search-property pairs in the test data"""
    logger.info("Generating predictions...")

    # Group test data by search ID to get all properties for each search
    search_property_pairs = defaultdict(list)
    for _, row in test_data.iterrows():
        search_property_pairs[row["srch_id"]].append(row["prop_id"])

    # Generate predictions
    predictions = []
    for srch_id, prop_ids in search_property_pairs.items():
        srch_predictions = []
        for prop_id in prop_ids:
            # Predict rating for this search-property pair
            predicted_rating = model.predict(str(srch_id), str(prop_id)).est
            srch_predictions.append((srch_id, prop_id, predicted_rating))

        # Sort properties by predicted rating (descending)
        srch_predictions.sort(key=lambda x: x[2], reverse=True)
        predictions.extend([(p[0], p[1]) for p in srch_predictions])

    # Convert to DataFrame with search_id, prop_id columns
    predictions_df = pd.DataFrame(predictions, columns=["srch_id", "prop_id"])

    return predictions_df


def save_submission(predictions_df, submission_dir):
    """Save predictions to a submission file"""
    os.makedirs(submission_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_path = os.path.join(submission_dir, f"submission_{timestamp}.csv")

    logger.info(f"Saving submission to {submission_path}")
    predictions_df.to_csv(submission_path, index=False)

    return submission_path


def main(model_path=None):
    """Main function to generate and save predictions"""
    # Set up logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Get project directory
    project_dir = Path(__file__).resolve().parents[3]

    # Define paths
    processed_dir = os.path.join(project_dir, "data", "processed")
    test_processed_path = os.path.join(processed_dir, "test_processed.feather")
    model_dir = os.path.join(project_dir, "models")
    submission_dir = os.path.join(project_dir, "submissions")

    # If model path not provided, use the most recent model
    if model_path is None:
        model_files = [f for f in os.listdir(model_dir) if f.startswith("svd_model_") and f.endswith(".pkl")]
        if not model_files:
            logger.error("No trained model found in models directory.")
            return None
        model_files.sort(reverse=True)
        model_path = os.path.join(model_dir, model_files[0])

    # Load model and test data
    model = load_model(model_path)
    test_data = load_test_data(test_processed_path)

    # Generate predictions
    predictions_df = generate_predictions(model, test_data)

    # Save submission
    submission_path = save_submission(predictions_df, submission_dir)

    logger.info("Prediction completed.")
    return submission_path


if __name__ == "__main__":
    main()
