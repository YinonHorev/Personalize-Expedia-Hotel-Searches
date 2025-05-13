import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_raw_data(train_path, test_path):
    """Read raw data files"""
    logger.info(f"Reading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Reading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df, save_path=None):
    """Preprocess data for matrix factorization

    For matrix factorization, we need to create a user-item-rating matrix
    where:
    - user = srch_id (search session)
    - item = prop_id (hotel property)
    - rating = derived from booking_bool and click_bool
    """
    logger.info("Processing training data...")

    # Create rating column based on booking and click information
    # Using the relevance grades as mentioned: 5 for booking, 1 for click, 0 for no interaction
    train_df["rating"] = 0
    train_df.loc[train_df["click_bool"] == 1, "rating"] = 1
    train_df.loc[train_df["booking_bool"] == 1, "rating"] = 5

    # Keep only necessary columns for matrix factorization
    train_processed = train_df[["srch_id", "prop_id", "rating"]]

    # Process test data (no ratings available)
    logger.info("Processing test data...")
    test_processed = test_df[["srch_id", "prop_id"]]

    # Save processed data if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        train_processed_path = os.path.join(save_path, "train_processed.feather")
        test_processed_path = os.path.join(save_path, "test_processed.feather")

        logger.info(f"Saving processed training data to {train_processed_path}")
        train_processed.reset_index(drop=True).to_feather(train_processed_path)

        logger.info(f"Saving processed test data to {test_processed_path}")
        test_processed.reset_index(drop=True).to_feather(test_processed_path)

    return train_processed, test_processed


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

    # Read and process data
    train_df, test_df = read_raw_data(train_path, test_path)
    train_processed, test_processed = preprocess_data(train_df, test_df, processed_dir)

    logger.info("Data processing completed.")


if __name__ == "__main__":
    main()
