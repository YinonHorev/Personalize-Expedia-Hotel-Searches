import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV, train_test_split

logger = logging.getLogger(__name__)


def load_data(train_processed_path):
    """Load the processed training data"""
    logger.info(f"Loading processed training data from {train_processed_path}")
    train_data = pd.read_feather(train_processed_path)
    return train_data


def prepare_surprise_data(train_data):
    """Prepare data for Surprise library"""
    # Define reader
    reader = Reader(rating_scale=(0, 5))

    # Load data into Surprise Dataset
    data = Dataset.load_from_df(train_data[["srch_id", "prop_id", "rating"]], reader)

    # Split into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    return data, trainset, testset


def tune_model(data):
    """Optional: Tune SVD hyperparameters using grid search"""
    logger.info("Tuning SVD hyperparameters...")

    # Define parameter grid
    param_grid = {"n_factors": [50, 100, 150], "n_epochs": [20, 30], "lr_all": [0.005, 0.01], "reg_all": [0.02, 0.1]}

    # Perform grid search
    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
    gs.fit(data)

    # Get best parameters
    best_params = gs.best_params["rmse"]
    logger.info(f"Best parameters: {best_params}")

    return best_params


def train_svd_model(trainset, testset, params=None, tune=False, data=None):
    """Train SVD model with optional hyperparameter tuning"""
    if tune and data is not None:
        params = tune_model(data)

    # Create SVD model with specified or default parameters
    if params:
        logger.info(f"Training SVD model with parameters: {params}")
        model = SVD(
            n_factors=params.get("n_factors", 100),
            n_epochs=params.get("n_epochs", 20),
            lr_all=params.get("lr_all", 0.005),
            reg_all=params.get("reg_all", 0.02),
        )
    else:
        logger.info("Training SVD model with default parameters")
        model = SVD()

    # Train the model
    model.fit(trainset)

    # Test the model
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    logger.info(f"Model performance - RMSE: {rmse}, MAE: {mae}")

    return model, predictions


def save_model(model, model_dir):
    """Save the trained model"""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(model_dir, f"svd_model_{timestamp}.pkl")

    logger.info(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path


def main(tune=False):
    """Main function to train and save the SVD model"""
    # Set up logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Get project directory
    project_dir = Path(__file__).resolve().parents[3]

    # Define paths
    processed_dir = os.path.join(project_dir, "data", "processed")
    train_processed_path = os.path.join(processed_dir, "train_processed.feather")
    model_dir = os.path.join(project_dir, "models")

    # Load data
    train_data = load_data(train_processed_path)

    # Prepare data for Surprise
    data, trainset, testset = prepare_surprise_data(train_data)

    # Train model
    model, predictions = train_svd_model(trainset, testset, tune=tune, data=data)

    # Save model
    model_path = save_model(model, model_dir)

    logger.info("Model training completed.")
    return model_path


if __name__ == "__main__":
    main(tune=False)  # Set tune=True to perform hyperparameter tuning
