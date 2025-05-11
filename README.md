# Expedia Hotel Recommendation System

This project implements a recommendation system for Expedia hotel searches using Matrix Factorization techniques. The goal is to predict which hotel a user is most likely to book based on their search query and hotel properties.

## Dataset

The dataset contains Expedia search and booking data with the following key features:

- Search session information (search ID, destination, dates, etc.)
- Hotel property details (star rating, review score, price, etc.)
- User interaction data (clicks and bookings)

## Approach

We implemented a Matrix Factorization approach using Singular Value Decomposition (SVD) with the Surprise library. This technique models the user-item interactions by decomposing the user-item matrix into latent factors.

Key steps in our approach:

1. Data preprocessing to create a user-item-rating matrix
2. Model training using SVD with hyperparameter tuning
3. Evaluation using NDCG@5 metric
4. Generation of recommendations for the test set

## Project Structure

```
├── data/
│   ├── raw/                # Original data from Kaggle
│   ├── processed/          # Processed data for modeling
│   ├── interim/            # Intermediate data files
│   └── external/           # External data sources
├── src/dmt/                # Source code
│   ├── data_processing/    # Data processing scripts
│   ├── features/           # Feature engineering
│   ├── models/             # Model training and prediction
│   ├── visualization/      # Visualization scripts
│   └── utils/              # Utility functions
├── models/                 # Saved models
├── submissions/            # Generated submission files
├── reports/                # Analysis reports and figures
└── config/                 # Configuration files
```

## Setup and Usage

This project uses [UV](https://github.com/astral-sh/uv) for Python package management and execution. UV automatically handles dependency installation and environment management, making setup extremely simple.

To run the pipeline:

1. Process the data:

   ```
   uv run python -m src.dmt.data_processing.make_dataset
   ```

2. Train the model:

   ```
   uv run python -m src.dmt.models.train_model
   ```

3. Generate predictions:

   ```
   uv run python -m src.dmt.models.predict_model
   ```

4. Create visualizations:

   ```
   uv run python -m src.dmt.visualization.visualize
   ```

No manual dependency installation is required as UV automatically handles all package dependencies defined in `pyproject.toml`.

## Visualization

Interactive visualizations are generated as HTML files and saved in the `reports/figures/` directory. These can be opened in any web browser to explore the data and model results.

## Evaluation

The model is evaluated using Normalized Discounted Cumulative Gain (NDCG) at rank 5, which is calculated per query and averaged over all queries with values weighted by the log_2 function. The NDCG metric takes into account the relevance grade of each item and its position in the ranked list.
