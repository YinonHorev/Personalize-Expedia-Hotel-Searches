from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore


def plot_missing_values(df: pd.DataFrame) -> str | Any:
    """Generates a bar plot showing the percentage of missing values for each column.
    Only columns with missing values are included in the plot.

    This visualization helps identify data quality issues and highlights
    columns with significant missing data that may need special treatment
    during preprocessing.
    """
    # Calculate missing values
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) == 0:
        return "No missing values found in the dataset"

    # Create the plot
    fig = px.bar(
        x=missing_values.index,
        y=(missing_values / len(df) * 100).round(2),
        labels={"x": "Features", "y": "Percentage of Missing Values"},
        title="Missing Values by Feature",
    )

    # Add percentage labels
    percentages = (missing_values / len(df) * 100).round(2)
    annotations = [f"{val} ({pct}%)" for val, pct in zip(missing_values.values, percentages, strict=False)]

    fig.update_traces(text=annotations, textposition="outside")
    fig.update_layout(height=600, width=1000)

    return fig


def plot_zero_values(df: pd.DataFrame) -> Any:
    """Generates a bar plot showing the number of zero values for each column.
    Only includes columns where zero values have significance based on the dataset description.

    This visualization helps identify patterns in the data where zero values
    represent specific conditions (e.g., no reviews, no historical price data)
    rather than just the absence of a value.
    """
    # Columns where zero values are significant according to dataset description
    sig_zero_cols = [
        "prop_starrating",  # 0 indicates no stars, unknown, or unpublicized rating
        "prop_review_score",  # 0 means no reviews
        "prop_log_historical_price",  # 0 if not sold in that period
        "random_bool",  # 0 for normal sort order
        "prop_brand_bool",  # 0 if independent hotel
        "promotion_flag",  # 0 if no promotion
        "click_bool",  # 0 if not clicked
        "booking_bool",  # 0 if not booked
    ]

    # Calculate zero values for significant columns
    zero_values = {}
    for col in sig_zero_cols:
        if col in df.columns:
            zero_values[col] = (df[col] == 0).sum()

    zero_df = pd.Series(zero_values).sort_values(ascending=False)

    # Create the plot
    fig = px.bar(
        x=zero_df.index,
        y=zero_df.values,
        labels={"x": "Features", "y": "Number of Zero Values"},
        title="Zero Values in Significant Features",
    )

    # Add percentage labels
    percentages = (zero_df / len(df) * 100).round(2)
    annotations = [f"{val} ({pct}%)" for val, pct in zip(zero_df.values, percentages, strict=False)]

    fig.update_traces(text=annotations, textposition="outside")
    fig.update_layout(height=600, width=1000)

    return fig


def plot_click_booking_relation(df: pd.DataFrame) -> Any:
    """Visualizes the relationship between click_bool and booking_bool.

    This plot shows the distribution of user behaviors in the dataset,
    highlighting the progression from clicks to bookings and providing
    insight into conversion rates on the Expedia platform.
    """
    # Group data to get counts
    click_book_counts = df.groupby(["click_bool", "booking_bool"]).size().reset_index(name="count")

    # Create labels for combinations
    conditions = [
        (click_book_counts["click_bool"] == 0) & (click_book_counts["booking_bool"] == 0),
        (click_book_counts["click_bool"] == 1) & (click_book_counts["booking_bool"] == 0),
        (click_book_counts["click_bool"] == 0) & (click_book_counts["booking_bool"] == 1),
        (click_book_counts["click_bool"] == 1) & (click_book_counts["booking_bool"] == 1),
    ]

    categories = ["No Click, No Booking", "Click, No Booking", "No Click, Booking (Error/Direct)", "Click and Booking"]

    click_book_counts["category"] = np.select(conditions, categories, default="Unknown")

    # Create the plot
    fig = px.bar(
        click_book_counts,
        x="category",
        y="count",
        title="Relationship Between Clicks and Bookings",
        labels={"count": "Number of Instances", "category": "User Behavior"},
        color="category",
    )

    # Add percentage labels
    total = click_book_counts["count"].sum()
    percentages = (click_book_counts["count"] / total * 100).round(2)

    # Fix: Add annotations as text to each bar individually
    for i, (val, pct) in enumerate(zip(click_book_counts["count"], percentages, strict=False)):
        fig.add_annotation(
            x=click_book_counts["category"][i], y=val, text=f"{val} ({pct}%)", showarrow=False, yshift=10
        )

    return fig


def plot_stay_vs_price_for_bookings(df: pd.DataFrame) -> Any:
    """Creates a scatter plot of length of stay vs price for booked properties.

    This visualization explores the relationship between the length of stay
    and the price users are willing to pay when making bookings, using a
    logarithmic scale to better visualize the wide price range.
    """
    # Filter to only booked properties
    booked_df = df[df["booking_bool"] == 1]

    # Create a scatter plot with log scale for price
    fig = px.scatter(
        booked_df,
        x="srch_length_of_stay",
        y="price_usd",
        title="Length of Stay vs Price for Booked Properties (Log Scale)",
        labels={"srch_length_of_stay": "Length of Stay (nights)", "price_usd": "Price (USD) - Log Scale"},
        opacity=0.6,
    )

    # Set y-axis to log scale
    fig.update_layout(yaxis_type="log")

    return fig


def compare_booking_rates_by_sort(df: pd.DataFrame) -> Any:
    """Compares booking rates between random sort (random_bool=1) and
    Expedia's normal sort order (random_bool=0).

    This analysis helps evaluate the effectiveness of Expedia's sorting algorithm
    compared to random sorting, providing insight into how sorting influences
    user booking behavior.
    """
    # Group by random_bool and calculate booking rates
    sort_booking = (
        df.groupby("random_bool")
        .agg(
            total_properties=("booking_bool", "count"),
            booked_properties=("booking_bool", "sum"),
        )
        .reset_index()
    )

    sort_booking["booking_rate"] = (sort_booking["booked_properties"] / sort_booking["total_properties"] * 100).round(2)

    # Add labels for clarity
    sort_booking["sort_type"] = sort_booking["random_bool"].apply(lambda x: "Random Sort" if x == 1 else "Expedia Sort")

    # Create the plot
    fig = px.bar(
        sort_booking,
        x="sort_type",
        y="booking_rate",
        title="Booking Rate by Sort Method",
        labels={"sort_type": "Sort Method", "booking_rate": "Booking Rate (%)"},
        color="sort_type",
    )

    # Fix: Add annotations as text to each bar individually
    for i, rate in enumerate(sort_booking["booking_rate"]):
        fig.add_annotation(x=sort_booking["sort_type"][i], y=rate, text=f"{rate}%", showarrow=False, yshift=10)

    # Add counts as hover info
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Booking Rate: %{y}%<br>Booked: %{customdata[0]}<br>Total: %{customdata[1]}",
        customdata=sort_booking[["booked_properties", "total_properties"]],
    )

    return fig


def run_all_analyses(df: pd.DataFrame = None) -> dict[str, Any]:
    """Runs all the defined analysis functions on the provided dataframe.
    If no dataframe is provided, it uses the training dataset by default.

    Returns a dictionary containing all the generated plots for easy access.
    """
    results = {
        "missing_values": plot_missing_values(df),
        "zero_values": plot_zero_values(df),
        "click_booking_relation": plot_click_booking_relation(df),
        "stay_vs_price": plot_stay_vs_price_for_bookings(df),
        "booking_rates_by_sort": compare_booking_rates_by_sort(df),
    }

    return results


if __name__ == "__main__":
    # This section runs if the script is executed directly
    # It's a good place to add any immediate visualizations for testing
    results = run_all_analyses()

    # Print a message about how to use these functions in a notebook
    print("Analysis functions are ready to use.")
    print("Import this module in a notebook and call the individual functions or run_all_analyses().")
