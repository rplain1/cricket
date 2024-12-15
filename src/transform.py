"""
I'm using this script to handle the transforming the extracted data into valid
business logic. This layer should contain anything related to Cricket, and creating
datasets that can help answer the questions we are trying to solve for.

Because I had already gotten started on this, the `extract` method I chose with
duckdb caused too much unnesting. There were duplicated rows for a ball because of
wicket.player_out. I handled the filtering here, but that should probably get moved
upstream if there was more time allocated to the project.

Question 3a. is handled here. I outputted it as a step in this process, it's
unlikely a request would be needed like this in production environment.
"""

import polars as pl
import os
import extract

INPUT_DIR = "data/extracted"
OUTPUT_DIR = "data/transformed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_input_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Reads match and innings result data from parquet files and returns them as Polars DataFrames.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing two Polars DataFrames:
            - match_results_df: DataFrame containing match results data
            - innings_results_df: DataFrame containing innings results data
    """
    match_results_df = pl.read_parquet(f"{INPUT_DIR}/match_results.parquet")
    innings_results_df = pl.read_parquet(f"{INPUT_DIR}/innings_results.parquet")

    return match_results_df, innings_results_df


def join_input_data(match_results_df: pl.DataFrame, innings_results_df: pl.DataFrame) -> pl.DataFrame:
    """
    Joins match results and innings results data, filters and processes specific columns.

    The function filters `match_results_df` for male matches with null results, groups by `matchid`, and
    joins it with selected columns from `innings_results_df`. The resulting DataFrame is further processed
    to split the 'over' column to derive `over` and `ball`, cast columns to appropriate types, and unnest the 'over' field.

    Args:
        match_results_df (pl.DataFrame): DataFrame containing match results data.
        innings_results_df (pl.DataFrame): DataFrame containing innings results data.

    Returns:
        pl.DataFrame: A processed DataFrame containing joined and transformed data.
    """
    df = (
        match_results_df.filter((pl.col("result").is_null()) & (pl.col("gender") == "male"))
        .group_by(["matchid", "dates"])
        .len()
        .join(
            innings_results_df.select(
                ["matchid", "over", "innings", "team", "batsman", "bowler", "runs.total", "wicket.player_out"]
            ),
            on="matchid",
        )
        .with_columns(pl.col("over").str.split_exact(".", 1).struct.rename_fields(["over", "ball"]))
        .unnest("over")
        .with_columns(
            dates=pl.col("dates").cast(pl.Date),
            over=pl.col("over").cast(pl.Int64),
            ball=pl.col("ball").cast(pl.Int64),
            wicket=pl.col("wicket.player_out")
            .is_not_null()
            .cast(pl.Int64),  # unnesting like I did caused multiple rows for wicket player
            rn=pl.col("wicket.player_out").rank("ordinal").over(["matchid", "over", "innings", "team"]),
        )
        .filter(pl.col("rn") == 1)
        .drop("rn")
    )

    return df


def aggregate_input_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate the data from `join_input_data() into a format that is 1 row
    per `team`, `matchid`, `over`, and `ball`. This is used to to make a dataset
    with the run and wicket outcomes for each delivery in a match.
    """

    df = (
        df.group_by(["team", "dates", "matchid", "innings", "over", "ball"])
        .agg(runs=pl.col("runs.total").sum(), wickets=pl.col("wicket").sum())
        .with_columns(
            total_runs=pl.col("runs").cum_sum().over(["team", "matchid"], order_by=["over", "ball"]),
            total_wickets=pl.col("wickets").cum_sum().over(["team", "matchid"], order_by=["over", "ball"]),
        )
        .with_columns(overs_remaining=50 - pl.col("over") - 1, wickets_remaining=10 - pl.col("total_wickets"))
    ).select(["team", "innings", "overs_remaining", "wickets_remaining", "runs"])

    return df


def main() -> None:
    """
    Main function to process input data, join, aggregate, and save the results.

    This function retrieves match and innings result data, performs a join operation on the data,
    aggregates the resulting DataFrame, and writes the final report as a CSV file.

    Used to answer question 3a

    Steps:
        1. Retrieves match and innings data using `get_input_data()`.
        2. Joins the data with `join_input_data()`.
        3. Aggregates the data using `aggregate_input_data()`.
        4. Writes the final aggregated data to a CSV file.
        5. Writes the dataset for 3a
        6. Write Ireland test case data

    Returns:
        None
    """
    matches = os.path.isfile(f"{INPUT_DIR}/match_results.parquet")
    innings = os.path.isfile(f"{INPUT_DIR}/innings_results.parquet")

    if not all([matches, innings]):
        extract.main()

    match_results_df, innings_results_df = get_input_data()
    df = join_input_data(match_results_df, innings_results_df)
    df = aggregate_input_data(df)
    df = df.sort(["team", "innings", "overs_remaining", "wickets_remaining"])
    df.write_parquet(f"{OUTPUT_DIR}/transformed_data.parquet")
    df.select(["team", "innings", "overs_remaining", "wickets_remaining"]).write_csv("report/q3a.csv")
    df.filter(pl.col("team") == "Ireland").slice(1, 5).drop("runs").write_csv("report/ireland_first_5_overs.csv")


if __name__ == "__main__":
    main()
