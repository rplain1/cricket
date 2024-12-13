import polars as pl
import os
import convert_file

INPUT_DIR = "data"
OUTPUT_DIR = "report"
assert os.path.exists(INPUT_DIR), "'data/' doesn't exits"


def get_input_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Reads match and innings result data from parquet files and returns them as Polars DataFrames.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing two Polars DataFrames:
            - match_results_df: DataFrame containing match results data
            - innings_results_df: DataFrame containing innings results data
    """
    match_results_df = pl.read_parquet("data/match_results.parquet")
    innings_results_df = pl.read_parquet("data/innings_results.parquet")

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
        .group_by(["matchid"])
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
            over=pl.col("over").cast(pl.UInt8),
            ball=pl.col("ball").cast(pl.UInt8),
            wicket=pl.col("wicket.player_out").is_not_null().cast(pl.UInt8),
        )
    )

    return df


def aggregate_input_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate the data from `join_input_data() into a format that is 1 row
    per `team`, `matchid`, `over`, and `ball`. This is used to to make a dataset
    with the run and wicket outcomes for each delivery in a match.
    """
    return df.group_by(["team", "matchid", "over", "ball"]).agg(
        runs=pl.col("runs.total").sum(), wickets=pl.col("wicket").sum()
    )


def main() -> None:
    """
    Main function to process input data, join, aggregate, and save the results.

    This function retrieves match and innings result data, performs a join operation on the data,
    aggregates the resulting DataFrame, and writes the final report as a JSON file.

    Used to answer question 3a

    Steps:
        1. Retrieves match and innings data using `get_input_data()`.
        2. Joins the data with `join_input_data()`.
        3. Aggregates the data using `aggregate_input_data()`.
        4. Writes the final aggregated data to a JSON file.

    Returns:
        None
    """
    matches = os.path.isfile(f"{INPUT_DIR}/match_results.parquet")
    innings = os.path.isfile(f"{INPUT_DIR}/innings_results.parquet")

    if not all([matches, innings]):
        convert_file.main()

    match_results_df, innings_results_df = get_input_data()
    df = join_input_data(match_results_df, innings_results_df)
    df = aggregate_input_data(df)
    df.write_csv(f"{OUTPUT_DIR}/q3a.csv")


if __name__ == "__main__":
    main()