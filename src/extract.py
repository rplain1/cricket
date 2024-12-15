import duckdb
from pathlib import Path

# this was the quickest way for me to deal with the json data
# and get it into a dataframe. I like to use duckdb for I/O wherever
# it makes sense to. Having it in an agreed on parquet format lets me use
# Positron to read in the data with R and do the initial EDA iterations
# more quickly


def main() -> None:
    raw_data_dir = Path("raw-data")
    match_results_file = raw_data_dir / "match_results.json"
    innings_results_file = raw_data_dir / "innings_results.json"

    if not match_results_file.exists():
        raise FileNotFoundError(f"Required file not found: {match_results_file}")
    if not innings_results_file.exists():
        raise FileNotFoundError(f"Required file not found: {innings_results_file}")

    out_dir = Path("data/extracted/")
    out_dir.mkdir(exist_ok=True, parents=True)
    """
    Convert the nested json input data into a parquet format that is in the form
    of a 2 dimensional table.

    Returns:
        None
    """
    duckdb.execute("""
        COPY (SELECT * FROM 'raw-data/match_results.json')
        TO 'data/extracted/match_results.parquet' (FORMAT PARQUET);
    """)

    duckdb.execute("""
        COPY (from read_json('raw-data/innings_results.json', maximum_depth = -1, sample_size = -1))
        TO 'data/extracted/innings_results.parquet' (FORMAT PARQUET);
    """)

    duckdb.execute("""
    COPY (select distinct teams from read_json('raw-data/match_results.json'))
    TO 'data/extracted/teams.csv' (FORMAT CSV)
    """)


if __name__ == "__main__":
    main()
