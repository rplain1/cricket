import duckdb
from pathlib import Path
import gdown

# this was the quickest way for me to deal with the json data
# and get it into a dataframe. I like to use duckdb for I/O wherever
# it makes sense to. Having it in an agreed on parquet format lets me use
# Positron to read in the data with R and do the initial EDA iterations
# more quickly


def main() -> None:
    """
    Downloads required JSON files from Google Drive if they do not already exist in the
    'raw-data' directory, and processes them using DuckDB to extract and save the data
    into Parquet and CSV formats.

    The function performs the following steps:
    1. Checks if the files 'match_results.json' and 'innings_results.json' exist in the
       'raw-data' directory.
    2. If either of the files is missing, downloads them from Google Drive using the
       'gdown' package.
    3. Ensures both 'match_results.json' and 'innings_results.json' exist after
       downloading. Raises a FileNotFoundError if any file is missing.
    4. Creates the 'data/extracted/' directory if it doesn't exist.
    5. Uses DuckDB to:
        - Copy the content of 'match_results.json' into a Parquet file.
        - Read and convert 'innings_results.json' to Parquet format with specific
          parameters on the data structure.
        - Extract distinct teams from 'match_results.json' and save the result into a
          CSV file.

    Raises:
        FileNotFoundError: If the required JSON files are not found after attempting
        to download them.

    Returns:
        None
    """
    # wanted to go with a programatic solution to be able to easily execute in docker
    # these are the source files from links in the challenge info shared. Covnerted into
    # the appropriate url
    source_data = {"innings": "1wQO9zr1VH8bY2W4Ca6cMxPdAoPOHo6X6", "match": "19hVoi9f7n7etcmSXx7WHeiDp9pOLpQvN"}

    raw_data_dir = Path("raw-data")
    raw_data_dir.mkdir(exist_ok=True)

    match_results_file = raw_data_dir / "match_results.json"
    innings_results_file = raw_data_dir / "innings_results.json"

    # Check if the files exist, if not, download them
    if not match_results_file.exists() or not innings_results_file.exists():
        print("Downloading missing files...")
        for file, file_id in source_data.items():
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            output = raw_data_dir / f"{file}_results.json"  # Save as appropriate file
            gdown.download(url, str(output), quiet=False)
    else:
        print("Files already exist. Skipping download.")

    if not match_results_file.exists():
        raise FileNotFoundError(f"Required file not found: {match_results_file}")
    if not innings_results_file.exists():
        raise FileNotFoundError(f"Required file not found: {innings_results_file}")

    out_dir = Path("data/extracted/")
    out_dir.mkdir(exist_ok=True, parents=True)

    # use duckdb for converting the file from json to parquet
    duckdb.execute("""
        COPY (SELECT * FROM 'raw-data/match_results.json')
        TO 'data/extracted/match_results.parquet' (FORMAT PARQUET);
    """)

    duckdb.execute("""
        COPY (from read_json('raw-data/innings_results.json', maximum_depth = -1, sample_size = -1))
        TO 'data/extracted/innings_results.parquet' (FORMAT PARQUET);
    """)

    # teams flat file is later used in model testing
    # ideally there would be a datawarehouse that this would be replaced by
    duckdb.execute("""
    COPY (select distinct teams from read_json('raw-data/match_results.json'))
    TO 'data/extracted/teams.csv' (FORMAT CSV)
    """)


if __name__ == "__main__":
    main()
