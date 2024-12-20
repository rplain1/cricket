import pytest
from transform import get_input_data, join_input_data, aggregate_input_data
import polars as pl


@pytest.fixture
def input_data():
    """
    Fixture that retrieves input data for the test.

    Returns:
        tuple: A tuple containing two DataFrames (matches, innings).
    """
    return get_input_data()


@pytest.fixture
def joined_data(input_data):
    """
    Fixture that joins the input data (matches and innings).

    Args:
        input_data (tuple): A tuple containing two DataFrames (matches, innings).

    Returns:
        DataFrame: The joined DataFrame with matches and innings data.
    """
    return join_input_data(input_data[0], input_data[1])


@pytest.fixture
def aggregated_data(joined_data):
    """
    Fixture that aggregates the joined data.

    Args:
        joined_data (DataFrame): The DataFrame containing joined matches and innings data.

    Returns:
        DataFrame: The aggregated data after applying business logic.
    """
    return aggregate_input_data(joined_data)


@pytest.mark.parametrize(
    "expected",
    ["over", "runs.total", "wicket.player_out"],
)
def test_get_input_data(input_data, expected):
    """
    Test `get_input_data()` to ensure it returns valid data for matches
    and innings.

    Args:
        input_data (tuple): A tuple containing two DataFrames (matches, innings).
        expected (str): Column name to check in innings data.

    Asserts:
        - The input data for matches and innings is a DataFrame.
        - The DataFrames are not empty.
        - The expected column is present in the innings DataFrame.
    """

    df_matches = input_data[0]
    df_innings = input_data[1]

    assert isinstance(df_matches, pl.DataFrame)
    assert isinstance(df_innings, pl.DataFrame)

    assert df_matches.shape[0] > 0
    assert df_innings.shape[0] > 0

    assert "matchid" in df_matches.columns, "'matchid' not in df_matches"
    assert expected in df_innings.columns


@pytest.mark.parametrize(
    "expected",
    ["ball", "wicket"],
)
def test_join_input_data(joined_data, expected):
    """
    Test `join_input_data()` to ensure matches and innings data are being
    joined correctly, and transformations are in place.

    Over is represented as a string "0.1", "1.2", "11.3" to represent over and
    ball. These are split to create numeric columns.

    Args:
        joined_data (DataFrame): The joined data of matches and innings.
        expected (str): Column name to check in the joined data.

    Asserts:
        - The joined data is a DataFrame.
        - Expected columns are present in the DataFrame.
        - The over and ball columns are properly split into numeric types.
        - The date column is correctly converted to a Date type.
        - No issues with duplicate values for `wicket.player_out`.
    """
    df = joined_data

    assert isinstance(df, pl.DataFrame)
    # ensuring that all the data type and column transformations
    # that were needed took place
    assert expected in df.columns

    # this had a character split to calculate numeric cols
    assert df["over"].dtype == pl.Int64
    assert df["ball"].dtype == pl.Int64

    # date column conversion
    assert df["dates"].dtype == pl.Date

    # testing that there are no situations where the wicket.player_out isn't going to
    # cause downstream errors
    group_cols = ["matchid", "over", "innings", "team"]
    assert df.group_by(group_cols).len()["len"].max() == 1


def test_aggregate_data(aggregated_data):
    """
    Test business logic after `aggregate_input_data()` to ensure correct
    aggregation and rule adherence.

    Args:
        aggregated_data (DataFrame): The aggregated data after applying business rules.

    Asserts:
        - The team column is a string.
        - The innings column has exactly two unique values.
        - Wickets remaining are within valid bounds (0 to 10).
        - Overs remaining are between 0 and 49.
    """
    df = aggregated_data

    assert df["team"].dtype == pl.String
    assert df["innings"].n_unique() == 2
    assert df["wickets_remaining"].max() <= 10
    assert df["wickets_remaining"].min() == 0
    assert df["overs_remaining"].min() == 0
    assert df["overs_remaining"].max() == 49
