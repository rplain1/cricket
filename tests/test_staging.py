import pytest
from staging import get_input_data, join_input_data, aggregate_input_data
import polars as pl


@pytest.fixture
def input_data():
    return get_input_data()


@pytest.fixture
def joined_data(input_data):
    return join_input_data(input_data[0], input_data[1])


@pytest.fixture
def aggregated_data(joined_data):
    return aggregate_input_data(joined_data)


@pytest.mark.parametrize(
    "expected",
    ["over", "runs.total", "wicket.player_out"],
)
def test_get_input_data(input_data, expected):
    """
    Test `get_input_data()` that it returns valid data for matches
    and innings.
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
    """
    df = joined_data

    assert isinstance(df, pl.DataFrame)
    # ensuring that all the data type and column transformations
    # that were needed took place
    assert expected in df.columns

    # this had a character split to calculate numeric cols
    assert df["over"].dtype == pl.Int64
    assert df["ball"].dtype == pl.Int64

    # testing that there are no situations where the wicket.player_out isn't going to
    # cause downstream errors
    group_cols = ["matchid", "over", "innings", "team"]
    assert df.group_by(group_cols).len()["len"].max() == 1


def test_aggregate_data(aggregated_data):
    """
    Test business logic is working correctly after `aggregate_input_data()`.
    This ensures all of the values follow the rules of the game.
    """
    df = aggregated_data

    assert df["total_runs"].min() >= 0
    assert df["total_wickets"].min() >= 0
    assert df["total_wickets"].max() <= 10
    assert df["over"].min() == 0
    assert df["over"].max() + 1 == 50, "All 50 overs not represented"
