import pytest
from staging import get_input_data
import polars as pl


@pytest.fixture
def input_data():
    return get_input_data()


def test_get_input_data(input_data):
    df_matches = input_data[0]
    df_innings = input_data[1]

    assert isinstance(df_matches, pl.DataFrame)
    assert isinstance(df_innings, pl.DataFrame)
