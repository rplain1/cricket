import duckdb
import polars as pl

duckdb.execute("""
COPY (SELECT * FROM 'data/match_results.json') TO 'data/match_results.parquet' (FORMAT PARQUET);

""")

duckdb.execute("""
COPY (from read_json('data/innings_results.json', maximum_depth = -1, sample_size = -1)) TO 'data/innings_results.parquet' (FORMAT PARQUET);

""")

duckdb.sql("""

with matches as (
    select matchid
    , count(*)
    from 'data/match_results.parquet'
    where result is NULL
    group by ALL
)

select inn.over
, string_split_regex(inn.over, '\\.')[1]
, string_split_regex(inn.over, '\\.')[2]
from 'data/innings_results.parquet' inn
join  matches m on inn where m.matchid = inn.matchid
""")


match_results_df = pl.read_parquet('data/match_results.parquet')
innings_results_df = pl.read_parquet('data/innings_results.parquet')

df = (
    match_results_df
    .filter(pl.col('result').is_null())['matchid']
    .value_counts()
    .join(
        innings_results_df
        .select(
            [
                'matchid',
                'over',
                'innings',
                'team',
                'batsman',
                'bowler',
                'runs.batsman',
                'runs.extras'
                ]
            ),
        on = 'matchid'
        )
    .with_columns(
        pl.col('over').str.split_exact('.', 1).struct.rename_fields(['over', 'runs'])
    )
    .unnest('over')
    .with_columns(
        pl.col('over').cast(pl.Int64),
        pl.col('runs').cast(pl.Int64)
    )
)
