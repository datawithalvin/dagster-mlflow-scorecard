## =============== import package =============== 
import pandas as pd
import duckdb
import os

from dagster import (
    asset, multi_asset, AssetOut, AssetIn, Output,
    TableSchema, TableColumn, MaterializeResult, 
)
from dagster_duckdb import DuckDBResource


## =============== prep =============== 
@asset(
    group_name="initial_prep",
    tags={"asset_type":"DuckDBResource", 
          "data_source":"transactions.db"},
    code_version="0.1",
    owners=["alvinnoza.data@gmail.com", "team:Data Scientist"],
    compute_kind="duckdb",
    description="Load train-test table to pandas dataframe"
)
def train_test_set(context):
    try:
        """
            Load train-test set dari table
        """
        ## load data
        query = """SELECT * FROM train_test_set"""
        conn = duckdb.connect(os.getenv("DUCKDB_DATABASE"))
        train_test_set = conn.execute(query).fetch_df()

        ## log dataframe
        context.log.info("train-test shape: %s", train_test_set.shape)
        context.log.info("first five rows: \n%s", train_test_set.head())

        ## materialisasi schema
        columns = [
            TableColumn(name=col, type=str(train_test_set[col].dtype),
                        description=f"Sample value: {train_test_set[col].iloc[33]}")
            for col in train_test_set.columns
        ]

        n_rows = train_test_set.shape[0]
        n_cols = train_test_set.shape[1]

        yield MaterializeResult(
                metadata={
                    "dagster/column_schema": TableSchema(columns=columns),
                    "dagster/type": "pandas-dataframe",
                    "dagster/column_count": n_cols,
                    "dagster/row_count": n_rows
                }
            )
        
        return train_test_set.to_csv("./data/outputs/train-test-set.csv", index=False)

    except Exception as e:
        context.log.error("An error occurred while build database connection/load credit application table: %s", str(e))
        raise e

## -------------- initial filter features -------------- 
## --------- filter based on iv dan chi2 p-value dari setiap fiturnya ---------
@multi_asset(
    deps=["train_test_set"],
    outs={
        "initial_filtered_set": AssetOut(
            description="filtered train-test set based on iv dan chi2 p-value.",
            code_version="0.0.2",
            tags={"asset_type":"pandas-dataframe"},
            owners=["alvinnoza.data@gmail.com", "team:ML-Engineer"],
        ), 
        "iv_dict": AssetOut(
            description="dictionary pairs signifikan fitur dan iv",
            code_version="0.1",
            tags={"asset_type":"python-dictionary"},
            owners=["alvinnoza.data@gmail.com", "team:ML-Engineer"],
        )
    },
    group_name="feature_filtering",
    compute_kind="pandas"
)
def significant_features(context):
    train_data_df = pd.read_csv("./data/outputs/train-test-set.csv")
    applications = train_data_df.copy()
    applications.set_index(keys="id", drop=True, inplace=True) ## <-- set id sbg index
    feature_names = list(applications.columns[1:]) ## <-- exclude credit event
    X = applications[feature_names]
    y = applications["credit_event"].values

    print(feature_names)

