## =============== import package =============== 
import pandas as pd
import numpy as np
import duckdb
import json
import os

from dagster import (
    asset, multi_asset, AssetOut, AssetIn, Output,
    TableSchema, TableColumn, MaterializeResult, 
)

from optbinning import BinningProcess

from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import chi2

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

## ===============  initial filter features =============== 
## --------- filter based on iv dan chi2 p-value dari setiap fiturnya ---------
@multi_asset(
    deps=["train_test_set"],
    outs={
        "filtered_by_iv": AssetOut(
            description="filtered train-test set based on information value dan chi2 p-value.",
            code_version="0.1",
            tags={"asset_type":"pandas-dataframe"},
            owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
        ), 
        "iv_dict": AssetOut(
            description="dictionary pairs antara signifikan fitur dengan information valuenya.",
            code_version="0.1",
            tags={"asset_type":"python-dictionary"},
            owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
        )
    },
    group_name="feature_filtering",
    compute_kind="pandas"
)
def significant_features(context):
    try: 
        train_data_df = pd.read_csv("./data/outputs/train-test-set.csv")
        applications = train_data_df.copy()
        feature_names = list(applications.columns[2:]) # exclude id dan credit event
        X = applications[feature_names]
        y = applications["credit_event"].values

        def is_dichotomic(column):
            unique_values = column.dropna().unique()
            return len(unique_values) == 2 and np.issubdtype(unique_values.dtype, np.integer)


        ## cek fitur-fitur dengan numerical values bersifat dichotomic (binary)
        dichotomic_feats = [col for col in applications.columns if col != 'credit_event' and is_dichotomic(applications[col])]

        ## define selection criteria
        selection_criteria = {
            "iv":{"min": 0.03, "max":0.7}, # information value, semakin tinggi iv mengindikasikan better predictive power. 
            # "js":{"min":0.02},# jensen-shannon divergence, semakin tinggi js values mengindikasikan fitur memiliki discriminative power yg baik.
            # "gini":{"min":0.02}, # semakin tinggi gini values mengindikasikan model discrimination yg lebih baik
            # "quality_score":{"min": 0.01}, # custom metric dari optbinning
        }

        ## fit binningprocess
        binning_process = BinningProcess(feature_names,
                                        categorical_variables=dichotomic_feats,
                                        selection_criteria=selection_criteria)
        binning_process.fit(X, y)

        ## build summary table
        summary_table = binning_process.summary()
        summary_table.sort_values(by="iv", ascending=False, inplace=True)

        ## transform original dataframe dengan selected fitur berdasarkan nilai iv ke dalam bins
        ## perform dan calculate chi2 p-value setiap fitur
        transform_selected_features = binning_process.fit_transform(X, y, metric="bins")
        X = np.array(transform_selected_features)
        Y = np.ravel(np.array(applications[["credit_event"]]))

        scores = pd.DataFrame()
        encoder = OrdinalEncoder()

        for i, feature in enumerate(transform_selected_features):
            X_feature = X[:, i].astype(str).reshape(-1, 1)
            encoder.fit(X_feature)
            X_enc = encoder.transform(X_feature)
            chi = chi2(X=X_enc, y=Y)
            new_row = pd.DataFrame({'feature': [feature],
                                    'chi2_pvalue': [chi[1][0]]})
            scores = pd.concat([scores, new_row], ignore_index=True)

        ## join summary table dari optbinning dengan scores dataframe
        ## flagging apakah fitur signifikan atau tidak berdasarkan iv dan chi2 p-value yg dimiliki oleh fitur tersebut
        join_scored = scores.merge(summary_table[["name", "n_bins", "iv", "js", "gini", "quality_score"]], right_on="name", left_on="feature", how="left")
        join_scored.drop(columns="name", inplace=True)

        ## set threshold
        iv_threshold = 0.02
        chi2_pvalue_threshold = 0.05
        join_scored['iv_threshold'] = iv_threshold
        join_scored['chi2_threshold'] = chi2_pvalue_threshold

        ## flagging
        join_scored["is_significant"] = np.where((join_scored["iv"] > join_scored["iv_threshold"]) & (join_scored["chi2_pvalue"] < join_scored["chi2_threshold"]),
                                                True, False)
        ## filter rows
        signficant_features = join_scored[join_scored["is_significant"]==True].reset_index(drop=True)

        ## store fitur-fitur signifikan ke dalam list
        ## filter fitur dataframe
        keep_features_list = list(signficant_features["feature"])
        keep_features_list.extend(["id", "credit_event"])

        applications_initial_filter = train_data_df[keep_features_list]
        applications_initial_filter.reset_index(drop=False, inplace=True) ## reset index biar id-nya jadi ke kolom lagi

        ## log dataframe
        context.log.info("number of significant features: %d", len(signficant_features))
        context.log.info("initial filtered applications: \n%s", applications_initial_filter.head())

        filtered_columns = [
            TableColumn(name=col, 
                        type=str(applications_initial_filter[col].dtype),
                        description=f"Sample value: {applications_initial_filter[col].iloc[33]}")
            for col in applications_initial_filter.columns
        ]

        filtered_shape = applications_initial_filter.shape
        context.log.info("filtered train-test set shape: %s", filtered_shape)

        yield MaterializeResult(
            asset_key="filtered_by_iv",
            metadata={
                "dagster/column_schema": TableSchema(columns=filtered_columns),
                "dagster/type": "pandas-dataframe",
                "dagster/column_count": filtered_shape[1], 
                "dagster/row_count": filtered_shape[0]
            }
        )

        ## store fitur dengan information valuenya ke dalam dictionary
        ## log dictionary
        ivs = {row["feature"]: row["iv"] for _, row in signficant_features.iterrows()}
        context.log.info(f"IV dictionary: {ivs}")

        yield MaterializeResult(
            asset_key="iv_dict",
            metadata={
                "dagster/type": "python-dictionary",
                "feature_count": len(ivs),
                "min_iv": float(min(ivs.values())),
                "max_iv": float(max(ivs.values())),
                "avg_iv": float(sum(ivs.values()) / len(ivs))
            }
        )

        return applications_initial_filter.to_csv("./data/outputs/applications-initial-filter.csv", index=False), json.dump(ivs, open("./data/outputs/information_values.json", 'w'))
    
    except Exception as e:
        context.log.error(f"An error occurred while filtering significant features: {str(e)}")
        raise e
    
# --------- filter multicollinearity features ---------
@asset(
    deps=["filtered_by_iv", "iv_dict"],
    group_name="feature_filtering",
    code_version="0.1",
    tags={"asset_type":"pandas-dataframe"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="pandas"
)
def filtered_by_multicollinearity(context):
    """
    filter out highly correlated features (>0.8) 
    """
    try:
        ## load iv dan dataset yg fiturnya sudah difilter
        ivs = json.load(open("./data/outputs/information_values.json"))
        initial_filtered_set = pd.read_csv("./data/outputs/applications-initial-filter.csv")
        context.log.info(f"Number of features before filtering: {initial_filtered_set.shape[1]}")
        
        initial_filtered = initial_filtered_set.copy()
        initial_filtered.set_index(keys="id", drop=True, inplace=True)

        exclude_cols = list(initial_filtered.select_dtypes(include=["object"]).columns)
        exclude_cols.extend(["credit_event"])

        applications_num_features = initial_filtered.drop(exclude_cols, axis=1)

        ## compute correlation matrix
        correlation_matrix = applications_num_features.corr(method="pearson") ## <-- pakai pearson karena butuh measures linear relationship
        columns = correlation_matrix.columns
        high_corr_list = []
        threshold = 0.8

        # pairwise correlations
        # cari pasangan fitur dengan korelasi yang tinggi antar fitur tersebut
        for i in range(len(columns) - 1):
            for j in range(i + 1, len(columns)):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_corr_list.append({"x": [columns[i]], "y": [columns[j]]})

        high_corr_pairs = pd.DataFrame(high_corr_list)

        ## compute correlation matrix
        ## cek pairwise correlations
        # correlation_matrix = applications_num_features.corr(method="pearson") ## <-- pakai pearson karena butuh measures linear relationship
        # threshold = 0.8

        # corr_pairs = correlation_matrix.abs().unstack()
        # high_corr = corr_pairs[(corr_pairs > threshold) & (corr_pairs < 1)]
        # high_corr_pairs = high_corr.reset_index()
        # high_corr_pairs.columns = ['x', 'y', 'correlation']
        # high_corr_pairs = high_corr_pairs.drop_duplicates()

        ## flagging fitur yang harus diremove based on komparasi nilai iv-nya
        var_to_remove = []
        for _, row in high_corr_pairs.iterrows():
            if ivs[row["x"][0]] > ivs[row["y"][0]]:
                var_to_remove.append(row["y"][0])
            else:
                var_to_remove.append(row["x"][0])

        var_to_remove = list(np.unique(var_to_remove))

        context.log.info(f"Number of features to remove due to multicollinearity: {len(var_to_remove)}")
        context.log.info(f"Features removed: {var_to_remove}")

        ## remove fitur dengan nilai iv rendah dan korelasi yg tinggi
        final_applications_features_filtered = initial_filtered_set.drop(var_to_remove, axis=1)
        
        ## log
        context.log.info("final filtered applications: \n%s", final_applications_features_filtered.head())

        filtered_columns = [
            TableColumn(name=col, 
                        type=str(final_applications_features_filtered[col].dtype),
                        description=f"Sample value: {final_applications_features_filtered[col].iloc[33]}")
            for col in final_applications_features_filtered.columns
        ]

        filtered_shape = final_applications_features_filtered.shape
        context.log.info("final filtered train-test set shape: %s", filtered_shape)

        yield MaterializeResult(
            asset_key="filtered_by_multicollinearity",
            metadata={
                "dagster/column_schema": TableSchema(columns=filtered_columns),
                "dagster/type": "pandas-dataframe",
                "dagster/column_count": filtered_shape[1], 
                "dagster/row_count": filtered_shape[0]
            }
        )

        return final_applications_features_filtered.to_csv("./data/outputs/applications-final-filter.csv", index=False)

    except Exception as e:
        context.log.error("An error occurred while filtering highly correlated features: %s", str(e))
        raise e
    

# --------- feature selection zone ---------
@asset(
    deps=["filtered_by_multicollinearity"],
    group_name="feature_selection",
    code_version="0.1",
    tags={"asset_type":"pandas-dataframe"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="pandas"
)
def transformed_applications(context):
    """
        transformed values ke woe per karakteristik fitur
    """
    try:
        applications = pd.read_csv("./data/outputs/applications-final-filter.csv")
        variable_names = list(applications.columns[:-2]) # exclude id dan credit event

        X = applications[variable_names]
        y = applications["credit_event"].values

        def is_dichotomic(column):
            unique_values = column.dropna().unique()
            return len(unique_values) == 2 and np.issubdtype(unique_values.dtype, np.integer)

        dichotomic_feats = [col for col in applications.columns if col != 'credit_event' and is_dichotomic(applications[col])]

        binning_process = BinningProcess(variable_names, categorical_variables=dichotomic_feats)
        binning_process.fit(X, y)

        summary_table = binning_process.summary()
        summary_table.sort_values(by="iv", ascending=False, inplace=True)
        transformed_table = binning_process.fit_transform(X, y, metric="woe", check_input=True)

        merged_by_index = pd.merge(transformed_table, applications[["id", "credit_event"]], left_index=True, right_index=True)

        # log the head of the final filtered dataframe
        context.log.info("transformed applications: \n%s", merged_by_index.head())

        filtered_shape = merged_by_index.shape

        filtered_columns = [
            TableColumn(
                name=col,
                type=str(merged_by_index[col].dtype),
                description=f"Sample value: {merged_by_index[col].iloc[33]}"
            )
            for col in merged_by_index.columns
        ]

        yield MaterializeResult(
            asset_key="transformed_applications",
            metadata={
                "dagster/column_schema": TableSchema(columns=filtered_columns),
                "dagster/type": "pandas-dataframe",
                "dagster/column_count": filtered_shape[1], 
                "dagster/row_count": filtered_shape[0]
            }
        )

        return merged_by_index.to_csv("./data/outputs/transformed-applications.csv", index=False)
    
    except Exception as e:
        context.log.error("An error occurred while filtering highly correlated features: %s", str(e))
        raise e