## =============== import package =============== 
import pandas as pd
import numpy as np
import duckdb

import logging
import json
import os

from dagster import (
    asset, multi_asset, AssetOut, AssetIn, Output,
    TableSchema, TableColumn, MaterializeResult, 
)

from optbinning import BinningProcess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import chi2, SelectFromModel

import matplotlib.pyplot as plt

import mlflow


# --------- konfigurasi ---------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow_tracking_uri = f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}"
mlflow.set_tracking_uri(mlflow_tracking_uri)

## =============== prep =============== 
@asset(
    group_name="initial_prep",
    tags={"asset_type":"DuckDBResource", 
          "data_source":"loan_data_2015.db"},
    code_version="0.1",
    owners=["alvinnoza.data@gmail.com", "team:Data Scientist"],
    compute_kind="duckdb",
    description="Load train-test table ke pandas dataframe"
)
def train_test_set(context):
    try:
        """
            Load dataset `train-test` dari SQL table yang sudah disiapkan sebelumnya.
            Output dari asset ini berupa CSV dan disimpan sementara ke dalam `./data/outputs/train-test-set.csv` yg nantinya akan kita gunakan untuk downstream assets lainnya.
        """
        ## load data
        query = """SELECT * FROM train_test_set"""
        conn = duckdb.connect(os.getenv("DUCKDB_DATABASE"))
        train_test_set = conn.execute(query).fetch_df()
        train_test_set.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)

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
    

@asset(
    deps=["train_test_set"],
    group_name="initial_prep",
    tags={"asset_type":"pandas-dataframe", 
          "data_source":"loan_data_2015.db"},
    code_version="0.1",
    owners=["alvinnoza.data@gmail.com", "team:Data Scientist"],
    compute_kind="pandas",
    description="Prepare train-test set"
)
def prepared_train_test(context):
    try:
        """
            Asset ini menghasilkan train-test set yang sudah diprepare sehingga dapat diconsume untuk steps selanjutnya.
            Adapun beberapa preparation yang dilakukan adalah cleaning, handle missing values, typecasting, dan ekstrak informasi dari time-related features.
        """
        ## load data
        train_test_set = pd.read_csv("./data/outputs/train-test-set.csv")
        train_test_set.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)

        ## cleaning emp_length
        def emp_length_converter(dataframe, column):
            dataframe[column] = dataframe[column].replace({
                r"\+ years": "",
                r"< 1 year": "0",
                r" years?": ""
            }, regex=True)
            
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').fillna(0).astype(int)

        ## apply
        emp_length_converter(train_test_set, "emp_length")

        ## remove whitespace
        ## coba keep sbg categorical dulu aja
        train_test_set["term"] = train_test_set["term"].str.lstrip()

        ## last_pymnt_d dan last_credit_pull_d punya missing values, padahal kita butuh convert dan hitung time difference mereka
        ## karena missingnya last_pymnt_d kurang dari 5%, rasanya masih save untuk didrop
        ## drop rows 
        train_test_set.dropna(subset=["last_pymnt_d", "last_credit_pull_d"], inplace=True)

        ## cleaning date-related columns
        def convert_and_calc_moths(df):
            ## define kolom
            date_columns = ['earliest_cr_line', 'issue_d', 'last_pymnt_d', 'last_credit_pull_d']
            
            # convert ke datetime
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], format='%b-%y', errors='coerce')
            
            ## hitung time difference antara issue date dgn earliest credit line
            df['months_since_earliest_cr_line'] = (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12 + \
                                                (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
            
            ## hitung time difference antara last payment date dng issue date
            df['months_since_issue_d_last_pymnt_d'] = (df['last_pymnt_d'].dt.year - df['issue_d'].dt.year) * 12 + \
                                                    (df['last_pymnt_d'].dt.month - df['issue_d'].dt.month)
            
            ## hitung time difference antara last credit pull date and last payment date
            df['months_since_last_pymnt_d_last_credit_pull_d'] = (df['last_credit_pull_d'].dt.year - df['last_pymnt_d'].dt.year) * 12 + \
                                                                (df['last_credit_pull_d'].dt.month - df['last_pymnt_d'].dt.month)
            
            ## hitung time difference antara issue date and last credit pull date
            df['months_since_issue_d_last_credit_pull_d'] = (df['last_credit_pull_d'].dt.year - df['issue_d'].dt.year) * 12 + \
                                                            (df['last_credit_pull_d'].dt.month - df['issue_d'].dt.month)

            ## drop kolom original
            df.drop(columns=date_columns, inplace=True)
            
            return df

        ## apply function
        train_test_set = convert_and_calc_moths(train_test_set)

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
        
        return train_test_set.to_csv("./data/outputs/prep-train-test-set.csv", index=False)

    except Exception as e:
        context.log.error("An error occurred while build database connection/load credit application table: %s", str(e))
        raise e

## ===============  initial filter features =============== 
## --------- filter based on iv dan chi2 p-value dari setiap fiturnya ---------
@multi_asset(
    deps=["prepared_train_test"],
    outs={
        "filtered_by_iv": AssetOut(
            description="Filtered train-test set based on information value dan chi2 p-value.",
            code_version="0.1",
            tags={"asset_type":"pandas-dataframe"},
            owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
        ), 
        "iv_dict": AssetOut(
            description="Dictionary pairs antara signifikan features dengan information valuenya.",
            code_version="0.1",
            tags={"asset_type":"python-dictionary"},
            owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
        )
    },
    group_name="feature_filtering",
    compute_kind="pandas"
)
def significant_features(context):
    """
    Fungsi ini menghasilkan output dua asset; `filtered_by_iv` dan `iv_dict`.
    Untuk menghitung information values, kita memanfaatkan package `optbinning` dan melakukan test chi-square.
    Information values digunakan untuk mengukur seberapa bagus sebuah fitur bisa memisahkan kreditur yg masuk ke kelas bad ataupun good.
    Chi-square test dilakukan sehingga kita bisa melihat hubungan significant antara fitur terhadap credit envent.
    Features yang ada kemudian difilter dan kita exclude fitur-fitur yang tidak dual-selection criteria.
    """
    try: 
        train_data_df = pd.read_csv("./data/outputs/prep-train-test-set.csv")
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
        # selection_criteria = {
        #     "iv":{"min": 0.02}, # information value, semakin tinggi iv mengindikasikan better predictive power. 
        #     # "js":{"min":0.02},# jensen-shannon divergence, semakin tinggi js values mengindikasikan fitur memiliki discriminative power yg baik.
        #     # "gini":{"min":0.02}, # semakin tinggi gini values mengindikasikan model discrimination yg lebih baik
        #     # "quality_score":{"min": 0.01}, # custom metric dari optbinning
        # }

        ## fit binningprocess
        # binning_process = BinningProcess(feature_names,
        #                                 categorical_variables=dichotomic_feats,
        #                                 selection_criteria=selection_criteria)
        binning_process = BinningProcess(feature_names,
                                categorical_variables=dichotomic_feats)

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
        # applications_initial_filter.reset_index(drop=False, inplace=True) ## reset index biar id-nya jadi ke kolom lagi

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
    compute_kind="pandas",
    description="Multicollinearity control"
)
def filtered_by_multicollinearity(context):
    """
    Filter multicollinearity. Kenapa penting? Karena multicollinearity bisa lead to UNSTABLE coefficient estimates di dalam credit model nantinya.
    Terlebih karena nantinya kita menggunakan model logistic regression, kontrol multicollinearity juga penting untuk menghindari variance inflation di dalam prediksi model.
    """
    try:
        ## load iv dan dataset yg fiturnya sudah difilter
        ivs = json.load(open("./data/outputs/information_values.json"))
        initial_filtered_set = pd.read_csv("./data/outputs/applications-initial-filter.csv")
        context.log.info(f"Number of features before filtering: {initial_filtered_set.shape[1]}")
        
        initial_filtered = initial_filtered_set.copy()
        initial_filtered.set_index(keys="id", drop=True, inplace=True)

        ## include numerical features aja, yg object kita exclude
        exclude_cols = list(initial_filtered.select_dtypes(include=["object"]).columns)
        exclude_cols.extend(["credit_event"])

        applications_num_features = initial_filtered.drop(exclude_cols, axis=1)

        ## compute dan anlisis correlation
        correlation_matrix = applications_num_features.corr(method="pearson") ## <-- pakai pearson karena butuh measures linear relationship
        columns = correlation_matrix.columns
        high_corr_list = []
        threshold = 0.8

        ## pairwise correlations
        ## cari pasangan fitur dengan korelasi yang tinggi antar fitur tersebut
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

        ## seleksi features
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

        ## check jumlah feature
        if len(final_applications_features_filtered.columns) < 5:
            context.log.warning("Very few features remaining after filtering!")

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
# @asset(
#     deps=["filtered_by_multicollinearity"],
#     group_name="feature_selection",
#     code_version="0.1",
#     tags={"asset_type":"pandas-dataframe"},
#     owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
#     compute_kind="pandas"
# )
# def transformed_applications(context):
#     """
#         transformed values ke woe per karakteristik fitur
#     """
#     try:
#         applications = pd.read_csv("./data/outputs/applications-final-filter.csv")
#         variable_names = list(applications.columns[:-2]) # exclude id dan credit event

#         X = applications[variable_names]
#         y = applications["credit_event"].values

#         def is_dichotomic(column):
#             unique_values = column.dropna().unique()
#             return len(unique_values) == 2 and np.issubdtype(unique_values.dtype, np.integer)

#         dichotomic_feats = [col for col in applications.columns if col != 'credit_event' and is_dichotomic(applications[col])]

#         binning_process = BinningProcess(variable_names, categorical_variables=dichotomic_feats)
#         binning_process.fit(X, y)

#         summary_table = binning_process.summary()
#         summary_table.sort_values(by="iv", ascending=False, inplace=True)
#         transformed_table = binning_process.fit_transform(X, y, metric="woe", check_input=True)

#         merged_by_index = pd.merge(transformed_table, applications[["id", "credit_event"]], left_index=True, right_index=True)

#         # log the head of the final filtered dataframe
#         context.log.info("transformed applications: \n%s", merged_by_index.head())

#         filtered_shape = merged_by_index.shape

#         filtered_columns = [
#             TableColumn(
#                 name=col,
#                 type=str(merged_by_index[col].dtype),
#                 description=f"Sample value: {merged_by_index[col].iloc[33]}"
#             )
#             for col in merged_by_index.columns
#         ]

#         yield MaterializeResult(
#             asset_key="transformed_applications",
#             metadata={
#                 "dagster/column_schema": TableSchema(columns=filtered_columns),
#                 "dagster/type": "pandas-dataframe",
#                 "dagster/column_count": filtered_shape[1], 
#                 "dagster/row_count": filtered_shape[0]
#             }
#         )

#         return merged_by_index.to_csv("./data/outputs/transformed-applications.csv", index=False)
    
#     except Exception as e:
#         context.log.error("An error occurred while filtering highly correlated features: %s", str(e))
#         raise e

@asset(
    deps=["filtered_by_multicollinearity"],
    group_name="feature_selection",
    code_version="0.1",
    tags={"asset_type":"ml-model"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="scikitlearn",
    description="RF model untuk more features selection"
)
def mlflow_rf_model():
    """
    Asset MLflow untuk feature selection dengan RandomForest.
    Random Forest model for feature selection using MLflow for experiment tracking.
    RF model bisa capture non-linear feature importance dan interaction effects.
    Kita akan keep 80% features.
    """
    try:
        ## load data 
        df = pd.read_csv("./data/outputs/applications-final-filter.csv")
        df.set_index("id", inplace=True)

        def is_dichotomic(column):
                    import numpy as np
                    unique_values = column.dropna().unique()
                    return len(unique_values) == 2 and np.issubdtype(unique_values.dtype, np.integer)

        ## baseline
        variable_names = list(df.copy().drop(labels=["credit_event"], axis=1).columns) ## <-- exclude credit event
        X = df[variable_names]
        y = df["credit_event"].values

        dichotomic_feats = [col for col in df.columns if col != 'credit_event' and is_dichotomic(df[col])]
        
        ## binning
        binning_process = BinningProcess(variable_names,
                                categorical_variables=dichotomic_feats)
        binning_process.fit(X, y)

        ## transform menjadi bin
        transformed_table = binning_process.fit_transform(X, y, metric="woe", check_input=True).reset_index(drop=False)

        to_merge = df[["credit_event"]].reset_index(drop=False)
        to_merge.drop_duplicates(subset=['id'], keep='last', inplace=True, ignore_index=True)
        transformed_applications = pd.merge(transformed_table, to_merge, left_on="id", right_on="id", how="inner") ## <-- merge by id
        transformed_applications.drop_duplicates(subset=['id'], keep='last', inplace=True, ignore_index=True)

        ## training random forest dengan transformed data
        variable_names = list(transformed_applications.copy().drop(labels=["credit_event"], axis=1).columns)
        X = transformed_applications[variable_names]
        y = transformed_applications["credit_event"].values

        ## split
        ## TODO: POTENTIAL LEAKAGE DISINI
        ## JANGAN PAKAI TRANSFORMED_DATA UNTUK SPLIT KE TRAIN-TEST
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

        ## init classifier
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }

        rf = RandomForestClassifier(random_state=777)
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=777, n_jobs=-1)

        mlflow.set_experiment("feature-selection-with-RF")

        ## run experiment
        with mlflow.start_run() as run:
            mlflow.set_tag("use_case", "credit_scorecard")
            mlflow.set_tag("dagster_zone", "feature_selection")

            ## fit model
            rf_random.fit(X_train, y_train)

            ## log setiap parameter dan masing-masing scorenya
            for i in range(len(rf_random.cv_results_['params'])):
                params = rf_random.cv_results_['params'][i]
                score = rf_random.cv_results_['mean_test_score'][i]
                mlflow.log_param(f"Params_{i}", params)
                mlflow.log_metric(f"Score_{i}", score)

            ## log parameter dan score terbaik
            best_params = rf_random.best_params_
            best_score = float(rf_random.best_score_)
            mlflow.log_params(best_params)
            mlflow.log_metric("Best_Score", best_score)

            ## extract estimator
            best_rf = rf_random.best_estimator_

            ## extract feature importances
            importances = best_rf.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            ## ambil top N features, seenggaknya 80% dari jumlah original features karena jumlah fitur dari multicollinearity filter udah sedikti
            num_features_to_select = int(0.8 * len(X.columns))
            top_features = feature_importance_df['Feature'][:num_features_to_select].values.tolist()  # Convert to list

            ## log selected features
            mlflow.log_param("Selected_Features", top_features)

            ## filter dataset dengan selected features
            X_train_selected = X_train[top_features]
            X_test_selected = X_test[top_features]

            ## retrain model dengan selected features
            best_rf.fit(X_train_selected, y_train)

            ## eval model
            y_pred = best_rf.predict(X_test_selected)
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            ## log confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title('Confusion Matrix')
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()

            ## log model
            mlflow.sklearn.log_model(best_rf, "random_forest_model")

            ## register model
            model_uri = f"runs:/{run.info.run_id}/random_forest_model"
            registered_model = mlflow.register_model(model_uri=model_uri, name="BestRandomForestModel")

            ## log feature importance plot
            plt.figure(figsize=(10, 10))
            feature_importance_df[:num_features_to_select].plot(x='Feature', y='Importance', kind='bar')
            plt.title('Top Feature Importances')
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "feature_importances.png")
            plt.close()

        ## return run ID dan selected features untuk downstream assets
        return Output(
            value={"run_id": run.info.run_id, "selected_features": top_features},
            metadata={
                "run_id": run.info.run_id,
                "num_selected_features": len(top_features),
                "selected_features":top_features,
                # "best_score": best_score,
                "registered_model_version": registered_model.version,
                "mlflow_tracking_uri": "run: 'mlflow ui --backend-store-uri sqlite:///mlflow.db', then go to http://127.0.0.1:5000"
            }
        )

    except Exception as e:
        raise RuntimeError(f"An error occurred during Random Forest feature selection: {str(e)}")
    
## ----------------- feature selection ------------------
@asset(
    deps=["filtered_by_multicollinearity"],
    ins={"mlflow_rf_model": AssetIn(key="mlflow_rf_model")},
    group_name="feature_selection",
    code_version="0.1",
    tags={"asset_type": "pandas-dataframe"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="pandas"
)
def selected_features_data(context, mlflow_rf_model):
    try:
        # load transformed woe dataframe
        df = pd.read_csv("./data/outputs/applications-final-filter.csv")
        selected_features = mlflow_rf_model["selected_features"]
        selected_features.extend(["credit_event"])

        selected_features_df = df[selected_features]
        
        context.log.info("selected features dataframe: \n%s", selected_features_df.head())

        selected_shape = selected_features_df.shape

        filtered_columns = [
            TableColumn(
                name=col,
                type=str(selected_features_df[col].dtypes),
                description=f"Sample value: {selected_features_df[col].iloc[33]}"
            )
            for col in selected_features_df.columns
        ]

        yield MaterializeResult(
            asset_key="selected_features_data",
            metadata={
                "dagster/column_schema": TableSchema(columns=filtered_columns),
                "dagster/type": "pandas-dataframe",
                "dagster/column_count": selected_shape[1], 
                "dagster/row_count": selected_shape[0]
            }
        )

        return selected_features_df.to_csv("./data/outputs/application_selected_features.csv", index=False)
    
    except Exception as e:
        context.log.error("An error occurred while selecting features: %s", str(e))
        raise e
    

# --------- model exprimentation ---------

@asset(
    deps=["selected_features_data"],
    # ins={"mlflow_rf_model": AssetIn(key="mlflow_rf_model")},
    group_name="credit_model_training",
    code_version="0.1",
    tags={"asset_type": "ml-model"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="scikit-learn"
)
def mlflow_credit_model(context):
    """
        deskripsi asset
    """
    try:
        ## load data
        df = pd.read_csv("./data/outputs/application_selected_features.csv")
        
        ## prep features dan target
        feature_names = list(df.copy().drop(labels=["credit_event", "id"], axis=1).columns)
        X = df[feature_names]
        y = df["credit_event"].values

        ## splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=666, stratify=y
        )

        binning_process = BinningProcess(feature_names)
        pipeline = Pipeline(steps=[
            ("binning_process", binning_process),
            ("estimator", LogisticRegression(random_state=666))
        ])

        param_grid = {
            'estimator__penalty': ['l1', 'l2', 'none'],
            'estimator__C': [0.01, 0.1, 1, 10],
            'estimator__class_weight': ['balanced'],
            'estimator__solver': ['liblinear', 'saga'],
        }

        param_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc"
        )

        mlflow.set_experiment("credit-model-logreg")

        ## run experiment
        with mlflow.start_run() as run:
            mlflow.set_tag("use_case", "credit_scorecard")
            mlflow.set_tag("dagster_zone", "credit_model_training")

            ## fit model
            param_search.fit(X_train, y_train)

            ## log setiap parameter dan masing-masing scorenya
            for i in range(len(param_search.cv_results_["params"])):
                params = param_search.cv_results_["params"][i]
                score = param_search.cv_results_["mean_test_score"][i]
                mlflow.log_param(f"Params_{i}", params)
                mlflow.log_metric(f"Score_{i}", score)

            ## log parameter dan score terbaik
            best_params = param_search.best_params_
            best_score = float(param_search.best_score_)
            mlflow.log_params(best_params)
            mlflow.log_metric("Best_Score", best_score)

            ## extract estimator
            best_logreg = param_search.best_estimator_

            ## extract coefficients
            coefs = param_search.best_estimator_.named_steps.estimator.coef_
            coefs_df = pd.DataFrame(zip(feature_names, coefs))

            ## eval, predict di test data
            y_pred = best_logreg.predict(X_test)
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            ## confusion matrix
            plt.figure(figsize=(10,10))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title("Best LogReg's Confusion Matrix on Test Data")
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

            ## log model
            mlflow.sklearn.log_model(best_logreg, "trained_credit_logreg")

            ## register
            model_uri = f"runs:/{run.info.run_id}/trained_credit_logreg"
            register_model = mlflow.register_model(model_uri=model_uri, name="BestLogRegModel")
        
            ## log input
            # mlflow.log_input(df, context="training")

            ## materialisasi koefisien df
            ## TODO: transpose coefs_df dulu lalu materialize 

        return Output(
            value={"run_id":run.info.run_id},
            metadata={
                "coefficients":coefs_df,
                "registered_model_version": register_model.version,
                "mlflow_tracking_uri": "run: 'mlflow ui --backend-store-uri sqlite:///mlflow.db', then go to http://127.0.0.1:5000"
            }
        )
    
    except Exception as e:
        logger.error("An error occurred while filtering highly correlated features: %s", str(e))
        raise e
    
@asset(
    group_name="credit_model_training"
)
def model_coefficients(mlflow_credit_model):
    """
        coeffients dari registered logistic regression model untuk membentuk scorecard
    """
    try:
        empty_list = []
        return empty_list
    
    except Exception as e:
        logger.error("An error occurred while filtering highly correlated features: %s", str(e))
        raise e


@asset(
    group_name="credit_model_training"
)
def score_card(model_coefficients):
    """
        credit scorecard 
    """
    try:
        empty_list = []
        return empty_list
    
    except Exception as e:
        logger.error("An error occurred while filtering highly correlated features: %s", str(e))
        raise e
    