## =============== IMPORT PACKAGE =============== 
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
from optbinning.scorecard import Scorecard, plot_auc_roc, plot_cap, plot_ks, ScorecardMonitoring

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt

import mlflow
from mlflow.models import infer_signature

from . procedures import calc_credit_metrics, is_dichotomic

# =============== CONFIG =============== 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow_tracking_uri = f"sqlite:///{os.path.join(os.getcwd(), 'mlflow.db')}"
mlflow.set_tracking_uri(mlflow_tracking_uri)

## =============== ASET2 INITIAL PREP =============== 
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

        conn.close()

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

## =============== ASET2 FILTER FEATURES =============== 
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

        ## cek fitur-fitur dengan numerical values bersifat dichotomic (binary)
        dichotomic_feats = [col for col in applications.columns if col != 'credit_event' and is_dichotomic(applications[col])]

        ## fit proses binning
        binning_process = BinningProcess(feature_names,
                                categorical_variables=dichotomic_feats)

        binning_process.fit(X, y)

        ## transform ke summary table
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
    
## =============== ASET2 FEATURE SELECTION =============== 
@asset(
    deps=["filtered_by_multicollinearity"],
    group_name="feature_selection",
    code_version="0.2",
    tags={"asset_type":"ml-model"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="scikitlearn",
    description="Feature selection dengan Random Forest"
)
def mlflow_rf_model(context):
    """
    Asset MLflow untuk feature selection dengan RandomForest.
    Random Forest model for feature selection using MLflow for experiment tracking.
    RF model bisa capture non-linear feature importance dan interaction effects.
    Kita akan keep 80% original features.
    """
    try:
        ## load data 
        df = pd.read_csv("./data/outputs/applications-final-filter.csv")
        # df.set_index("id", inplace=True)

        ## baseline
        feature_names = list(df.copy().drop(labels=["credit_event", "id"], axis=1).columns) ## <-- exclude credit event
        X = df[feature_names]
        y = df["credit_event"].values

        dichotomic_feats = [col for col in df.columns if col != 'credit_event' and is_dichotomic(df[col])]
        
        ## spliting
        X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
            X, y, test_size=0.25, 
            random_state=777, 
            stratify=y
        )

        ## binning
        binning_process = BinningProcess(feature_names, 
                                         categorical_variables=dichotomic_feats)
        pipeline = Pipeline(steps=[
                    ("binning_process", binning_process),
                    ("classifier", RandomForestClassifier(random_state=777))
                ])
        
        ## set model parameter
        params = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_features": ["sqrt"],
            "classifier__max_depth": [10, None],
            "classifier__min_samples_split": [5],
            "classifier__min_samples_leaf": [2],
            "classifier__bootstrap": [True],
            "classifier__class_weight":["balanced"]
        }

        ## pakai stratified kfold supaya distribusi target tetap terjaga
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

        rf_random = RandomizedSearchCV(
            estimator=pipeline, 
            param_distributions=params, 
            n_iter=5, 
            cv=cv, 
            verbose=2, 
            random_state=777, 
            scoring="roc_auc"
        )

        ## train  model
        rf_random.fit(X_train_val, y_train_val)

        ## extract best parameter dan dan best score dari hasil cv
        best_params = rf_random.best_params_
        best_score = float(rf_random.best_score_)

        ## extract estimator
        best_rf = rf_random.best_estimator_

        ## extract feature importances
        importances = best_rf.named_steps['classifier'].feature_importances_
        feature_names = X_train_val.columns.tolist()
        feature_importance_df = pd.DataFrame({'Feature': feature_names,'Importance': importances}).sort_values(by='Importance', ascending=False)

        ## ambil top N features, seenggaknya 80% dari jumlah original features karena output jumlah fitur dari multicollinearity filter udah cukup sedikit
        num_features_to_select = int(0.8 * len(feature_names))
        top_features = feature_importance_df['Feature'][:num_features_to_select].values.tolist()  ## convert ke list

        ## filter dataset dengan selected features
        X_train_selected = X_train_val[top_features]

        ## binning dan training model ulang untuk cek performa pada features yg sudah difilter
        binning_process_sec = BinningProcess(top_features)
        selected_rf = Pipeline([
            ("binning_process", binning_process_sec),
            ("classifier", RandomForestClassifier(**best_rf.named_steps['classifier'].get_params()))
        ])

        ## re-train model
        selected_rf.fit(X_train_selected, y_train_val)

        ## predict dan eval model
        y_pred = selected_rf.predict(X_holdout[top_features])
        y_prob = selected_rf.predict_proba(X_holdout[top_features])[:, 1]

        mlflow.set_experiment("feature-selection-with-RF")

        ## log model, metadata, dan performance metricsnya ke mlflow
        with mlflow.start_run() as run:

            ## set tag
            mlflow.set_tag("use_case", "credit_scorecard")
            mlflow.set_tag("dagster_zone", "feature_selection")

            ## log setiap parameter dan masing-masing scorenya
            for i in range(len(rf_random.cv_results_['params'])):
                params = rf_random.cv_results_['params'][i]
                score = rf_random.cv_results_['mean_test_score'][i]
                mlflow.log_param(f"Params_{i}", params)
                mlflow.log_metric(f"Score_{i}", score)

            ## log parameter dan score terbaik
            mlflow.log_params(best_params)
            mlflow.log_metric("Best_Score", best_score)

            ## log selected features
            mlflow.log_param("Selected_Features", top_features)

            ## calculate & log performance metrics 
            credit_metrics = calc_credit_metrics(y_holdout, y_pred, y_prob)
            for metric_name, metric_value in credit_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            if credit_metrics['gini_coefficient'] < 0.3:
                context.log.warning(f"Low Gini coefficient: {credit_metrics['gini_coefficient']:.3f}")
            if credit_metrics['ks_statistic'] < 0.3:
                context.log.warning(f"Low KS statistic: {credit_metrics['ks_statistic']:.3f}")

            ## log classification report 
            report = classification_report(y_holdout, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            ## log confusion matrix
            cm = confusion_matrix(y_holdout, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title('Confusion Matrix')
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()

            ## log model
            # signature = infer_signature(X_train_selected, selected_rf.predict(X_train_selected))
            mlflow.sklearn.log_model(selected_rf, "random_forest_model")
            # mlflow.sklearn.log_model(
            #     sk_model=selected_rf,
            #     registered_model_name="random_forest_model",
            #     input_example=X_train_selected,
            #     signature=signature,
            #     artifact_path="BestRandomForestModel"
            # )

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
                "num_selected_features": len(top_features),
                "selected_features":top_features,
                # "gini_coefficient": float(credit_metrics['gini_coefficient']),
                # "ks_statistic": float(credit_metrics['ks_statistic']),
                "precision_bad_rate": float(credit_metrics['precision_bad_rate']),
                "recall_bad_rate": float(credit_metrics['recall_bad_rate']),
                # "best_score": best_score,
                "registered_model_version": registered_model.version,
                "mlflow_tracking_uri": "run: 'mlflow ui --backend-store-uri sqlite:///mlflow.db', then go to http://127.0.0.1:5000"
            }
        )

    except Exception as e:
        raise RuntimeError(f"An error occurred during Random Forest feature selection: {str(e)}")


@asset(
    deps=["filtered_by_multicollinearity"],
    ins={"mlflow_rf_model": AssetIn(key="mlflow_rf_model")},
    group_name="feature_selection",
    code_version="0.1",
    tags={"asset_type": "pandas-dataframe"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="pandas",
    description="Filtered train-test set based on feature selection"
)
def selected_features_data(context, mlflow_rf_model):
    """
        Filter by selected features.
    """
    try:
        ## load data, filter by selected features
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
    

## =============== ASET2 BUILD SCORECARD =============== 

@asset(
    deps=["selected_features_data"],
    # ins={"mlflow_rf_model": AssetIn(key="mlflow_rf_model")},
    group_name="build_scorecard",
    code_version="0.1",
    tags={"asset_type": "ml-model"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="scikit-learn",
    description="Experiment Tracking LogReg Model di MLflow"
)
def mlflow_credit_model(context):
    """
        Logistic Regression model untuk scorecard.
    """
    try:
        ## load data
        df = pd.read_csv("./data/outputs/application_selected_features.csv")
        
        ## prep features dan target
        feature_names = list(df.copy().drop(labels=["credit_event"], axis=1).columns)
        X = df[feature_names]
        y = df["credit_event"].values

        ## splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, 
            random_state=777, 
            stratify=y
        )

        binning_process = BinningProcess(feature_names)
        pipeline = Pipeline(steps=[
            ("binning_process", binning_process),
            ("estimator", LogisticRegression(random_state=666))
        ])

        param_grid = {
            'estimator__penalty': ['l1', 'l2'],
            'estimator__C': [0.1, 1, 10],
            'estimator__class_weight': ['balanced'],
            'estimator__solver': ['liblinear'],
        }

        ## pakai stratified kfold supaya distribusi target tetap terjaga
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

        param_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
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
            # coefs = param_search.best_estimator_.named_steps.estimator.coef_
            # coefs_df = pd.DataFrame(zip(feature_names, coefs))

            # coefs_dict = {
            # 'features': coefs_df['Feature'].tolist(),
            # 'coefficients': coefs_df['Coefficient'].tolist()
            # }
                
            ## eval, predict di test data
            y_pred = best_logreg.predict(X_test)
            y_proba = best_logreg.predict_proba(X_test)[:, 1]
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            ## eval metrics
            roc_auc = roc_auc_score(y_test, y_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

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

        return Output(
            value={"run_id":run.info.run_id},
            metadata={
                # "coefficients":coefs_dict,
                "roc-auc":float(roc_auc),
                "precision":float(precision),
                "recall":float(recall),
                "best cv score":float(best_score),
                "model parameters":best_params,
                "registered_model_version": register_model.version,
                "mlflow_tracking_uri": "run: 'mlflow ui --backend-store-uri sqlite:///mlflow.db', then go to http://127.0.0.1:5000"
            }
        )
    
    except Exception as e:
        logger.error("An error occurred while filtering highly correlated features: %s", str(e))
        raise e

@asset(
    ins={"mlflow_credit_model": AssetIn(key="mlflow_credit_model")},
    group_name="build_scorecard",
    code_version="0.1",
    tags={"asset_type": "scorecard-model"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="pandas",
    description="Generate scorecard dengan OptBinning & LogReg Model"
)
def score_card(context, mlflow_credit_model):
    """
    Creates a credit scorecard using OptBinning's Scorecard class.
    The scorecard converts model predictions into easily interpretable point values.
    """
    try:
        ## load modl dan dataset
        model_run_id = mlflow_credit_model["run_id"]
        model = mlflow.sklearn.load_model(f"runs:/{model_run_id}/trained_credit_logreg")
        df = pd.read_csv("./data/outputs/application_selected_features.csv")
        
        feature_names = list(df.drop(["credit_event"], axis=1).columns)
        X = df[feature_names]
        y = df["credit_event"].values
        
        ## extract components dari model logreg model
        estimator = model.named_steps['estimator']
        binning_process = model.named_steps['binning_process']
        
        ## init scorecard
        scorecard = Scorecard(
            binning_process=binning_process,
            estimator=estimator,
            scaling_method="pdo_odds",
            scaling_method_params={
                "pdo": 20,
                "odds": 50,
                "scorecard_points": 600
            },
            intercept_based=True,
            reverse_scorecard=False,
            rounding=True
        )
        
        ## fit scorecard
        scorecard.fit(X, y)
        
        ## generate table scorecard
        summary_table = scorecard.table(style='summary')
        detailed_table = scorecard.table(style='detailed')
        
        ## export table
        summary_table.to_csv("./data/outputs/scorecard_summary.csv", index=False)
        detailed_table.to_csv("./data/outputs/scorecard_detailed.csv", index=False)
        
        ## log scorecard info
        context.log.info("\nScorecard Summary:")
        context.log.info(summary_table)
        
        ## calc statistik score
        scores = scorecard.score(X)
        score_stats = {
            "min_score": int(np.min(scores)),
            "max_score": int(np.max(scores)),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "std_score": float(np.std(scores))
        }
        
        ## simpan object scorecard
        scorecard.save("./data/outputs/objects/scorecard.pkl")
        
        ## generate plots performance scorecard dan export ke png
        y_pred_proba = scorecard.predict_proba(X)[:, 1]
        
        for plot_type, plot_func in [
            ("roc", plot_auc_roc), 
            ("cap", plot_cap), 
            ("ks", plot_ks)
        ]:
            plt.figure(figsize=(10, 6))
            plot_func(y, y_pred_proba, title=f"{plot_type.upper()} Curve - Scorecard Performance")
            plt.tight_layout()
            plt.savefig(f"./data/outputs/plots/scorecard_{plot_type}.png")
            plt.close()
        
        ## log ke MLflow
        with mlflow.start_run(run_id=model_run_id):
            mlflow.log_artifacts("./data/outputs", "scorecard_artifacts")
            mlflow.log_params({
                "scorecard_pdo": 20,
                "scorecard_base_odds": 50,
                "scorecard_base_points": 600
            })
            mlflow.log_metrics(score_stats)
        
        return Output(
            value=scorecard,
            metadata={
                "feature_count": len(feature_names),
                "score_range": f"{score_stats['min_score']} to {score_stats['max_score']}",
                "mean_score": score_stats['mean_score'],
                "median_score": score_stats['median_score'],
                "std_score": score_stats['std_score'],
                "scaling_method": "pdo_odds",
                "pdo": 20,
                "base_score": 600,
                "base_odds": 50,
                "artifacts": {
                    "summary_table": "./data/outputs/scorecard_summary.csv",
                    "detailed_table": "./data/outputs/scorecard_detailed.csv",
                    "scorecard_object": "./data/outputs/scorecard.pkl",
                    "performance_plots": [
                        "./data/outputs/plots/scorecard_roc.png",
                        "./data/outputs/plots/scorecard_cap.png",
                        "./data/outputs/plots/scorecard_ks.png"
                    ]
                }
            }
        )
    
    except Exception as e:
        context.log.error(f"An error occurred while generating the scorecard: {str(e)}")
        raise e


@asset(
    group_name="scorecard_validation",
    tags={"asset_type":"DuckDBResource", 
          "data_source":"loan_data_2015.db"},
    code_version="0.1",
    owners=["alvinnoza.data@gmail.com", "team:Data Scientist"],
    compute_kind="duckdb",
    description="Load validation set table ke pandas dataframe"
)
def validation_set(context):
    try:
        """
        """
        ## load data
        query = """SELECT * FROM validation_set"""
        conn = duckdb.connect(os.getenv("DUCKDB_DATABASE"))
        validation_set = conn.execute(query).fetch_df()
        validation_set.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)

        conn.close()

        ## log dataframe
        context.log.info("train-test shape: %s", validation_set.shape)
        context.log.info("first five rows: \n%s", validation_set.head())

        ## materialisasi schema
        columns = [
            TableColumn(name=col, type=str(validation_set[col].dtype),
                        description=f"Sample value: {validation_set[col].iloc[33]}")
            for col in validation_set.columns
        ]

        n_rows = validation_set.shape[0]
        n_cols = validation_set.shape[1]

        yield MaterializeResult(
                metadata={
                    "dagster/column_schema": TableSchema(columns=columns),
                    "dagster/type": "pandas-dataframe",
                    "dagster/column_count": n_cols,
                    "dagster/row_count": n_rows
                }
            )
        
        return validation_set.to_csv("./data/outputs/validation-set.csv", index=False)

    except Exception as e:
        context.log.error("An error occurred while build database connection/load credit application table: %s", str(e))
        raise e

@asset(
    deps=["validation_set"],
    group_name="scorecard_validation",
    tags={"asset_type":"pandas-dataframe", 
          "data_source":"loan_data_2015.db"},
    code_version="0.1",
    owners=["alvinnoza.data@gmail.com", "team:Data Scientist"],
    compute_kind="pandas",
    description="Prepare validation set"
)
def prepared_validation(context):
    try:
        """
        """
        ## load data
        validation_set = pd.read_csv("./data/outputs/validation-set.csv")
        validation_set.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)

        ## cleaning emp_length
        def emp_length_converter(dataframe, column):
            dataframe[column] = dataframe[column].replace({
                r"\+ years": "",
                r"< 1 year": "0",
                r" years?": ""
            }, regex=True)
            
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').fillna(0).astype(int)

        ## apply
        emp_length_converter(validation_set, "emp_length")

        ## remove whitespace
        ## coba keep sbg categorical dulu aja
        validation_set["term"] = validation_set["term"].str.lstrip()

        ## last_pymnt_d dan last_credit_pull_d punya missing values, padahal kita butuh convert dan hitung time difference mereka
        ## karena missingnya last_pymnt_d kurang dari 5%, rasanya masih save untuk didrop
        ## drop rows 
        validation_set.dropna(subset=["last_pymnt_d", "last_credit_pull_d"], inplace=True)

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
        validation_set = convert_and_calc_moths(validation_set)

        ## filter features
        ## ambil features dari latest selected features untuk train-test
        app_selected_feats = pd.read_csv("./data/outputs/application_selected_features.csv")
        selected_features = app_selected_feats.columns.tolist()
        print(selected_features)
        # selected_features.extend(["credit_event"])
        if "credit_event" not in selected_features:
            selected_features.extend(["credit_event"])

        validation_set_filtered = validation_set[selected_features]

        ## log dataframe
        context.log.info("validation shape: %s", validation_set_filtered.shape)
        context.log.info("first five rows: \n%s", validation_set_filtered.head())

        ## materialisasi schema
        columns = [
            TableColumn(name=col, type=str(validation_set_filtered[col].dtypes),
                        description=f"Sample value: {validation_set_filtered[col].iloc[33]}")
            for col in validation_set_filtered.columns
        ]

        n_rows = validation_set_filtered.shape[0]
        n_cols = validation_set_filtered.shape[1]

        yield MaterializeResult(
                metadata={
                    "dagster/column_schema": TableSchema(columns=columns),
                    "dagster/type": "pandas-dataframe",
                    "dagster/column_count": n_cols,
                    "dagster/row_count": n_rows
                }
            )
        
        return validation_set_filtered.to_csv("./data/outputs/prep-validation-set.csv", index=False)

    except Exception as e:
        context.log.error("An error occurred while build database connection/load credit application table: %s", str(e))
        raise e


@asset(
    deps=["prepared_validation", "score_card"],
    group_name="scorecard_validation",
    code_version="0.1",
    tags={"asset_type": "model-validation"},
    owners=["alvinnoza.data@gmail.com", "team:data-scientist"],
    compute_kind="pandas",
    description="Validate scorecard using OptBinning's ScorecardMonitoring"
)
def scorecard_validation(context):
    """
    Validates the scorecard using OptBinning's ScorecardMonitoring class.
    Includes:
    - Population Stability Analysis
    - Variable-level PSI Analysis
    - Statistical Tests
    - Performance Metrics
    """
    try:
        ## load train-val data
        train_df = pd.read_csv("./data/outputs/application_selected_features.csv")
        val_df = pd.read_csv("./data/outputs/prep-validation-set.csv")
        score_card = Scorecard.load("./data/outputs/objects/scorecard.pkl")
        
        ## get features & targets
        features = list(train_df.drop(['credit_event'], axis=1).columns)
        X_train = train_df[features]
        y_train = train_df['credit_event']
        X_val = val_df[features]
        y_val = val_df['credit_event']
        
        ## init monitoring
        monitor = ScorecardMonitoring(
            scorecard=score_card,
            psi_method="cart",
            psi_n_bins=20,
            psi_min_bin_size=0.05,
            show_digits=2
        )
        
        ## fit monitoring with actual & expected data
        monitor.fit(
            X_actual=X_val,
            y_actual=y_val,
            X_expected=X_train,
            y_expected=y_train
        )
        
        ## get reports
        psi_table = monitor.psi_table()  
        var_psi_summary = monitor.psi_variable_table(style="summary")
        var_psi_detailed = monitor.psi_variable_table(style="detailed")
        stat_tests = monitor.tests_table()
        # stability_report = monitor.system_stability_report()

        ## buat validation folder kalau belum exist
        os.makedirs("./data/outputs/validation", exist_ok=True)
        
        ## save reports
        psi_table.to_csv("./data/outputs/validation/psi_table.csv", index=False)
        var_psi_summary.to_csv("./data/outputs/validation/variable_psi_summary.csv", index=False)
        var_psi_detailed.to_csv("./data/outputs/validation/variable_psi_detailed.csv", index=False)
        stat_tests.to_csv("./data/outputs/validation/statistical_tests.csv", index=False)
        
        ## generate classification report
        y_pred = score_card.predict(X_val)
        class_report = classification_report(y_val, y_pred)
        with open("./data/outputs/validation/classification_report.txt", "w") as f:
            f.write(class_report)
        
        ## generate confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Validation Set Confusion Matrix')
        plt.savefig("./data/outputs/validation/confusion_matrix.png")
        plt.close()
        
        ## PSI plot
        plt.figure(figsize=(10, 6))
        monitor.psi_plot()
        plt.savefig("./data/outputs/validation/psi_plot.png")
        plt.close()
        
        ## calculate scores & performance metrics
        train_scores = score_card.score(X_train)
        val_scores = score_card.score(X_val)
        val_proba = score_card.predict_proba(X_val)[:, 1]
        
        ## performance plots
        plt.figure(figsize=(10, 6))
        plot_auc_roc(y_val, val_proba, title="Validation ROC Curve")
        plt.savefig("./data/outputs/validation/val_roc.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plot_ks(y_val, val_proba, title="Validation KS Plot")
        plt.savefig("./data/outputs/validation/val_ks.png")
        plt.close()
        
        ## calculate metrics
        metrics = {
            "val_roc_auc": roc_auc_score(y_val, val_proba),
            "val_precision": precision_score(y_val, score_card.predict(X_val)),
            "val_recall": recall_score(y_val, score_card.predict(X_val)),
            "train_mean_score": np.mean(train_scores),
            "val_mean_score": np.mean(val_scores),
            "score_diff": np.mean(train_scores) - np.mean(val_scores)
        }

        ## buat experiment baru kalau belum ada
        exp_name = "scorecard-validation"
        try:
            mlflow.create_experiment(exp_name)
        except Exception:
            pass
        
        mlflow.set_experiment(exp_name)
        
        ## total PSI
        total_psi = float(psi_table.iloc[-1]["PSI"]) if "PSI" in psi_table.columns else None

        ## log ke MLflow
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_artifacts("./data/outputs/validation", "validation_artifacts")
            
            if total_psi is not None:
                mlflow.log_metric("system_psi", total_psi)
            
        ## calculate additional classification metrics
        confusion_mat = confusion_matrix(y_val, y_pred).tolist()
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        
        ## add confusion matrix based metrics
        additional_metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "neg_pred_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        }
        metrics.update(additional_metrics)

        return Output(
            value={
                "metrics": metrics,
                "psi_table": psi_table,
                "variable_psi_summary": var_psi_summary,
                "variable_psi_detailed": var_psi_detailed,
                "statistical_tests": stat_tests,
                "confusion_matrix": confusion_mat,
                "classification_report": class_report
            },
            metadata={
                "roc_auc": float(metrics["val_roc_auc"]),
                "score_difference": float(metrics["score_diff"]),
                "system_psi": total_psi,
                "artifacts": {
                    "psi_table": "./data/outputs/validation/psi_table.csv",
                    "variable_psi_summary": "./data/outputs/validation/variable_psi_summary.csv", 
                    "variable_psi_detailed": "./data/outputs/validation/variable_psi_detailed.csv",
                    "statistical_tests": "./data/outputs/validation/statistical_tests.csv",
                    "classification_report": "./data/outputs/validation/classification_report.txt",
                    "plots": {
                        "psi_plot": "./data/outputs/validation/psi_plot.png",
                        "roc_curve": "./data/outputs/validation/val_roc.png",
                        "ks_plot": "./data/outputs/validation/val_ks.png",
                        "confusion_matrix": "./data/outputs/validation/confusion_matrix.png"
                    },
                    "confusion_matrix_values": {
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "true_positives": int(tp)
                    }
                }
            }
        )
        
    except Exception as e:
        context.log.error(f"An error occurred during scorecard validation: {str(e)}")
        raise e