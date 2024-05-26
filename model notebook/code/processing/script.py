# | filename: script.py
# | code-line-numbers: true
import warnings
warnings.filterwarnings('ignore')
import os
import tarfile
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer 


def preprocess(base_directory):

    df = _read_data_from_input_csv_files(base_directory)
    
    df = convert_to_lowercase(df)  
    df = handle_missing_values(df)                         
    df = extract_and_transform_features(df)                
    df = adjust_numeric_features(df)                       
    df = add_new_feature(df)                               
    df = drop_column(df)          

    # Define the order of categories for ordinal encoding
    pre_approved_offer_categories = [['no', 'yes']]

# Define the target transformer
    target_transformer = ColumnTransformer(
    transformers=[("pre_approved_offer", OrdinalEncoder(categories=pre_approved_offer_categories), [0])]
    )

# Define the numeric transformer
    numeric_transformer = make_pipeline(
    FunctionTransformer(np.log1p)
    )

    # Define the categories for ordinal encoding
    grade_categories = [['a', 'b', 'c', 'd', 'e', 'f', 'g']]
    revol_util_bins_categories = [['0-50', '50-75', '75+']]
    home_ownership_categories = [['own', 'mortgage', 'rent']]


# Define the ordinal encoders
    ordinal_grade_transformer = OrdinalEncoder(categories=grade_categories)
    ordinal_revol_util_transformer = OrdinalEncoder(categories=revol_util_bins_categories)
    ordinal_home_ownership_transformer = OrdinalEncoder(categories=home_ownership_categories)

    #Define the one-hot encoder for the 'pincode' variable
    one_hot_encoder = OneHotEncoder()

    
    
        # Define the categorical transformer pipeline
    categorical_transformer = ColumnTransformer(
        transformers=[
            ('grade', ordinal_grade_transformer, ['grade']),
            ('revol_util_bins', ordinal_revol_util_transformer, ['revol_util_bins']),
            ('home_ownership', ordinal_home_ownership_transformer, ['home_ownership']),
            ('pincode', one_hot_encoder, ['pincode'])
                ]
        )


# Define the column transformer
    feature_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, ['loan_amnt', 'installment', 'annual_inc', 'dti', 'revol_bal']),
            ('categorical', categorical_transformer, ['grade', 'revol_util_bins', 'home_ownership', 'pincode'])
        ],
        remainder='passthrough'  # Pass through the remaining columns unchanged
    )

    df_train, df_validation, df_test = _split_data(df)

    _save_train_baseline(base_directory, df_train)
    _save_test_baseline(base_directory, df_test)

    y_train = target_transformer.fit_transform(
        np.array(df_train.pre_approved_offer.values).reshape(-1, 1),
    )
    y_validation = target_transformer.transform(
        np.array(df_validation.pre_approved_offer.values).reshape(-1, 1),
    )
    y_test = target_transformer.transform(
        np.array(df_test.pre_approved_offer.values).reshape(-1, 1),
    )

    df_train = df_train.drop("pre_approved_offer", axis=1)
    df_validation = df_validation.drop("pre_approved_offer", axis=1)
    df_test = df_test.drop("pre_approved_offer", axis=1)

    X_train = feature_transformer.fit_transform(df_train)  # noqa: N806
    X_validation = feature_transformer.transform(df_validation)  # noqa: N806
    X_test = feature_transformer.transform(df_test)  # noqa: N806

    _save_splits(
        base_directory,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )
    _save_model(base_directory, target_transformer, feature_transformer)
    
    
def convert_to_lowercase(df):
    return df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def extract_and_transform_features(df):
    # Extract pincode if needed
    df['pincode'] = df['address'].str.extract(r'(\d{5})$')
    
    df['term'] = df['term'].astype(str).str.strip().map({'36 months': 3, '60 months': 5}).astype(int)
    
    df['annual_inc'] = np.ceil(df['annual_inc'] / 12)
    df['revol_bal'] = np.ceil(df['revol_bal'] * 0.05)
    
    return df

def handle_missing_values(df):
    df['revol_util'].fillna(df['revol_util'].mean(), inplace=True)
    df['mort_acc'].fillna(0.0, inplace=True)
    df['home_ownership'].replace(['other', 'none', 'any'], np.nan, inplace=True)
    df.dropna(subset=['pub_rec_bankruptcies', 'home_ownership'], inplace=True)
    
    return df

def adjust_numeric_features(df):
    df['pub_rec'] = df['pub_rec'].apply(lambda x: 0 if x < 1.0 else 1)
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].apply(lambda x: 0 if x == 0.0 else 1)
    df['mort_acc'] = df['mort_acc'].apply(lambda x: 1 if x > 0.0 else 0)
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x == 'fully paid' else 0)
    df['verification_status'] = df['verification_status'].apply(lambda x: 1 if x in ['source verified', 'verified'] else 0)
    df['dti'] = df['dti'].apply(lambda x: x if x < 60 else 60)
    
    return df

def add_new_feature(df):
    bins = [0, 50, 75, float('inf')]
    bin_labels = ['0-50', '50-75', '75+']

    # Cut 'revol_util' values into bins
    df['revol_util_bins'] = pd.cut(df['revol_util'], bins=bins, labels=bin_labels, right=False)

    def calculate_pre_approved_offer(row):
        if row['loan_status'] == 1:
            if row['home_ownership'] in ['mortgage', 'own'] and row['grade'] in ['a', 'b'] and row['verification_status'] == 1 and row['pub_rec'] == 0 and row['mort_acc'] == 1 and row['pub_rec_bankruptcies'] == 0 and row['revol_util_bins'] in ['0-50', '50-75']:
                foir = 50
            elif row['home_ownership'] in ['mortgage', 'own', 'rent'] and row['grade'] in ['a', 'b', 'c', 'd'] and row['verification_status'] in [0, 1] and row['pub_rec'] == 0 and row['revol_util_bins'] in ['0-50', '50-75']:
                foir = 40
            elif row['home_ownership'] in ['mortgage', 'own', 'rent'] and row['grade'] in ['e', 'f', 'g'] and row['verification_status'] in [0, 1] and row['pub_rec'] == 0 and row['revol_util_bins'] in ['0-50', '50-75']:
                foir = 30
            else:
                return 'no'  # If none of the conditions are met, Pre-approved offer is 0

            new_emi = (row['annual_inc'] * (foir / 100)) - row['revol_bal'] - ((row['annual_inc'] * (row['dti'] / 100)) / 2)
            return 'yes' if new_emi > (row['installment'] * 1.25) else 'no'

        else:
            return 'no'

    # Apply the function to create the 'pre_approved_offer' column
    df['pre_approved_offer'] = df.apply(calculate_pre_approved_offer, axis=1)

    return df

    
def drop_column(df):

    """
    Drops unnecessary columns from the DataFrame.

    """
    df.drop(['revol_util','address','issue_d','purpose','title','earliest_cr_line','initial_list_status','int_rate','application_type','total_acc','emp_title','open_acc','emp_length','sub_grade'],axis = 1,inplace=True)

    return df

def _read_data_from_input_csv_files(base_directory):
    """Read the data from the input CSV files.

    This function reads every CSV file available and
    concatenates them into a single dataframe.
    """
    input_directory = Path(base_directory) 
    files = list(input_directory.glob("*.csv"))

    if len(files) == 0:
        message = f"The are no CSV files in {input_directory.as_posix()}/"
        raise ValueError(message)

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)

    # Shuffle the data
    return df.sample(frac=1, random_state=42)


def _split_data(df):
    """Split the data into train, validation, and test."""
    df_train, temp = train_test_split(df, test_size=0.3)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    return df_train, df_validation, df_test

def _save_train_baseline(base_directory, df_train):
    """Save the untransformed training data to disk.

    We will need the training data to compute a baseline to
    determine the quality of the data that the model receives
    when deployed.
    """
    baseline_path = Path(base_directory) / "train-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = df_train.copy().dropna()

    # To compute the data quality baseline, we don't need the
    # target variable, so we'll drop it from the dataframe.
    df = df.drop("pre_approved_offer", axis=1)

    df.to_csv(baseline_path / "train-baseline.csv", header=True, index=False)

def _save_test_baseline(base_directory, df_test):
    """Save the untransformed test data to disk.

    We will need the test data to compute a baseline to
    determine the quality of the model predictions when deployed.
    """
    baseline_path = Path(base_directory) / "test-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = df_test.copy().dropna()

    # We'll use the test baseline to generate predictions later,
    # and we can't have a header line because the model won't be
    # able to make a prediction for it.
    df.to_csv(baseline_path / "test-baseline.csv", header=False, index=False)


def _save_splits(
    base_directory,
    X_train,  # noqa: N803
    y_train,
    X_validation,  # noqa: N803
    y_validation,
    X_test,  # noqa: N803
    y_test,
):
    """Save data splits to disk.

    This function concatenates the transformed features
    and the target variable, and saves each one of the split
    sets to disk.
    """
    train = np.concatenate((X_train, y_train), axis=1)
    validation = np.concatenate((X_validation, y_validation), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv",
        header=False,
        index=False,
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def _save_model(base_directory, target_transformer, features_transformer):
    """Save the Scikit-Learn transformation pipelines.

    This function creates a model.tar.gz file that
    contains the two transformation pipelines we built
    to transform the data.
    """
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(target_transformer, Path(directory) / "target.joblib")
        joblib.dump(features_transformer, Path(directory) / "features.joblib")

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(f"{(model_path / 'model.tar.gz').as_posix()}", "w:gz") as tar:
            tar.add(Path(directory) / "target.joblib", arcname="target.joblib")
            tar.add(
                Path(directory) / "features.joblib", arcname="features.joblib",
            )


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")
