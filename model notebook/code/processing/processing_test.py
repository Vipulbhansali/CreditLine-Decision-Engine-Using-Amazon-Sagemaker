
%%script false
import pandas as pd
from processing.script import convert_to_lowercase, handle_missing_values, extract_and_transform_features, adjust_numeric_features, add_new_feature, drop_column
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer

# Path to the CSV file
csv_file_path = r'C:\Users\Vipul\CreditLine-Decision-Engine-Using-Amazon-Sagemaker\credit.csv'

# Read the CSV file
raw_data = pd.read_csv(csv_file_path)

# Sample the first 1000 rows
raw_data_sample = raw_data.head(500)

# Perform preprocessing steps
preprocessed_data = convert_to_lowercase(raw_data_sample)
preprocessed_data = handle_missing_values(preprocessed_data)
preprocessed_data = extract_and_transform_features(preprocessed_data)
preprocessed_data = adjust_numeric_features(preprocessed_data)
preprocessed_data = add_new_feature(preprocessed_data)
preprocessed_data = drop_column(preprocessed_data)

# Count the number of unique values in the 'pre_approved_offer' column
unique_values_count = preprocessed_data['pre_approved_offer'].nunique()
print("Number of unique values in the target column:", unique_values_count)
value_counts = preprocessed_data['pre_approved_offer'].value_counts()
print("Number of unique values in the target column:", value_counts)

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

# Fit the target transformer
target_transformer.fit(preprocessed_data[['pre_approved_offer']])

# Fit the feature transformer
feature_transformer.fit(preprocessed_data)

# Apply transformation
X_transformed = feature_transformer.transform(preprocessed_data)  # Transform features
y_transformed = target_transformer.transform(preprocessed_data[['pre_approved_offer']])  # Transform target

print("\nTransformed target values:")
print(y_transformed)


