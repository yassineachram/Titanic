import pandas as pd
import numpy as np

def feature_engineering(X_train, X_test):
    """Perform feature engineering on the training and testing datasets.
    
    This function creates new features, handles missing values, encodes categorical variables,
    and ensures that both the training and testing sets have the same columns.

    Args:
        X_train (pd.DataFrame): The training dataset containing the features.
        X_test (pd.DataFrame): The testing dataset containing the features.

    Returns:
        pd.DataFrame, pd.DataFrame: The processed training and testing datasets with the new features.
    """
    # Extract the first character of the cabin column for both train and test
    X_train['cabin'] = X_train['cabin'].str[0]
    X_test['cabin'] = X_test['cabin'].str[0]
    
    # Handle missing values for 'age' and 'fare' columns
    for var in ['age', 'fare']:
        X_train[var+'_NA'] = np.where(X_train[var].isnull(), 1, 0)
        X_test[var+'_NA'] = np.where(X_test[var].isnull(), 1, 0)
        median_val = X_train[var].median()
        X_train[var] = X_train[var].fillna(median_val)
        X_test[var] = X_test[var].fillna(median_val)
    
    # Fill missing values in categorical columns
    vars_cat = [c for c in X_train.columns if X_train[c].dtypes == 'O']
    X_train[vars_cat] = X_train[vars_cat].fillna('Missing')
    X_test[vars_cat] = X_test[vars_cat].fillna('Missing')
    
    # Replace infrequent categories with 'Rare' in categorical variables
    for var in vars_cat:
        frequent_ls = find_frequent_labels(X_train, var, 0.05)
        X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
        X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')
    
    # One-hot encode categorical variables and drop the original columns
    for var in vars_cat:
        X_train = pd.concat([X_train, pd.get_dummies(X_train[var], prefix=var, drop_first=True)], axis=1)
        X_test = pd.concat([X_test, pd.get_dummies(X_test[var], prefix=var, drop_first=True)], axis=1)
    
    X_train.drop(labels=vars_cat, axis=1, inplace=True)
    X_test.drop(labels=vars_cat, axis=1, inplace=True)

    # Ensure both datasets have the same columns by adding missing columns to X_test
    expected_features = X_train.columns
    for col in expected_features:
       if col not in X_test.columns:
          X_test[col] = 0

    # Align the column order between X_train and X_test
    X_test = X_test[expected_features]
    
    return X_train, X_test

def find_frequent_labels(df, var, rare_perc):
    """Find the frequent labels in a categorical variable.

    This function calculates the frequency of each category in a column, and returns the categories
    that exceed the specified percentage of the total dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        var (str): The column name (variable) to check for frequent labels.
        rare_perc (float): The percentage threshold for considering a label as frequent.

    Returns:
        pd.Index: The index of categories that are frequent in the specified column.
    """
    tmp = df.groupby(var)[var].count() / len(df)  # Calculate frequency of each label
    return tmp[tmp > rare_perc].index  # Return labels that are more frequent than the threshold