import pandas as pd
import numpy as np
import re

def load_data(filepath):
    """Function to load the dataset and clean missing values

    Args:
        filepath (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The cleaned dataset with 'nan' values replaced and column names in lowercase.
    """
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)  # Replace '?' with NaN for missing values
    data.columns = data.columns.str.lower()  # Standardize column names to lowercase
    return data

def preprocess_data(data):
    """Preprocess the dataset by extracting relevant features and converting data types.

    Args:
        data (pd.DataFrame): The dataset to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed dataset with extracted features like 'cabin' and 'title', 
        and with 'fare' and 'age' columns converted to float.
    """
    data['cabin'] = data['cabin'].apply(get_first_cabin)  # Extract first cabin letter
    data['title'] = data['name'].apply(get_title)  # Extract title from name
    data['fare'] = data['fare'].astype('float')  # Convert fare to float type
    data['age'] = data['age'].astype('float')  # Convert age to float type
    #data.drop(labels=['name', 'ticket', 'boat', 'body', 'home.dest'], axis=1, inplace=True)  # Optionally drop irrelevant columns
    return data

def get_first_cabin(row):
    """Extract the first cabin letter from the 'cabin' column.

    Args:
        row (str): The cabin data in string format (e.g., 'C123').

    Returns:
        str or np.nan: The first letter of the cabin (e.g., 'C') or NaN if no cabin is present.
    """
    try:
        return row.split()[0]  # Split and return the first part of the cabin string
    except:  # noqa: E722
        return np.nan  # Return NaN if no valid cabin data is found

def get_title(passenger):
    """Extract the title (Mr, Mrs, Miss, etc.) from the passenger's name.

    Args:
        passenger (str): The passenger's name in string format.

    Returns:
        str: The extracted title (e.g., 'Mr', 'Mrs', 'Miss', 'Master', or 'Other' if not found).
    """
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'  # Return 'Other' if no known title is found