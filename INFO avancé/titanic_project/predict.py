from memory_profiler import profile
import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import feature_engineering

@profile  
def main():
    """Main function to load a pre-trained model and scaler, preprocess new data, and make predictions.

    This function performs the following tasks:
    1. Loads the pre-trained model and scaler from disk.
    2. Loads the new test data from a CSV file.
    3. Preprocesses the new data (handles missing values, feature extraction, etc.).
    4. Applies feature engineering to ensure the test data has the same structure as the training data.
    5. Reindexes the data to match the feature columns used in the model training (with missing columns filled with 0).
    6. Scales the new data using the previously saved scaler.
    7. Makes predictions using the pre-trained model.
    8. Prints the predictions for the new data.

    The process ensures that the test data is transformed to match the format expected by the model, including feature scaling and feature engineering.

    Args:
        None

    Returns:
        None
    """
    # Load the model and scaler
    model = joblib.load(r'C:\Users\HP\Documents\INFO avancé\titanic_project\models\model.pkl')
    scaler = joblib.load(r'C:\Users\HP\Documents\INFO avancé\titanic_project\models\scaler.pkl')
    
    # Load and preprocess new data
    new_data = load_data(r'C:\Users\HP\Documents\INFO avancé\titanic_project\data\test.csv')
    new_data = preprocess_data(new_data)
    new_data, _ = feature_engineering(new_data, new_data)

    feature_columns = ['passengerid', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'age_NA','fare_NA', 'sex_male', 'cabin_C', 'cabin_Missing', 'cabin_Rare','embarked_Q', 'embarked_Rare', 'embarked_S', 'title_Mr', 'title_Mrs','title_Rare']
    
    new_data = new_data.reindex(columns=feature_columns, fill_value=0)
    
    # Scale the new data
    new_data = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data)
    print(predictions)

if __name__ == "__main__":
    main()