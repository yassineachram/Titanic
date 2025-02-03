import joblib
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.model_evaluation import evaluate_model 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    """Main function to train and evaluate the Titanic survival prediction model.

    This function performs the following tasks:
    1. Loads the training data from a CSV file.
    2. Preprocesses the data (handles missing values, feature extraction, etc.).
    3. Splits the data into training and testing sets.
    4. Applies feature engineering (e.g., handling categorical variables).
    5. Standardizes the features using `StandardScaler`.
    6. Trains a logistic regression model on the training data.
    7. Saves the trained model and the scaler for future use.
    8. Evaluates the model on both training and testing datasets using various metrics.

    The model and scaler are saved as `.pkl` files for later use in prediction tasks.

    Args:
        None

    Returns:
        None
    """
    data = load_data(r"C:\Users\HP\Documents\INFO avancé\titanic_project\data\train.csv")
    data = preprocess_data(data)
    
     # Cette ligne va générer une KeyError car 'non_existent_column' n'existe pas dans les données
    print(data['non_existent_column'])  # Provoque une erreur

    X_train, X_test, y_train, y_test = train_test_split(data.drop('survived', axis=1), data['survived'], test_size=0.2, random_state=0)
    X_train, X_test = feature_engineering(X_train, X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, r'C:\Users\HP\Documents\INFO avancé\titanic_project\models\scaler.pkl')
    X_test = scaler.transform(X_test)
    model = train_model(X_train, y_train)
    joblib.dump(model, r'C:\Users\HP\Documents\INFO avancé\titanic_project\models\model.pkl')
    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()