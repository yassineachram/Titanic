from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, C=0.0005, random_state=0):
    """Train a logistic regression model on the given training data.

    This function initializes a Logistic Regression model with the specified 
    regularization strength (`C`) and random seed for reproducibility. The model 
    is then trained using the provided training data.

    Args:
        X_train (pd.DataFrame): The feature matrix for the training data.
        y_train (pd.Series): The target labels for the training data.
        C (float, optional): The regularization strength for the logistic regression model.
                              Smaller values specify stronger regularization. Default is 0.0005.
        random_state (int, optional): The random seed for reproducibility. Default is 0.

    Returns:
        model: The trained Logistic Regression model.
    """
    model = LogisticRegression(C=C, random_state=random_state)
    model.fit(X_train, y_train)
    return model