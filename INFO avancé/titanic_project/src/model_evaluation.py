from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate the performance of the trained model on both training and test datasets.
    
    This function computes the ROC-AUC and Accuracy for both the training and test datasets,
    providing an indication of the model's performance on classification tasks.

    Args:
        model: The trained machine learning model (e.g., a classifier from scikit-learn).
        X_train (pd.DataFrame): The feature matrix for the training set.
        y_train (pd.Series): The target labels for the training set.
        X_test (pd.DataFrame): The feature matrix for the test set.
        y_test (pd.Series): The target labels for the test set.
    
    Prints:
        - Train ROC-AUC score
        - Train Accuracy score
        - Test ROC-AUC score
        - Test Accuracy score
    """
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    
    print('Train ROC-AUC: {}'.format(roc_auc_score(y_train, train_proba)))
    print('Train Accuracy: {}'.format(accuracy_score(y_train, train_pred)))
    print('Test ROC-AUC: {}'.format(roc_auc_score(y_test, test_proba)))
    print('Test Accuracy: {}'.format(accuracy_score(y_test, test_pred)))