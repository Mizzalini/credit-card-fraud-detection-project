import os
import feature_gen
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Constants
MAX_DEPTH = 10

def generate_new_features() -> str:
    """
    If the training data has not been feature engineered, new features are generated.
    
    Args:
        None
        
    Returns:
        None
    """
    if not os.path.exists(feature_gen.OUTPUT_FILEPATH):
        feature_gen.main()
    return feature_gen.OUTPUT_FILEPATH

def read_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Read the dataset from the given file path and perform necessary cleaning.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Label encoding for binary categories
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])

    return data

def train_dtc(X: pd.DataFrame, y: pd.Series, max_depth: int) -> DecisionTreeClassifier:
    """
    Train a decision tree classifier.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable.
        max_depth (int): Maximum depth of the decision tree.

    Returns:
        DecisionTreeClassifier: Trained classifier.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf

def eval_classifier(clf: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the classifier on the test set and print accuracy and classification report.

    Args:
        clf (DecisionTreeClassifier): Trained classifier.
        X_test (pd.DataFrame): Test set features.
        y_test (pd.Series): Test set target variable.

    Returns:
        None
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(report)
    
def main() -> None:
    """
    Main function for training & testing dataset.

    Reads in a processed dataset and cleans it. Selects the features for training.
    Trains the Decision Tree Classifier and evaluates it on the test set.
    
    Args:
        None
        
    Returns:
        None
    """
    training_data = generate_new_features()
    # Process training data
    processed_data_train = read_and_clean_data(training_data)

    # Select features and target variable for training
    features_train = ['category', 'amt', 'city_pop', 'total_transactions', 'average amount over 30 days', 'maximum amount over 30 days']
    X_train = processed_data_train[features_train]
    y_train = processed_data_train['is_fraud']

    # Train the decision tree classifier
    clf = train_dtc(X_train, y_train, MAX_DEPTH)
    
    # Printing the importance of each feature
    print(clf.feature_importances_)

    # Evaluate the classifier on the test set
    processed_data_test = read_and_clean_data('assets/fraudTest.csv')
    
    # Check if features are present in the test data
    features_test = features_train
    missing_features = set(features_test) - set(processed_data_test.columns)
    
    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")
    
    features_test = features_train  # Assuming the same features are used for testing
    X_test = processed_data_test[features_test]
    y_test = processed_data_test['is_fraud']
    
    eval_classifier(clf, X_test, y_test)

if __name__ == '__main__':
    main()
