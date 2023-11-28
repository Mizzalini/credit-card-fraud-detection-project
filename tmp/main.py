## Import necessary libraries 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report


def processDB(file):
    ## Read training dataset
    data = pd.read_csv(file)

    ## Clean dataset
    data.dropna(inplace=True)

    Label encoding for binary categories
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])

    return data

processedData = processDB('assets/processedFraudTrain.csv')

# Select features and target variable
#features = ['amt', 'category', 'trans_hour', 'age', 'time_since_last_purchase', 'total_transactions', 'average amount over 30 days', 'maximum amount over 30 days']

#features = ['amt', 'category', 'city_pop', 'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'age', 'time_since_last_purchase', 'total_transactions', 'average amount over 30 days', 'maximum amount over 30 days', 'speed']

#features = ['amt', 'category', 'city_pop', 'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'age']

#features = ['amt', 'category', 'trans_hour', 'age']

#features = ['category']

features = ['amt', 'category', 'trans_hour']

X_train = processedData[features]
y_train = processedData['is_fraud']

#from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=0)
#X_train, y_train = smote.fit_resample(X_train, y_train)

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=10)

# Train the model
clf.fit(X_train, y_train)

# Test the model
testData = processDB('assets/processedFraudTest.csv')

X_test = testData[features]
y_test = testData['is_fraud']

# Evaluate the tested model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)

print(features)

# Print feature importances with three significant figures
print("Feature Importances:")
for feature, importance in zip(features, clf.feature_importances_):
    print(f"{feature}: {importance:.3f}")
