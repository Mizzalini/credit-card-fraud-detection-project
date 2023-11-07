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

    # Label encoding for binary categories
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])

    # Frequency encoding for high-cardinality categories
    frequency_map = data['merchant'].value_counts().to_dict()
    data['merchant'] = data['merchant'].map(frequency_map)

    # Extract date and time components
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['trans_year'] = data['trans_date_trans_time'].dt.year
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_day'] = data['trans_date_trans_time'].dt.day
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour
    data['trans_minute'] = data['trans_date_trans_time'].dt.minute
    data['trans_second'] = data['trans_date_trans_time'].dt.second

    return data

processedData = processDB('assets/fraudTrain.csv')

# Select features and target variable
# features = ['merchant', 'category', 'amt', 'unix_time', 'merch_lat', 'merch_long', 'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'trans_minute', 'trans_second']
features = ['category', 'amt', 'city_pop']
X_train = processedData[features]
y_train = processedData['is_fraud']

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=10)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model (optional)
# If you want to evaluate the model's performance on the training data, you can use the same code as before, with X and y being the full dataset

testData = processDB('assets/fraudTest.csv')

X_test = testData[features]
y_test = testData['is_fraud']

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)
