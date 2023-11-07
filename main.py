## Import necessary libraries 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report

## Read training dataset
trainingData = pd.read_csv('assets/fraudTrain.csv')

## Clean dataset
trainingData.dropna(inplace=True)

# Label encoding for binary categories
label_encoder = LabelEncoder()
trainingData['category'] = label_encoder.fit_transform(trainingData['category'])

# Frequency encoding for high-cardinality categories
frequency_map = trainingData['merchant'].value_counts().to_dict()
trainingData['merchant'] = trainingData['merchant'].map(frequency_map)

# Extract date and time components
trainingData['trans_date_trans_time'] = pd.to_datetime(trainingData['trans_date_trans_time'])
trainingData['trans_year'] = trainingData['trans_date_trans_time'].dt.year
trainingData['trans_month'] = trainingData['trans_date_trans_time'].dt.month
trainingData['trans_day'] = trainingData['trans_date_trans_time'].dt.day
trainingData['trans_hour'] = trainingData['trans_date_trans_time'].dt.hour
trainingData['trans_minute'] = trainingData['trans_date_trans_time'].dt.minute
trainingData['trans_second'] = trainingData['trans_date_trans_time'].dt.second

trainingData.to_csv('assets/preprocessed_fraudTrain.csv', index=False)

tempData = pd.read_csv('assets/preprocessed_fraudTrain.csv')

# Select features and target variable
features = ['merchant', 'category', 'amt', 'unix_time', 'merch_lat', 'merch_long', 'trans_year', 'trans_month', 'trans_day', 'trans_hour', 'trans_minute', 'trans_second']
X = tempData[features]
y = tempData['is_fraud']

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=10)

# Train the model
clf.fit(X, y)

# Evaluate the model (optional)
# If you want to evaluate the model's performance on the training trainingData, you can use the same code as before, with X and y being the full dataset.
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)