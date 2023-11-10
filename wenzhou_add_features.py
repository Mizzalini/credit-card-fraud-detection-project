## NECESSARY LIBRARIES AND CONSTANTS ##
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

input_filepath='./assets/fraudTrain.csv'
output_filepath='./assets/processedFraudTrain.csv'

## READ CSV FILE INTO DATAFRAME ##
raw_df = pd.read_csv(input_filepath)
raw_df = raw_df.sort_values(by=['trans_date_trans_time'])
raw_df['trans_date_trans_time'] = pd.to_datetime(raw_df['trans_date_trans_time'])

## CREATE NEW DATAFRAME TO ADD PROCESSED COLUMNS ##
processed_df = pd.DataFrame()

### DEFINE FUNCTIONS TO ADD COLUMNS TO DATAFRAME ###

## columns directly from original
def add_original_columns():
    processed_df['trans_num'] = raw_df['trans_num']
    processed_df['cc_num'] = raw_df['cc_num']
    processed_df['amt'] = raw_df['amt']
    processed_df['job'] = raw_df['job']

    print("added original columns")

def add_category_column():
    label_encoder = LabelEncoder()
    processed_df['category'] = label_encoder.fit_transform(raw_df['category'])

    print("added category column")

def add_time_columns():
    processed_df['trans_year'] = raw_df['trans_date_trans_time'].dt.year
    processed_df['trans_month'] = raw_df['trans_date_trans_time'].dt.month
    processed_df['trans_day'] = raw_df['trans_date_trans_time'].dt.day
    processed_df['trans_hour'] = raw_df['trans_date_trans_time'].dt.hour

    print("added time columns")

def add_age_column():
    ## age
    raw_df['dob'] = pd.to_datetime(raw_df['dob'])
    processed_df['age'] = (pd.Timestamp('now') - raw_df['dob']).astype('<m8[Y]').astype(int)

    print("added age column")

# location
## zip code is associated to billing address of credit card
## merch_lat - Latitude Location of Merchant
## merch_long - Longitude Location of Merchant
## unix_time - UNIX Time of transaction
## trans_num - Transaction Number
## cc_num - Credit Card Number
## street - Street Address of Credit Card Holder
## city - City of Credit Card Holder
## state - State of Credit Card Holder

## number of retail locations per day in a time frame

## minimum number of minutes between transactions of two retail locations in a time frame


def add_maximum_amount_per_transaction_over_timeframe_column(column_name, timeframe):
    ## maximum amount spent per transaction over a month on all transactions ##
    def calculate_rolling_maximum(group):
        return group.rolling(timeframe, on='trans_date_trans_time')['amt'].max()

    processed_df[column_name] = raw_df.groupby('cc_num', group_keys=False, as_index=False).apply(calculate_rolling_maximum)
    print(f"added {column_name} column")

def add_average_amount_per_transaction_over_timeframe_column(column_name, timeframe):
    ## average amount spent per transaction over a month on all transactions ##
    def calculate_rolling_average(group):
        return group.rolling(timeframe, on='trans_date_trans_time')['amt'].mean()

    processed_df[column_name] = raw_df.groupby('cc_num', group_keys=False, as_index=False).apply(calculate_rolling_average)
    print(f"added {column_name} column")

def add_total_number_of_transactions_column():
    processed_df["total number of transactions"] = raw_df['cc_num'].groupby(raw_df['cc_num']).transform(len)

    print(f"added total number of transactions column")

## number of retail locations per day in a time frame
'''
def add_total_number_of_unique_zip_codes_column():
    def nunique_zip(group):
        return group['zip'].nunique()

    processed_df["total number of transactions"] = raw_df.groupby('cc_num').transform(nunique_zip)

    print(f"added total unique zip codes column")
'''

add_original_columns()
add_category_column()
add_time_columns()
add_total_number_of_transactions_column()

add_maximum_amount_per_transaction_over_timeframe_column("maximum amount per transaction over 30 days", '30D')
add_average_amount_per_transaction_over_timeframe_column("average amount per transaction over 30 days", '30D')

#add_total_number_of_unique_zip_codes_column()

## SAVE TO CSV FILE ##
processed_df.to_csv(output_filepath, index=False)
