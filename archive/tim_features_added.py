## NECESSARY LIBRARIES AND CONSTANTS ##
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

input_filepath='./assets/fraudTrain.csv'
output_filepath='./assets/processedFraudTrain.csv'

## READ CSV FILE INTO DATAFRAME ##
raw_df = pd.read_csv(input_filepath)
# raw_df = raw_df.sample(frac=0.01)  # TEMP
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
    processed_df['is_fraud'] = raw_df['is_fraud']

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
## trans_date_trans_time - Time of transaction already in datetime format

## unix_time - UNIX Time of transaction
## trans_num - Transaction Number
## cc_num - Credit Card Number
## street - Street Address of Credit Card Holder
## city - City of Credit Card Holder
## state - State of Credit Card Holder


## average amount per day for buyer over timeframe
def add_average_amount_per_day_over_timeframe_column(column_name, timeframe):
    ## average amount spent per transaction over a month on all transactions ##     

    processed_df[column_name] = raw_df.groupby(['first','last'])['amt'].sum().div(timeframe)
    print(f"added {column_name} column")

## maximum amount per day for buyer over timefrime
def add_maximum_amount_per_day_over_timeframe_column(column_name, timeframe):
    ## maximum amount spent per transaction over a month on all transactions ##
    processed_df[column_name] = raw_df.groupby(['first','last'])['amt'].max()
    print(f"added {column_name} column")
    print(f"added {column_name} column")

## average amount per day for buyer over timefrime
def add_average_number_of_transactions__by_merchant_column(column_name, timeframe):
    processed_df[column_name] = raw_df.groupby(raw_df['merchant']).transform(len).div(timeframe)
    print(f"added {column_name} column")
    print(f"added {column_name} column")

## maximum number of transactions over timefrime
def add_max_amount_of_transactions_by_merchant_column(column_name,timeframe):
    processed_df[column_name] = raw_df.groupby(raw_df['merchant']).transform(len).max()
    print(f"added {column_name} column")
    
## number of transactions by a merchant in the last 30 days
def find_num_of_transactions_by_merchant_30_days(column_name):
    start_date = pd.to_datetime(raw_df['trans_date_trans_time'])
    end_date = start_date - pd.to_timedelta(30)
    processed_df[column_name] = raw_df.groupby('merchant')[raw_df['trans_date_trans_time'].isin(pd.date_range(start_date, end_date, freq='D'))].transform(len)

## average_amount spent over one week during the past three months on the same merchant type
def average_amount_spent_over_one_week_during_past_three_months_on_merchant_type(column_name):
    start_date = pd.to_datetime(raw_df['trans_date_trans_time'])
    end_date = start_date - pd.to_timedelta(90)
    processed_df[column_name] = raw_df.groupby('merchant',raw_df['trans_date_trans_time'].isin(pd.date_range(start_date, end_date, freq='D')))['amt'].div(12)
    ## assuming 4 weeks in a month
    ## couldn't figure out the exact syntax
