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

## number of total retail locations
def add_total_number_of_unique_zip_codes_column():
    processed_df["total_number_of_transactions"] = raw_df.groupby('cc_num')['zip'].transform('nunique')

    print(f"added total unique zip codes column")

## number of retail locations per day in a time frame TODO
#def add_number_of_unique_zip_codes_in_timeframe_column(column_name, timeframe):
    # couldn't do

## minimum number of minutes between transactions of two retail locations in a time frame
#def minimum_minutes_(column_name, timeframe):
    # couldn't do

def calculate_speed():
    from geopy.distance import geodesic

    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return geodesic(coords_1, coords_2).meters

    # Sort the DataFrame by 'cc_num' and 'trans_date_trans_time'
    raw_df.sort_values(by=['cc_num', 'trans_date_trans_time'], inplace=True)

    raw_df['time_diff'] = raw_df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds()

    raw_df['lat_diff'] = raw_df.groupby('cc_num')['merch_lat'].diff().fillna(0)
    raw_df['lon_diff'] = raw_df.groupby('cc_num')['merch_long'].diff().fillna(0)

    raw_df['distance'] = raw_df.apply(lambda row: calculate_haversine_distance(row['merch_lat'], row['merch_long'], row['merch_lat'] - row['lat_diff'], row['merch_long'] - row['lon_diff']), axis=1)

    # Calculate the speed (distance divided by time)
    processed_df['speed'] = raw_df['distance'] / raw_df['time_diff']

    print("calculated speed")

add_original_columns()
add_category_column()
add_time_columns()
add_total_number_of_transactions_column()

add_maximum_amount_per_transaction_over_timeframe_column("maximum amount per transaction over 30 days", '30D')
add_average_amount_per_transaction_over_timeframe_column("average amount per transaction over 30 days", '30D')

add_total_number_of_unique_zip_codes_column()

calculate_speed()

## SAVE TO CSV FILE ##
processed_df.to_csv(output_filepath, index=False)
