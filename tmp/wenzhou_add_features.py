import pandas as pd
import numpy as np


def process_database(input_filepath, output_filepath):
    # Read CSV file into DataFrame
    raw_df = pd.read_csv(input_filepath, parse_dates=['dob'])
    raw_df['trans_date_trans_time'] = pd.to_datetime(raw_df['trans_date_trans_time'])
    raw_df['dob'] = pd.to_datetime(raw_df['dob'])

    raw_df = raw_df.sort_values(by=['cc_num', 'trans_date_trans_time'])

    processed_df = pd.DataFrame()

    # Define function to add columns to the DataFrame
    # Columns directly from original
    def add_original_columns():
        processed_df['cc_num'] = raw_df['cc_num']
        processed_df['amt'] = raw_df['amt']
        processed_df['category'] = raw_df['category']
        processed_df['city_pop'] = raw_df['city_pop']

        processed_df['is_fraud'] = raw_df['is_fraud']

        print("Added original columns")
        
    def add_time_columns():
        processed_df['trans_year'] = raw_df['trans_date_trans_time'].dt.year
        processed_df['trans_month'] = raw_df['trans_date_trans_time'].dt.month
        processed_df['trans_day'] = raw_df['trans_date_trans_time'].dt.day
        processed_df['trans_hour'] = raw_df['trans_date_trans_time'].dt.hour

        print("added time columns")

    def add_age_column():
        from datetime import datetime

        ## age
        today = datetime.today()
        processed_df['age'] = raw_df['dob'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

        print("added age column")

    # Time passed since last purchase
    def add_time_since_last_purchase():
        processed_df['time_since_last_purchase'] = raw_df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds()
        
        # Drop first row as there is no time since last purchase at start
        processed_df.drop(index=processed_df.index[0], axis=0, inplace=True)

        print(f"Added Time Since Last Purchase column")

    # Total number of transactions for the buyer
    def add_total_transactions():
        processed_df['total_transactions'] = raw_df.groupby('cc_num')['cc_num'].transform('count')
        
        print(f"Added Total Transactions column")

    # Average amount spent per transaction for buyer in a time frame
    def add_avg_transaction_over_timeframe(column_name, timeframe):
        def calculate_average(group):
            return group.rolling(window=timeframe, on='trans_date_trans_time')['amt'].mean()

        processed_df[column_name] = raw_df.groupby('cc_num', group_keys=False, as_index=False).apply(calculate_average)

        
        print(f"Added Average Transaction Over {timeframe} column")

    # Maximum amount spent for buyer in a time frame
    def add_max_transaction_over_timeframe(column_name, timeframe):
        def calculate_max(group):
            return group.rolling(window=timeframe, on='trans_date_trans_time')['amt'].max()

        processed_df[column_name] = raw_df.groupby('cc_num', group_keys=False, as_index=False).apply(calculate_max)
        
        print(f"Added Maximum Transaction Over {timeframe} column")
        
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

        # Drop first row as there is no speed here
        processed_df.drop(index=processed_df.index[0], axis=0, inplace=True)

        processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows containing NaN values
        processed_df.dropna(inplace=True)

        print("calculated speed")
        
    def create_df():
        add_original_columns()
        add_time_columns()
        add_age_column()
        add_time_since_last_purchase()
        add_total_transactions()
        add_avg_transaction_over_timeframe("average amount over 30 days", '30D')
        add_max_transaction_over_timeframe("maximum amount over 30 days", '30D')

        calculate_speed()

        processed_df.to_csv(output_filepath, index=False)
        
    create_df()

process_database('./assets/fraudTrain.csv', './assets/processedFraudTrain.csv')
process_database('./assets/fraudTest.csv', './assets/processedFraudTest.csv')
