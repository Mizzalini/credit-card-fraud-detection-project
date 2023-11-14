import pandas as pd
from datetime import datetime, timedelta

input_filepath='./assets/fraudTrain.csv'
output_filepath='./assets/processedFraudTrain.csv'

# Read CSV file into DataFrame
raw_df = pd.read_csv(input_filepath)
raw_df['trans_date_trans_time'] = pd.to_datetime(raw_df['trans_date_trans_time'])
raw_df = raw_df.sort_values(by=['cc_num', 'trans_date_trans_time'])

processed_df = pd.DataFrame()

# Define function to add columns to the DataFrame

# Columns directly from original
def add_original_columns():
    processed_df['cc_num'] = raw_df['cc_num']
    processed_df['amt'] = raw_df['amt']
    processed_df['is_fraud'] = raw_df['is_fraud']
    processed_df['trans_date_trans_time'] = raw_df['trans_date_trans_time']

    print("Added original columns")

# Time passed since last purchase
def add_time_since_last_purchase():
    processed_df['time_since_last_purchase'] = raw_df.groupby('cc_num')['trans_date_trans_time'].diff()
    # Note that if there's no previous transaction for a buyer, the column will be ''
    
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
    
def create_df():
    add_original_columns()
    add_time_since_last_purchase()
    add_total_transactions()
    add_avg_transaction_over_timeframe("average amount over 30 days", '30D')
    add_max_transaction_over_timeframe("maximum amount over 30 days", '30D')
    
    processed_df.to_csv(output_filepath, index=False)
    
create_df()