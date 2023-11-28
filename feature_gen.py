import pandas as pd

# Constants
# INPUT_FILEPATH = './assets/fraudTrain.csv'
OUTPUT_FILEPATH = 'assets/processedFraudTrain.csv'
TIME_FRAME = '30D'

# Define functions to add columns to the DataFrame

def add_original_columns(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """
    Add original columns directly from the original DataFrame.

    Args:
        processed_df (pd.DataFrame): The DataFrame to which columns will be added.
        raw_df (pd.DataFrame): The original DataFrame containing the source columns.

    Returns:
        None
    """
    columns_to_copy = ['cc_num', 'amt', 'is_fraud', 'trans_date_trans_time', 'category', 'city_pop']
    processed_df[columns_to_copy] = raw_df[columns_to_copy]
    print("Added Original columns")

def add_time_columns(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """
    Add columns related to transaction time.

    Args:
        processed_df (pd.DataFrame): The DataFrame to which columns will be added.
        raw_df (pd.DataFrame): The original DataFrame containing the source columns.

    Returns:
        None
    """
    processed_df['trans_year'] = raw_df['trans_date_trans_time'].dt.year
    processed_df['trans_month'] = raw_df['trans_date_trans_time'].dt.month
    processed_df['trans_day'] = raw_df['trans_date_trans_time'].dt.day
    processed_df['trans_hour'] = raw_df['trans_date_trans_time'].dt.hour
    print("Added Time columns")

def add_time_since_last_purchase(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """
    Add a column representing time since the last purchase for each buyer.

    Args:
        processed_df (pd.DataFrame): The DataFrame to which the column will be added.
        raw_df (pd.DataFrame): The original DataFrame containing the source columns.

    Returns:
        None
    """
    processed_df['time_since_last_purchase'] = raw_df.groupby('cc_num')['trans_date_trans_time'].diff()
    print("Added Time Since Last Purchase column")

def add_total_transactions(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """
    Add a column representing the total number of transactions for each buyer.

    Args:
        processed_df (pd.DataFrame): The DataFrame to which the column will be added.
        raw_df (pd.DataFrame): The original DataFrame containing the source columns.

    Returns:
        None
    """
    processed_df['total_transactions'] = raw_df.groupby('cc_num')['cc_num'].transform('count')
    print("Added Total Transactions column")

def add_avg_transaction_over_timeframe(processed_df: pd.DataFrame, raw_df: pd.DataFrame, column_name: str, time_frame: str) -> None:
    """
    Add a column representing the average transaction amount over a specified time frame.

    Args:
        processed_df (pd.DataFrame): The DataFrame to which the column will be added.
        raw_df (pd.DataFrame): The original DataFrame containing the source columns.
        column_name (str): The name of the new column.
        time_frame (str): The time frame for rolling window calculations.

    Returns:
        None
    """
    def calculate_average(group: pd.DataFrame) -> pd.Series:
        # Calculate the average transaction amount over a specified time frame for a group of transactions.
        return group.rolling(window=time_frame, on='trans_date_trans_time')['amt'].mean()

    processed_df[column_name] = raw_df.groupby('cc_num', group_keys=False, as_index=False).apply(calculate_average)
    print(f"Added Average Transaction Over {time_frame} column")

def add_max_transaction_over_timeframe(processed_df: pd.DataFrame, raw_df: pd.DataFrame, column_name: str, time_frame: str) -> None:
    """
    Add a column representing the maximum transaction amount over a specified time frame.

    Args:
        processed_df (pd.DataFrame): The DataFrame to which the column will be added.
        raw_df (pd.DataFrame): The original DataFrame containing the source columns.
        column_name (str): The name of the new column.
        time_frame (str): The time frame for rolling window calculations.

    Returns:
        None
    """
    def calculate_max(group: pd.DataFrame) -> pd.Series:
        # Calculate the maximum transaction amount over a specified time frame for a group of transactions.
        return group.rolling(window=time_frame, on='trans_date_trans_time')['amt'].max()

    processed_df[column_name] = raw_df.groupby('cc_num', group_keys=False, as_index=False).apply(calculate_max)
    print(f"Added Maximum Transaction Over {time_frame} column")

def create_df(processed_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    """
    Create the final processed DataFrame and save it to a CSV file.

    Args:
        processed_df (pd.DataFrame): The DataFrame containing processed data.
        raw_df (pd.DataFrame): The original DataFrame containing source data.

    Returns:
        None
    """
    add_original_columns(processed_df, raw_df)
    add_time_columns(processed_df, raw_df)
    add_time_since_last_purchase(processed_df, raw_df)
    add_total_transactions(processed_df, raw_df)
    add_avg_transaction_over_timeframe(processed_df, raw_df, "average amount over 30 days", TIME_FRAME)
    add_max_transaction_over_timeframe(processed_df, raw_df, "maximum amount over 30 days", TIME_FRAME)

    processed_df.to_csv(OUTPUT_FILEPATH, index=False)
    print(f"Processed DataFrame saved to {OUTPUT_FILEPATH}")
    
def main() -> str:
    """
    Main function for processing fraud data.

    Reads a CSV file containing fraud data, performs various data processing tasks,
    and saves the processed data to a new CSV file.
    
    Args:
        None
        
    Returns:
        str: The filepath of the processed dataframe.
    """
    # Read CSV file into DataFrame
    try:
        input_filepath = input("Input file: ")
        raw_df = pd.read_csv(input_filepath)
    except FileNotFoundError as e:
        print(f"Error: File not found at {input_filepath}")
        print(e)
        return
    except pd.errors.EmptyDataError as e:
        print(f"Error: Empty data or invalid file format in {input_filepath}")
        print(e)
        return
    
    raw_df['trans_date_trans_time'] = pd.to_datetime(raw_df['trans_date_trans_time'])
    raw_df = raw_df.sort_values(by=['cc_num', 'trans_date_trans_time'])
    
    # Initialize processed DataFrame
    processed_df = pd.DataFrame()
    create_df(processed_df, raw_df)

# Call the main function to create and save the processed DataFrame
if __name__ == '__main__':
    main()  
