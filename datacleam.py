import pandas as pd

def missing_value(df):
    col_na = list(df.columns[df.isna().any()])
    missing_numbers = list(df[col_na].isna().sum())
    ratio_na = list(map(lambda num: (num / len(df)) * 100, missing_numbers))
    final_na = pd.DataFrame({'Column_name': col_na, 'Missing_num': missing_numbers, 'Ratio': ratio_na})
    print(final_na)

# Import the CSV file with the first row as column names
hotel_raw = pd.read_csv('hotel_bookings.csv', header=0)

# Display the first few rows of the DataFrame
# print(hotel_raw.head())

# Call the missing_value function to analyze missing values
missing_value(hotel_raw)
