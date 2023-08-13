import pandas as pd
import numpy as np

def missing_value(df):
    col_na = list(df.columns[df.isna().any()])
    missing_numbers = list(df[col_na].isna().sum())
    ratio_na = list(map(lambda num: (num / len(df)) * 100, missing_numbers))
    final_na = pd.DataFrame({'Column_name': col_na, 'Missing_num': missing_numbers, 'Ratio': ratio_na})
    print(final_na)

# Import the CSV file with the first row as column names
hotel_raw = pd.read_csv('hotel_bookings.csv', header=0)

# Display the first few rows of the DataFrame
print(hotel_raw.head())

# Call the missing_value function to analyze missing values
missing_value(hotel_raw)

# Replace missing values in 'children' column with 0
hotel_raw.loc[hotel_raw['children'].isna(), 'children'] = 0

# Replace missing values in 'company' column with "Not_using_company"
hotel_raw['company'].fillna("Not_using_company", inplace=True)

# Replace missing values in 'agent' column with "Not_using_agent"
hotel_raw['agent'].fillna("Not_using_agent", inplace=True)

# Analyze 'country' column and drop rows with missing values
print("Before dropping rows:", len(hotel_raw))
hotel_raw = hotel_raw.drop(hotel_raw[hotel_raw['country'].isna()].index)
print("After dropping rows:", len(hotel_raw))

# Display the updated DataFrame
print(hotel_raw.head())

# Call the missing_value function to analyze missing values
missing_value(hotel_raw)
