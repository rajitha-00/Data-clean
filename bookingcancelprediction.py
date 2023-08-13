import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
hotel_raw = pd.read_csv('hotel_bookings.csv', header=0)

# Preprocess data, handle missing values, and convert categorical variables

# Select relevant features
features = ['lead_time', 'arrival_date_year', 'total_of_special_requests']

# One-hot encode categorical variable 'customer_type'
X = pd.get_dummies(hotel_raw[features + ['customer_type']], drop_first=True)
y = hotel_raw['is_canceled']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate total cancellations and average cancellations for each customer type
cancellation_info = hotel_raw.groupby('customer_type')['is_canceled'].agg(['sum', 'mean'])

# Print the cancellation information
print("\nTotal Cancellations and Average Cancellations by Customer Type:")
print(cancellation_info)
