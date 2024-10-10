import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject1\sample_customer_data_for_exam.xlsx'
df = pd.read_excel(file_path)

# Filter the DataFrame to keep only the relevant columns
df = df[['age', 'income', 'product_category', 'gender', 'purchase_amount']]

# Convert 'gender' to numeric: 0 for Male, 1 for Female
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values (if any)
df['age'].fillna(df['age'].mean(), inplace=True)
df['income'].fillna(df['income'].mean(), inplace=True)
df['purchase_amount'].fillna(df['purchase_amount'].mean(), inplace=True)

# If product_category is categorical, use OneHotEncoder
df = pd.get_dummies(df, columns=['product_category'], drop_first=True)

# Define features (X) and target (y)
X = df.drop('purchase_amount', axis=1)
y = df['purchase_amount']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Get feature names and coefficients
feature_names = X.columns
coefficients = model.coef_

# Create a DataFrame for better visualization
coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort the DataFrame by absolute value of coefficients
coeff_df['Absolute Coefficient'] = coeff_df['Coefficient'].abs()
top_features = coeff_df.sort_values(by='Absolute Coefficient', ascending=False).head(3)

print("Top 3 Features contributing to predicting 'purchase_amount':")
print(top_features[['Feature', 'Coefficient']])
