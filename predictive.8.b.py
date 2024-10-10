import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject1\sample_customer_data_for_exam.xlsx'
df = pd.read_excel(file_path)

# Filter relevant columns (ensure 'promotion_usage' is included)
df = df[['age', 'income', 'product_category', 'gender', 'promotion_usage']]

# Convert 'gender' to numeric: 0 for Male, 1 for Female
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Convert 'promotion_usage' to numeric: 0 for No, 1 for Yes
df['promotion_usage'] = df['promotion_usage'].map({'No': 0, 'Yes': 1})

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values if any
df.fillna(df.mean(), inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['product_category'], drop_first=True)

# Define features (X) and target (y)
X = df.drop('promotion_usage', axis=1)
y = df['promotion_usage']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance
top_features = feature_importances.sort_values(by='Importance', ascending=False).head(3)

print("Top 3 Features contributing to predicting 'promotion_usage':")
print(top_features)
