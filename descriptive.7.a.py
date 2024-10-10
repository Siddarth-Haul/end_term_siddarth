import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Correct file path
file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject1\sample_customer_data_for_exam.xlsx'

# Load the dataset
df = pd.read_excel(file_path)

# Display the first few rows
print(df.head())

# Filter the DataFrame to keep only the relevant columns
df = df[['age', 'income', 'product_category', 'gender', 'purchase_amount']]

# Check the data types to identify non-numeric columns
print(df.dtypes)

# Convert 'gender' to numeric: 0 for Male, 1 for Female
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Summary statistics for numerical columns
print(df.describe())

# Calculate correlation matrix
correlation_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation Between Numerical Variables')
plt.show()

# Histograms for 'age' and 'income'
plt.figure(figsize=(14, 5))

# Age histogram
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Income histogram
plt.subplot(1, 2, 2)
plt.hist(df['income'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Box plot of 'purchase_amount' across 'product_category'
plt.figure(figsize=(10, 6))
sns.boxplot(x='product_category', y='purchase_amount', data=df)
plt.title('Box Plot of Purchase Amount by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Purchase Amount')
plt.show()

# Pie chart for 'gender'
gender_counts = df['gender'].value_counts()
labels = ['Male', 'Female']
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightpink'], startangle=90, explode=(0.1, 0))
plt.title('Proportion of Customers by Gender')
plt.show()

