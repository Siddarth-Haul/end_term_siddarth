import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject1\sample_customer_data_for_exam.xlsx'
df = pd.read_excel(file_path)

# Step 1: Calculate average purchase amount for each education level
average_purchase_by_education = df.groupby('education')['purchase_amount'].mean().sort_values(ascending=False)
print("Average Purchase Amount by Education Level:")
print(average_purchase_by_education)

# Step 2: Calculate average satisfaction score for each loyalty status
average_satisfaction_by_loyalty = df.groupby('loyalty_status')['satisfaction_score'].mean().sort_values(ascending=False)
print("Average Satisfaction Score by Loyalty Status:")
print(average_satisfaction_by_loyalty)

# Step 3: Create a bar plot comparing purchase frequency across different regions
purchase_frequency_by_region = df['region'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=purchase_frequency_by_region.index, y=purchase_frequency_by_region.values, palette='viridis')
plt.title('Purchase Frequency Across Different Regions')
plt.xlabel('Region')
plt.ylabel('Purchase Frequency')
plt.xticks(rotation=45)
plt.show()

# Step 4: Compute the percentage of customers who used promotional offers
promo_usage_percentage = (df['promotion_usage'].sum() / df.shape[0]) * 100
print(f"Percentage of Customers Who Used Promotional Offers: {promo_usage_percentage:.2f}%")

# Step 5: Investigate the correlation between income and purchase amount
correlation_income_purchase = df['income'].corr(df['purchase_amount'])
print(f"Correlation between Income and Purchase Amount: {correlation_income_purchase:.2f}")
