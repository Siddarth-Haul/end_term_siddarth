import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\ASUS\PycharmProjects\pythonProject1\sample_customer_data_for_exam.xlsx'
df = pd.read_excel(file_path)

# Step 1: Scatter plot of purchase frequency vs purchase amount, color-coded by loyalty status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='purchase_frequency', y='purchase_amount', hue='loyalty_status', palette='viridis', alpha=0.7)
plt.title('Purchase Frequency vs Purchase Amount Color-Coded by Loyalty Status')
plt.xlabel('Purchase Frequency')
plt.ylabel('Purchase Amount')
plt.legend(title='Loyalty Status')
plt.show()

# Step 2: Calculate average purchase amount for promotion users vs non-users
average_purchase_by_promo = df.groupby('promotion_usage')['purchase_amount'].mean()
print("Average Purchase Amount for Promotion Users vs Non-Users:")
print(average_purchase_by_promo)

# Step 3: Create a violin plot for satisfaction scores by loyalty status
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='loyalty_status', y='satisfaction_score', palette='pastel')
plt.title('Distribution of Satisfaction Score by Loyalty Status')
plt.xlabel('Loyalty Status')
plt.ylabel('Satisfaction Score')
plt.show()

# Step 4: Create a stacked bar chart for promotion usage across product categories
promo_counts = df.groupby(['product_category', 'promotion_usage']).size().unstack()
promo_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightcoral', 'lightgreen'])
plt.title('Proportion of Promotion Usage Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Count of Customers')
plt.xticks(rotation=45)
plt.legend(title='Promotion Usage', labels=['Not Used', 'Used'])
plt.show()

# Step 5: Calculate correlation between satisfaction score and purchase frequency
correlation_satisfaction_frequency = df['satisfaction_score'].corr(df['purchase_frequency'])
print(f"Correlation between Satisfaction Score and Purchase Frequency: {correlation_satisfaction_frequency:.2f}")
