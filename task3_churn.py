# Task 3 - Customer Churn Prediction
# Analyzing which customers might stop using a service

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("Starting Task 3 - Customer Churn Analysis")

# For this example, we'll create sample data to show the approach
print("In a real project, we would use actual customer data")
print("This example shows the complete process we would follow")

# Create sample data to demonstrate
sample_data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 30, 55, 28],
    'monthly_charges': [50, 80, 60, 90, 45, 75, 100, 55, 85, 52],
    'customer_service_calls': [1, 3, 0, 5, 2, 1, 4, 1, 6, 0],
    'churned': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(sample_data)

print("Sample customer data created:")
print(df.head())

print("\nSteps we follow in churn analysis:")
print("1. Look at customer data and clean it")
print("2. Find patterns in why customers leave")
print("3. Build a model to predict at-risk customers")
print("4. Help the business reduce customer turnover")

# Create a visualization showing common churn factors
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Show age distribution of customers who churned vs stayed
sns.boxplot(data=df, x='churned', y='age')
plt.title('Age vs Customer Churn')
plt.xlabel('Churned (0=No, 1=Yes)')
plt.ylabel('Age')

plt.subplot(1, 2, 2)
# Show relationship between service calls and churn
sns.scatterplot(data=df, x='customer_service_calls', y='monthly_charges', hue='churned', s=100)
plt.title('Service Calls & Monthly Charges vs Churn')
plt.xlabel('Customer Service Calls')
plt.ylabel('Monthly Charges ($)')

plt.tight_layout()
plt.show()

print("\nTypical insights from churn analysis:")
print("- Customers with more service calls are more likely to leave")
print("- Higher monthly charges can increase churn risk")
print("- Certain age groups may be more likely to switch services")

print("Task 3 approach demonstrated successfully!")
