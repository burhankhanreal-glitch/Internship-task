# Task 4 - Predicting Insurance Costs
# Analyzing how different factors affect medical insurance prices

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("Working on Task 4 - Insurance Cost Analysis")

print("Loading real insurance data from online source...")
# Get actual insurance data
data_url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
insurance_df = pd.read_csv(data_url)

print("Real insurance data loaded successfully!")
print(f"I found {len(insurance_df)} customers in the dataset")
print("\nHere's what the data looks like:")
print(insurance_df.head())

print("\nThe data columns I'm working with:")
print("- age: Customer age")
print("- sex: Gender")
print("- bmi: Body mass index")
print("- children: Number of children covered")
print("- smoker: Smoking status")
print("- region: Customer region")
print("- charges: Insurance costs (what I want to predict)")

print("\nPreparing the data for analysis...")
# Convert text categories to numbers
processed_data = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Separate features from target
features = processed_data.drop('charges', axis=1)
insurance_costs = processed_data['charges']

print("Splitting data into training and testing sets...")
# Split data - 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(features, insurance_costs, test_size=0.2, random_state=42)

print("Training my prediction model...")
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Making predictions on test data...")
# Test my model
predictions = model.predict(X_test)

# Calculate how accurate my predictions are
average_error = mean_absolute_error(y_test, predictions)
rmse_error = np.sqrt(mean_squared_error(y_test, predictions))

print(f"\nHere's how my model performed:")
print(f"Average prediction error: ${average_error:.2f}")
print(f"Root mean squared error: ${rmse_error:.2f}")

print("\nCreating charts to visualize my findings...")
plt.figure(figsize=(15, 5))

# Chart 1: BMI vs charges for smokers/non-smokers
plt.subplot(1, 3, 1)
sns.scatterplot(data=insurance_df, x='bmi', y='charges', hue='smoker', alpha=0.7)
plt.title('BMI vs Insurance Costs')
plt.xlabel('Body Mass Index')
plt.ylabel('Insurance Charges ($)')

# Chart 2: Age vs charges
plt.subplot(1, 3, 2)
sns.scatterplot(data=insurance_df, x='age', y='charges', hue='smoker', alpha=0.7)
plt.title('Age vs Insurance Costs')
plt.xlabel('Age (years)')
plt.ylabel('Insurance Charges ($)')

# Chart 3: Cost comparison smokers vs non-smokers
plt.subplot(1, 3, 3)
sns.boxplot(data=insurance_df, x='smoker', y='charges')
plt.title('Cost Comparison: Smokers vs Non-Smokers')
plt.xlabel('Smoker')
plt.ylabel('Insurance Charges ($)')

plt.tight_layout()
plt.show()

print("\nMy key observations from this analysis:")
print("- Smokers pay much higher insurance costs, which makes sense")
print("- Age strongly correlates with higher insurance costs")
print("- BMI affects costs, but smoking status seems to be the biggest factor")
print(f"- My model's average error of ${average_error:.2f} seems reasonable given the data range")

print("This was an interesting project - I learned how different factors impact insurance pricing.")
