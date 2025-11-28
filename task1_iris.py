# Task 1 - Understanding Flower Data
# Looking at different types of iris flowers and their measurements

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

print("Working on Task 1 - Flower Data Analysis")

# Get the flower data
print("Loading the iris flower dataset...")
flower_data = load_iris()
flower_df = pd.DataFrame(flower_data.data, columns=flower_data.feature_names)
flower_df['flower_type'] = flower_data.target

print("Data loaded successfully!")
print(f"We have {len(flower_df)} flowers in our data")
print("Each flower has these measurements: sepal length, sepal width, petal length, petal width")
print("\nFirst few flowers in our data:")
print(flower_df.head())

print("\nCreating charts to visualize the patterns...")

# Create charts to visualize the data
plt.figure(figsize=(15, 10))

# Chart 1: Compare sepal sizes
plt.subplot(2, 2, 1)
sns.scatterplot(data=flower_df, x='sepal length (cm)', y='sepal width (cm)', hue='flower_type', palette='viridis')
plt.title('Sepal Size Comparison')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Chart 2: Distribution of sepal lengths
plt.subplot(2, 2, 2)
sns.histplot(data=flower_df, x='sepal length (cm)', hue='flower_type', kde=True)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')

# Chart 3: Compare petal lengths
plt.subplot(2, 2, 3)
sns.boxplot(data=flower_df, x='flower_type', y='petal length (cm)')
plt.title('Petal Length by Flower Type')
plt.xlabel('Flower Type (0,1,2 = Setosa,Versicolor,Virginica)')
plt.ylabel('Petal Length (cm)')

# Chart 4: Compare sepal widths
plt.subplot(2, 2, 4)
sns.boxplot(data=flower_df, x='flower_type', y='sepal width (cm)')
plt.title('Sepal Width by Flower Type')
plt.xlabel('Flower Type (0,1,2 = Setosa,Versicolor,Virginica)')
plt.ylabel('Sepal Width (cm)')

plt.tight_layout()
plt.show()

print("From these charts, I can see that different flower types have distinct measurement patterns.")
print("The Setosa flowers (type 0) seem to have smaller petals but wider sepals.")
