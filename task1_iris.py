# TASK 1 - IRIS DATASET 
print("🎯 STARTING TASK 1 - IRIS DATASET")

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("✅ DATA LOADED SUCCESSFULLY!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# Create visualizations
print("\n📊 CREATING VISUALIZATIONS...")
plt.figure(figsize=(15, 10))

# Scatter plot
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species')
plt.title('Sepal Length vs Width')

# Histogram
plt.subplot(2, 2, 2)
df['sepal length (cm)'].hist()
plt.title('Distribution of Sepal Length')

# Box plot
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='species', y='petal length (cm)')
plt.title('Petal Length by Species')

plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='species', y='sepal width (cm)')
plt.title('Sepal Width by Species')

plt.tight_layout()
plt.show()
