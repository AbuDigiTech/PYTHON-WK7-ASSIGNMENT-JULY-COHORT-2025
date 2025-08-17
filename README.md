# PYTHON-WK7-ASSIGNMENT-JULY-COHORT-2025

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map species to their names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset structure:")
print(df.info())
print(df.describe())

# Task 1: Load and Explore the Dataset
print("\nMissing values:")
print(df.isnull().sum())

# No missing values in this dataset, but if there were, we could handle them like this:
# df.fillna(df.mean(), inplace=True)  # Fill missing values with mean
# df.dropna(inplace=True)  # Drop rows with missing values

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute mean
print("\nMean of features by species:")
print(df.groupby('species').mean())

# Task 3: Data Visualization
# Line chart (not applicable for this dataset, but we can create a line chart for each feature)
for feature in iris.feature_names:
    plt.figure(figsize=(8, 6))
    for species in df['species'].unique():
        species_df = df[df['species'] == species]
        plt.plot(species_df[feature], label=species)
    plt.title(f"Line chart of {feature} by species")
    plt.xlabel("Index")
    plt.ylabel(feature)
    plt.legend()
    plt.show()

# Bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title("Bar chart of sepal length by species")
plt.xlabel("Species")
plt.ylabel("Sepal length (cm)")
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal length (cm)'], kde=True)
plt.title("Histogram of sepal length")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter plot of sepal length vs petal length")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.legend()
plt.show()
