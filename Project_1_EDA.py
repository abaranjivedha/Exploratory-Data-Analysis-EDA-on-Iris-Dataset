import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris_data = pd.read_csv("iris.data.csv", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

# Initial inspection of the data
print("First 5 rows of the dataset:")
print(iris_data.head())
print("\nBasic statistics:")
print(iris_data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(iris_data.isnull().sum())

# Pairplot for pairwise relationships
sns.pairplot(iris_data, hue="class")
plt.suptitle("Pairwise Relationships of Iris Features", y=1.02)
plt.show()

# Box plot for sepal length by species
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_data, x="class", y="sepal_length")
plt.title("Box Plot of Sepal Length by Iris Species")
plt.show()

# Box plot for all features by species
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=iris_data, x="class", y=feature)
    plt.title(f"Box Plot of {feature.replace('_', ' ').title()} by Iris Species")
    plt.show()

# Distribution of each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=iris_data, x=feature, hue="class", kde=True, element="step", stat="density", common_norm=False)
    plt.title(f"Distribution of {feature.replace('_', ' ').title()} by Iris Species")
    plt.show()

# Correlation matrix
corr_matrix = iris_data.drop(columns="class").corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Iris Features")
plt.show()

# Summary of key findings
print("\nSummary of Key Findings:")
print("1. The pairplot shows distinct clusters for each species, indicating that the features can differentiate between the species.")
print("2. Box plots reveal that petal length and petal width have the most variation between species.")
print("3. Histograms show that the distributions of sepal length, sepal width, petal length, and petal width vary significantly between species.")
print("4. The correlation matrix indicates a strong positive correlation between petal length and petal width.")
