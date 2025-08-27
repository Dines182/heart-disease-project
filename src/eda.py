# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path="data/heart.csv"):
    """Load dataset and return dataframe"""
    df = pd.read_csv(path)
    return df


def run_eda(df):
    """Run Exploratory Data Analysis with plots"""
    
    # Show dataset info
    print(df.head())
    print(df.describe())
    print("Missing values:\n", df.isnull().sum())

    # Class distribution
    sns.countplot(x='target', data=df)
    plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Age distribution
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title("Age Distribution")
    plt.show()

    # Cholesterol distribution
    sns.histplot(df['chol'], bins=30, kde=True, color="teal")
    plt.title("Cholesterol Distribution")
    plt.xlabel("Cholesterol (mg/dl)")
    plt.ylabel("Count")
    plt.show()

    # Resting Blood Pressure distribution
    sns.histplot(df['trestbps'], bins=30, kde=True, color="purple")
    plt.title("Resting Blood Pressure Distribution")
    plt.xlabel("Resting Blood Pressure (mm Hg)")
    plt.ylabel("Count")
    plt.show()

    # Cholesterol vs Target
    sns.boxplot(x='target', y='chol', data=df, palette="Set2")
    plt.title("Cholesterol Levels by Heart Disease Status")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Cholesterol (mg/dl)")
    plt.show()

    # Resting BP vs Target
    sns.boxplot(x='target', y='trestbps', data=df, palette="Set1")
    plt.title("Resting Blood Pressure by Heart Disease Status")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel("Resting Blood Pressure (mm Hg)")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    run_eda(df)
