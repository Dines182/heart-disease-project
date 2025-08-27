import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # for interactive plots (or "Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(path="data/heart.csv"):
    """Load dataset from CSV."""
    return pd.read_csv(path)


def visualize_distributions(df):
    """Plot age & cholesterol distributions grouped by heart disease status."""
    # Age
    sns.histplot(data=df, x="age", hue="target", bins=20, kde=True, palette="Set1")
    plt.title("Age Distribution by Heart Disease Status")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    # Cholesterol
    sns.histplot(data=df, x="chol", hue="target", bins=30, kde=True, palette="Set2")
    plt.title("Cholesterol Distribution by Heart Disease Status")
    plt.xlabel("Cholesterol (mg/dl)")
    plt.ylabel("Count")
    plt.show()


def test_cholesterol(df):
    """Independent t-test: cholesterol vs heart disease."""
    chol_no_hd = df[df['target'] == 0]['chol']
    chol_hd = df[df['target'] == 1]['chol']

    t_stat, p_val = stats.ttest_ind(chol_no_hd, chol_hd)
    print("\nHypothesis Test: Cholesterol vs Heart Disease")
    print("T-statistic:", round(t_stat, 4))
    print("P-value:", round(p_val, 4))

    if p_val < 0.05:
        print("✅ Significant difference: Cholesterol levels differ between groups")
    else:
        print("❌ No significant difference found")


def test_sex_vs_disease(df):
    """Chi-square test: sex vs heart disease."""
    contingency_table = pd.crosstab(df['sex'], df['target'])
    chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency_table)

    print("\nChi-square Test: Sex vs Heart Disease")
    print("Chi2:", round(chi2, 4))
    print("P-value:", round(p_val_chi, 4))

    if p_val_chi < 0.05:
        print("✅ Significant association: Sex and heart disease are related")
    else:
        print("❌ No significant association found")


if __name__ == "__main__":
    df = load_data()
    visualize_distributions(df)
    test_cholesterol(df)
    test_sex_vs_disease(df)
