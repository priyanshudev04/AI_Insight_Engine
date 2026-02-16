import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(data):
    numeric_data = data.select_dtypes(include=["number"])

    if numeric_data.shape[1] > 1:
        plt.figure(figsize=(6,4))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric columns for correlation.")
