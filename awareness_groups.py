import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis

def awareness_descriptive_analysis(csv_file):
    """
    Perform descriptive statistics and distribution analysis for awareness subscales.
    Input:
        csv_file (str): Path to CSV containing respondent_id + awareness groups (aw1, aw2, aw3)
    """

    # --- Load dataset ---
    df = pd.read_csv(csv_file)

    # Identify awareness columns (exclude respondent_id)
    awareness_cols = [col for col in df.columns if col != "respondent_id"]

    print("\n=== Awareness Descriptive Statistics ===\n")

    # --- Prepare summary table ---
    summary_data = []
    for col in awareness_cols:
        data = df[col].dropna()
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        skew_val = skew(data)
        kurt_val = kurtosis(data)
        shapiro_stat, shapiro_p = shapiro(data)

        summary_data.append({
            "Awareness Group": col,
            "Mean": round(mean_val, 3),
            "Median": round(median_val, 3),
            "Std. Dev": round(std_val, 3),
            "Skewness": round(skew_val, 3),
            "Kurtosis": round(kurt_val, 3),
            "Shapiro-Wilk p-value": round(shapiro_p, 4),
            "Normality": "Yes" if shapiro_p > 0.05 else "No"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    # Save summary
    summary_df.to_csv("awareness_descriptive_summary.csv", index=False)
    print("\n✅ Saved results to 'awareness_descriptive_summary.csv'")

        # --- Plot distributions with consistent colors ---
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=len(awareness_cols))

    for color, col in zip(palette, awareness_cols):
        sns.histplot(df[col], bins=6, kde=False, alpha=0.25, color=color)
        sns.kdeplot(df[col], fill=False, color=color, lw=2, label=col)

    plt.xlabel("Score (0–5)")
    plt.ylabel("Density / Count")
    plt.title("Distribution of Awareness Scores")
    plt.legend(title="Awareness Groups")
    plt.tight_layout()
    plt.savefig("awareness_distributions.png", dpi=300)
    plt.show()


    print("\n✅ Plots saved as 'awareness_distributions.png'")
    print("\nInterpretation tip:")
    print("- p > 0.05 in Shapiro-Wilk means approximately normal distribution.")
    print("- High skewness → asymmetric distribution; high kurtosis → heavy tails or peakedness.")

    return summary_df

results = awareness_descriptive_analysis("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/database_awareness_AW1.csv")
