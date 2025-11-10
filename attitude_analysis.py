import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# === Paths ===
base_input = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/"
base_output = "/Users/bazam/dev/GW-MP-survey-analysis/"

attitude_file = os.path.join(base_input, "database_attitude_norm.csv")
knowledge_file = os.path.join(base_input, "knowledge_score_clusters.csv")
awareness_file = os.path.join(base_input, "database_awareness_questions_norm.csv")

# === Load datasets ===
att = pd.read_csv(attitude_file)
know = pd.read_csv(knowledge_file)
aware = pd.read_csv(awareness_file)

# === Merge datasets ===
df = att.merge(know, on="respondent_id", how="left").merge(aware, on="respondent_id", how="left")

print(f"Merged dataset shape: {df.shape}")

# === Define lists ===
attitude_qs = ["Q2", "Q3", "Q4", "Q7", "Q20", "Q30"]
awareness_qs = ["Q8", "Q9", "Q10", "Q14", "Q19", "Q21", "Q24", "Q29"]

# === Output containers ===
corr_results = []
anova_results = []

# === 1. Correlations with Knowledge and Awareness ===
for q in attitude_qs:
    # correlation with knowledge
    r_k, p_k = pearsonr(df[q], df["knowledge_score"])
    corr_results.append(["Knowledge", q, r_k, p_k])
    
    # correlation with mean awareness
    df["awareness_mean"] = df[awareness_qs].mean(axis=1)
    r_a, p_a = pearsonr(df[q], df["awareness_mean"])
    corr_results.append(["Awareness", q, r_a, p_a])

corr_df = pd.DataFrame(corr_results, columns=["Reference", "Question", "Pearson_r", "p_value"])
corr_df.to_csv(os.path.join(base_output, "attitude_correlations.csv"), index=False)
print("📈 Correlations saved to attitude_correlations.csv")

# === 2. ANOVA across Knowledge Clusters ===
for q in attitude_qs:
    model = ols(f"{q} ~ C(cluster)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    F = anova_table["F"][0]
    p = anova_table["PR(>F)"][0]
    anova_results.append([q, F, p])

anova_df = pd.DataFrame(anova_results, columns=["Question", "F_statistic", "p_value"])
anova_df.to_csv(os.path.join(base_output, "attitude_anova_results.csv"), index=False)
print("📊 ANOVA results saved to attitude_anova_results.csv")

# === 3. Post-hoc Tukey (only for significant ANOVAs) ===
for q, pval in zip(anova_df["Question"], anova_df["p_value"]):
    if pval < 0.05:
        tukey = pairwise_tukeyhsd(endog=df[q], groups=df["cluster"], alpha=0.05)
        print(f"\nPost-hoc Tukey for {q}:")
        print(tukey)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df.to_csv(os.path.join(base_output, f"attitude_{q}_tukey.csv"), index=False)

# === 4. Boxplots by cluster ===
sns.set(style="whitegrid")
for q in attitude_qs:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="cluster", y=q, data=df, palette="Set3")
    plt.title(f"Distribution of {q} across clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Normalised Attitude/Practice Score")
    plt.tight_layout()
    out_path = os.path.join(base_output, f"attitude_{q}_by_cluster.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# === 5. Summary output ===
print("\n✅ Analysis complete.")
print("Correlation results:", corr_df.shape)
print("ANOVA results:", anova_df.shape)
