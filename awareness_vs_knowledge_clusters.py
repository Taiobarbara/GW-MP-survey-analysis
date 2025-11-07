import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# === File paths ===
base_input = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/"
base_output = "/Users/bazam/dev/GW-MP-survey-analysis/"

awareness_file = os.path.join(base_input, "database_awareness_questions_norm.csv")
knowledge_file = os.path.join(base_input, "knowledge_score_clusters.csv")

output_corr = os.path.join(base_output, "awareness_question_correlations.csv")
output_reg = os.path.join(base_output, "awareness_question_regressions.txt")
output_anova = os.path.join(base_output, "awareness_anova_results.csv")

# === Load data ===
df_aw = pd.read_csv(awareness_file)
df_know = pd.read_csv(knowledge_file)

# Merge on respondent_id
df = df_aw.merge(df_know, on="respondent_id", how="left")

# Rename cluster column for clarity
df.rename(columns={"cluster": "cluster_label"}, inplace=True)

print(f"✅ Data merged successfully: {df.shape[0]} respondents, {df.shape[1]} columns")

# Identify awareness question columns
awareness_cols = [c for c in df.columns if c.startswith("Q")]

# === 1. Correlation with Knowledge Score ===
corrs = []
for q in awareness_cols:
    valid = df[[q, "knowledge_score"]].dropna()
    if len(valid) > 2:
        r, p = pearsonr(valid[q], valid["knowledge_score"])
        corrs.append({"Question": q, "Pearson_r": r, "p_value": p})

corr_df = pd.DataFrame(corrs)
corr_df.to_csv(output_corr, index=False)
print(f"📈 Correlation results saved to: {output_corr}")
print(corr_df.round(3))

# === 2. Simple regression (Knowledge → Awareness) ===
with open(output_reg, "w") as f:
    for q in awareness_cols:
        X = sm.add_constant(df["knowledge_score"])
        y = df[q]
        model = sm.OLS(y, X, missing="drop").fit()
        f.write(f"\n=== Regression for {q} ===\n")
        f.write(str(model.summary()))
        f.write("\n" + "="*80 + "\n")

print(f"📘 Regression results saved to: {output_reg}")

# === 3. Awareness differences across clusters (ANOVA + Tukey) ===
anova_results = []
for q in awareness_cols:
    groups = [g[q].dropna() for _, g in df.groupby("cluster_label")]
    if len(groups) > 1:
        F, p = f_oneway(*groups)
        anova_results.append({"Question": q, "F_statistic": F, "p_value": p})

anova_df = pd.DataFrame(anova_results)
anova_df.to_csv(output_anova, index=False)
print(f"📊 ANOVA results saved to: {output_anova}")
print(anova_df.round(3))

# === 4. Visualization: Awareness by Cluster ===
for q in awareness_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="cluster_label", y=q, data=df, palette="pastel")
    plt.title(f"{q} Awareness by Cluster")
    plt.ylabel("Normalized awareness (0–1)")
    plt.xlabel("Knowledge Cluster")
    plt.tight_layout()
    fig_path = os.path.join(base_output, f"awareness_{q}_by_cluster.png")
    plt.savefig(fig_path)
    plt.close()

    # === Post-hoc Tukey test if ANOVA significant ===
    p_value = anova_df.loc[anova_df["Question"] == q, "p_value"].values[0]
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(df[q], df["cluster_label"])
        print(f"\nPost-hoc Tukey for {q}:")
        print(tukey.summary())

print("✅ All analyses completed successfully.")
