#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final DKAP Analysis Pipeline
Author: Barbara Zambelli
Purpose: Integrate Demographics (D), Knowledge (K), Awareness (A), and Attitude/Practice (P)
         into a comprehensive analytical framework.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# === PATH SETUP ===
base_data = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/"
base_output = "/Users/bazam/dev/GW-MP-survey-analysis/"

# === FILE INPUTS ===
knowledge_file = base_data + "knowledge_score_clusters.csv"
awareness_file = base_data + "database_awareness_questions_norm.csv"
attitude_file = base_data + "database_attitude_norm.csv"
demographics_file = base_data + "demographics_clean.csv" 

# === LOAD DATASETS ===
df_k = pd.read_csv(knowledge_file)
df_a = pd.read_csv(awareness_file)
df_p = pd.read_csv(attitude_file)

# === COMPUTE COMPOSITES ===
df_a["awareness_composite"] = df_a.drop(columns=["respondent_id"]).mean(axis=1)
df_p["attitude_composite"] = df_p.drop(columns=["respondent_id"]).mean(axis=1)

# === MERGE ALL ===
df = df_k.merge(df_a[["respondent_id", "awareness_composite"]], on="respondent_id", how="inner")
df = df.merge(df_p[["respondent_id", "attitude_composite"]], on="respondent_id", how="inner")

# === BASIC DESCRIPTIVES ===
desc = df[["knowledge_score", "awareness_composite", "attitude_composite"]].describe()
desc.to_csv(base_output + "dkap_descriptive_summary.csv", index=True)

# === CORRELATION MATRIX ===
corr = df[["knowledge_score", "awareness_composite", "attitude_composite"]].corr(method="pearson")
corr.to_csv(base_output + "dkap_correlations.csv", index=True)

# === CLUSTER-LEVEL ANALYSIS (ANOVA) ===
anova_results = {}
for var in ["knowledge_score", "awareness_composite", "attitude_composite"]:
    model = ols(f"{var} ~ C(cluster)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[var] = anova_table

# === TUKEY POST-HOC ===
tukey_results = {}
for var in ["knowledge_score", "awareness_composite", "attitude_composite"]:
    tukey = pairwise_tukeyhsd(endog=df[var], groups=df["cluster"], alpha=0.05)
    tukey_results[var] = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_results[var].to_csv(base_output + f"tukey_{var}.csv", index=False)

# === DEMOGRAPHIC VALIDATION (if file available) ===
if demographics_file:
    df_demo = pd.read_csv(demographics_file)
    df_full = df.merge(df_demo, on="respondent_id", how="left")
    df_full.columns = df_full.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    for var in ["knowledge_score", "awareness_composite", "attitude_composite"]:
        print(f"\nRegression on demographics for {var}:")
        demo_vars = "+".join(df_demo.columns.difference(["respondent_id"]))
        model = ols(f"{var} ~ {demo_vars}", data=df_full).fit()

        print(model.summary())

# === VISUALIZATIONS ===
sns.set(style="whitegrid")

# Scatter matrix (DKAP relationships)
g = sns.pairplot(df[["knowledge_score", "awareness_composite", "attitude_composite"]], diag_kind="kde")
g.fig.suptitle("Scatterplot Matrix: Knowledge, Awareness, Attitude", y=1.02)
g.fig.savefig(base_output + "dkap_scatter_matrix.png", bbox_inches="tight")

# Cluster radar / heatmap
cluster_means = df.groupby("cluster")[["knowledge_score", "awareness_composite", "attitude_composite"]].mean()
plt.figure(figsize=(8, 6))
sns.heatmap(cluster_means, annot=True, cmap="viridis", cbar_kws={'label': 'Mean Score'})
plt.title("DKAP Cluster Profile (Mean Scores)")
plt.savefig(base_output + "dkap_cluster_heatmap.png", bbox_inches="tight")
plt.close()

# === PDF REPORT ===
report_path = base_output + "DKAP_Summary_Report.pdf"
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("<b>DKAP Analysis Summary</b>", styles["Title"]))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>1. Descriptive Statistics</b>", styles["Heading2"]))
story.append(Paragraph(desc.to_html(), styles["Normal"]))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>2. Correlation Matrix</b>", styles["Heading2"]))
story.append(Paragraph(corr.to_html(), styles["Normal"]))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>3. Cluster Differences (ANOVA)</b>", styles["Heading2"]))
for var, table in anova_results.items():
    story.append(Paragraph(f"<b>{var}</b>", styles["Heading3"]))
    story.append(Paragraph(table.to_html(), styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>4. Visualizations</b>", styles["Heading2"]))
story.append(Image(base_output + "dkap_scatter_matrix.png", width=6*inch, height=6*inch))
story.append(Spacer(1, 0.2 * inch))
story.append(Image(base_output + "dkap_cluster_heatmap.png", width=6*inch, height=4*inch))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>End of DKAP Summary Report</b>", styles["Normal"]))
doc = SimpleDocTemplate(report_path, pagesize=A4)
doc.build(story)

print(f"\n✅ DKAP analysis complete.\nResults saved to: {base_output}")
print(f"Generated files:\n - dkap_descriptive_summary.csv\n - dkap_correlations.csv\n - tukey_[variable].csv\n - DKAP_Summary_Report.pdf")
