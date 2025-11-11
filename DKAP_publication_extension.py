import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from statsmodels.formula.api import ols
from fpdf import FPDF
from PyPDF2 import PdfMerger

# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------
base_in = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/"
base_out = "/Users/bazam/dev/GW-MP-survey-analysis/"

# ------------------------------------------------
# DATA IMPORT
# ------------------------------------------------
knowledge = pd.read_csv(base_in + "knowledge_score_clusters.csv")
awareness = pd.read_csv(base_in + "database_awareness_questions_norm.csv")
attitude = pd.read_csv(base_in + "database_attitude_norm.csv")

# Merge composites
awareness["awareness_composite"] = awareness.drop(columns=["respondent_id"]).mean(axis=1)
attitude["attitude_composite"] = attitude.drop(columns=["respondent_id"]).mean(axis=1)

df = knowledge.merge(awareness[["respondent_id", "awareness_composite"]], on="respondent_id")
df = df.merge(attitude[["respondent_id", "attitude_composite"]], on="respondent_id")

# ------------------------------------------------
# 1️⃣ CLUSTER RADAR PLOTS
# ------------------------------------------------
cluster_summary = (
    df.groupby("cluster")[["knowledge_score", "awareness_composite", "attitude_composite"]]
    .mean()
    .reset_index()
)

# Normalize to 0–1 range for visual balance
normalized = cluster_summary.copy()
normalized.iloc[:, 1:] = (cluster_summary.iloc[:, 1:] - cluster_summary.iloc[:, 1:].min()) / (
    cluster_summary.iloc[:, 1:].max() - cluster_summary.iloc[:, 1:].min()
)

# Radar plot setup
categories = list(normalized.columns[1:])
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(10, 10))
for _, row in normalized.iterrows():
    values = row[categories].tolist()
    values += values[:1]
    plt.polar(angles, values, label=f'Cluster {int(row["cluster"])}', linewidth=2)

plt.xticks(angles[:-1], categories, color='grey', size=12)
plt.title("DKAP Cluster Profiles", size=16, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()

# ✅ Save as PNG (FPDF compatible)
radar_path = os.path.join(base_out, "DKAP_cluster_profiles.png")
plt.savefig(radar_path, dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------
# 2️⃣ REGRESSION SUMMARY (Demographics → DKAP composites)
# ------------------------------------------------
# Use demographics that were already one-hot encoded in your previous dataset
demo_path = os.path.join(base_in, "demographics_clean.csv")
if os.path.exists(demo_path):
    demo = pd.read_csv(demo_path)
    df_full = df.merge(demo, on="respondent_id")
else:
    print("⚠️ demographics_encoded.csv not found; skipping regression summary.")
    df_full = None

regression_summary = []

if df_full is not None:
    dep_vars = ["knowledge_score", "awareness_composite", "attitude_composite"]
    demo_vars = "+".join(demo.columns[1:])
    for var in dep_vars:
        model = ols(f"{var} ~ {demo_vars}", data=df_full).fit()
        summary_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coef": model.params.values,
            "P>|t|": model.pvalues.values,
            "R2": model.rsquared
        })
        summary_df["Dependent"] = var
        regression_summary.append(summary_df)

    regression_df = pd.concat(regression_summary)
    regression_df.to_csv(os.path.join(base_out, "dkap_regression_summary.csv"), index=False)

# ------------------------------------------------
# 3️⃣ DKAP CORRELATION HEATMAP
# ------------------------------------------------
corr = df[["knowledge_score", "awareness_composite", "attitude_composite"]].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("DKAP Correlation Heatmap")
corr_path = os.path.join(base_out, "DKAP_Correlation_Heatmap.png")
plt.tight_layout()
plt.savefig(corr_path)
plt.close()

# ------------------------------------------------
# 4️⃣ APPEND TO EXISTING REPORT
# ------------------------------------------------
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "DKAP Publication Extension Summary", ln=True, align="C")

pdf.set_font("Helvetica", "", 12)
pdf.multi_cell(0, 8, """
This document complements the DKAP Summary Report with visual and analytical enhancements:
1. Cluster-based DKAP radar profiles.
2. Regression summary across demographic variables.
3. Correlation heatmap connecting the main DKAP composites.
""")

pdf.image(radar_path, x=20, y=70, w=160)
pdf.image(corr_path, x=30, y=200, w=140)

extension_path = os.path.join(base_out, "DKAP_Publication_Extension.pdf")
pdf.output(extension_path)

# Merge with main report
merged_path = os.path.join(base_out, "DKAP_Publication_Report.pdf")
merger = PdfMerger()
for file in [
    os.path.join(base_out, "DKAP_Summary_Report.pdf"),
    extension_path
]:
    merger.append(file)
merger.write(merged_path)
merger.close()

print(f"✅ DKAP publication-ready materials generated:\n- {merged_path}\n- {radar_path}\n- {corr_path}")
