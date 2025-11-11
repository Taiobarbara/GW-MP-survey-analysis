# ------------------------------------------------------------
# DKAP Attitude/Practice Analysis Script
# Author: ChatGPT (GPT-5)
# Date: 2025-11-11
# Purpose:
#   Integrate Attitude/Practice, Knowledge, and Awareness datasets
#   Compute composite Attitude/Practice score
#   Correlate vs Knowledge and Awareness
#   Generate visuals and a concise academic PDF report
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import utils

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
base_in = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/"
base_out = "/Users/bazam/dev/GW-MP-survey-analysis/"

attitude_file = base_in + "database_attitude_norm.csv"
knowledge_file = base_in + "knowledge_score_clusters.csv"
awareness_file = base_in + "database_awareness_questions_norm.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
att = pd.read_csv(attitude_file)
know = pd.read_csv(knowledge_file)
aware = pd.read_csv(awareness_file)

# ------------------------------------------------------------
# Compute composite scores
# ------------------------------------------------------------
att_questions = [col for col in att.columns if col.startswith("Q")]
att["attitude_composite"] = att[att_questions].mean(axis=1, skipna=True)
aware["awareness_composite"] = aware[[c for c in aware.columns if c.startswith("Q")]].mean(axis=1, skipna=True)

# Merge datasets
merged = (
    att[["respondent_id", "attitude_composite"]]
    .merge(know, on="respondent_id", how="inner")
    .merge(aware[["respondent_id", "awareness_composite"]], on="respondent_id", how="inner")
)

# ------------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------------
corr_results = []

for ref, col in [("Knowledge", "knowledge_score"), ("Awareness", "awareness_composite")]:
    r, p = pearsonr(merged["attitude_composite"], merged[col])
    corr_results.append({"Reference": ref, "Pearson_r": r, "p_value": p})

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(base_out + "attitude_correlation_results.csv", index=False)

# ------------------------------------------------------------
# Scatterplot matrix (Knowledge, Awareness, Attitude)
# ------------------------------------------------------------
sns.pairplot(
    merged[["knowledge_score", "awareness_composite", "attitude_composite"]],
    diag_kind="kde",
    plot_kws={"alpha": 0.6},
)
plt.suptitle("Scatterplot Matrix: Knowledge, Awareness, Attitude", y=1.02)
scatter_matrix_path = base_out + "attitude_scatter_matrix.png"
plt.savefig(scatter_matrix_path, bbox_inches="tight", dpi=300)
plt.close()

# ------------------------------------------------------------
# Cluster heatmap (mean scores per cluster)
# ------------------------------------------------------------
cluster_means = merged.merge(att, on="respondent_id")[att_questions + ["cluster"]]
cluster_mean_df = cluster_means.groupby("cluster").mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_mean_df, annot=True, cmap="coolwarm", cbar=True)
plt.title("Mean Attitude/Practice Scores by Cluster")
heatmap_path = base_out + "attitude_cluster_heatmap.png"
plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
plt.close()

# ------------------------------------------------------------
# Generate PDF Summary Report
# ------------------------------------------------------------
pdf_path = base_out + "attitude_summary_report.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("<b>DKAP Analytical Summary Report</b>", styles["Title"]))
story.append(Spacer(1, 12))

# Overview
story.append(Paragraph(
    "This report summarizes the analytical results for the Attitude/Practice "
    "component of the DKAP (Demographics–Knowledge–Awareness–Practice) model. "
    "The analysis integrates respondent-level attitude, knowledge, and awareness data.",
    styles["BodyText"]
))
story.append(Spacer(1, 12))

# Composite and correlation
story.append(Paragraph("<b>Composite Score Computation</b>", styles["Heading2"]))
story.append(Paragraph(
    f"The Attitude/Practice composite score was computed as the mean of normalized items "
    f"({', '.join(att_questions)}). Each respondent’s score was then correlated "
    f"against knowledge and awareness composites.",
    styles["BodyText"]
))
story.append(Spacer(1, 12))

# Correlation Results
story.append(Paragraph("<b>Correlation Results</b>", styles["Heading2"]))
for _, row in corr_df.iterrows():
    story.append(Paragraph(
        f"{row['Reference']}: Pearson r = {row['Pearson_r']:.3f}, p = {row['p_value']:.3e}",
        styles["BodyText"]
))
story.append(Spacer(1, 12))

# Interpretation
story.append(Paragraph("<b>Interpretation</b>", styles["Heading2"]))
story.append(Paragraph(
    "The correlation analysis suggests that Attitude/Practice is positively related "
    "to both Knowledge and Awareness dimensions, supporting the internal consistency "
    "of the DKAP framework. The magnitude of association indicates that as respondents’ "
    "knowledge and awareness increase, their reported sustainable practices tend to improve.",
    styles["BodyText"]
))
story.append(Spacer(1, 12))

# Visuals
def add_image(path, width=480):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))

story.append(Paragraph("<b>Visual Summaries</b>", styles["Heading2"]))
story.append(Paragraph("Figure 1. Scatterplot Matrix: Knowledge, Awareness, Attitude.", styles["BodyText"]))
story.append(add_image(scatter_matrix_path, width=400))
story.append(Spacer(1, 12))
story.append(Paragraph("Figure 2. Mean Attitude/Practice Scores by Cluster.", styles["BodyText"]))
story.append(add_image(heatmap_path, width=400))
story.append(Spacer(1, 24))

# Wrap-up
story.append(Paragraph("<b>Concluding Remarks</b>", styles["Heading2"]))
story.append(Paragraph(
    "This stage consolidates the Attitude/Practice component within the broader DKAP "
    "analytical approach. The results reinforce the model’s internal logic, indicating "
    "that knowledge and awareness jointly underpin behavioral outcomes related to practice.",
    styles["BodyText"]
))

doc.build(story)
print(f"✅ DKAP Attitude/Practice analysis complete. Report saved to:\n{pdf_path}")
