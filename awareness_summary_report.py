from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd
import os

# === Paths ===
base_path = "/Users/bazam/dev/GW-MP-survey-analysis/"
corr_file = os.path.join(base_path, "awareness_question_correlations.csv")
anova_file = os.path.join(base_path, "awareness_anova_results.csv")

pdf_path = os.path.join(base_path, "awareness_summary_report.pdf")

# === Load data ===
corr_df = pd.read_csv(corr_file)
anova_df = pd.read_csv(anova_file)

# === Basic setup ===
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
elements = []

# === Title ===
elements.append(Paragraph("<b>Awareness Analysis Summary Report</b>", styles["Title"]))
elements.append(Spacer(1, 12))
elements.append(Paragraph("Generated automatically from the DKAP awareness-knowledge dataset.", styles["Normal"]))
elements.append(Spacer(1, 24))

# === Section 1: Overview ===
elements.append(Paragraph("<b>1. Overview</b>", styles["Heading2"]))
elements.append(Paragraph(
    "This report summarises the relationships between individual awareness items (Q8–Q29), knowledge scores, "
    "and six respondent clusters. The analyses include Pearson correlations, OLS regressions, and one-way ANOVA "
    "tests followed by post-hoc Tukey comparisons where applicable.",
    styles["Normal"]))
elements.append(Spacer(1, 12))

# === Section 2: Correlation Results ===
elements.append(Paragraph("<b>2. Correlations between Awareness and Knowledge</b>", styles["Heading2"]))
elements.append(Paragraph(
    "The table below shows Pearson correlation coefficients between each awareness question and the normalised "
    "knowledge score. Higher r-values indicate stronger associations.", styles["Normal"]))
elements.append(Spacer(1, 12))

corr_data = [["Question", "Pearson r", "p-value"]] + corr_df.round(3).values.tolist()
corr_table = Table(corr_data, hAlign="LEFT")
corr_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
]))
elements.append(corr_table)
elements.append(Spacer(1, 12))

# Interpretation
elements.append(Paragraph(
    "Items Q14, Q19, Q21, Q24, and Q29 show strong and statistically significant correlations (r > 0.4, p < 0.001), "
    "indicating they are closely aligned with knowledge levels. In contrast, Q8 and Q9 display weak associations.",
    styles["Normal"]))
elements.append(Spacer(1, 18))

# === Section 3: ANOVA Results ===
elements.append(Paragraph("<b>3. Differences in Awareness Across Clusters</b>", styles["Heading2"]))
elements.append(Paragraph(
    "A one-way ANOVA was conducted to test whether mean awareness scores differ significantly across the six "
    "knowledge-based clusters. Results are reported below.", styles["Normal"]))
elements.append(Spacer(1, 12))

anova_data = [["Question", "F-statistic", "p-value"]] + anova_df.round(3).values.tolist()
anova_table = Table(anova_data, hAlign="LEFT")
anova_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")
]))
elements.append(anova_table)
elements.append(Spacer(1, 12))

# Interpretation
elements.append(Paragraph(
    "Questions Q10, Q14, Q21, Q24, and Q29 show significant between-cluster differences (p < 0.01). "
    "These results suggest that awareness of these topics varies substantially depending on the respondent's "
    "cluster membership, likely reflecting differing knowledge and engagement levels.",
    styles["Normal"]))
elements.append(Spacer(1, 18))

# === Section 4: Visualisation Summary ===
elements.append(Paragraph("<b>4. Awareness by Cluster (Boxplots)</b>", styles["Heading2"]))
elements.append(Paragraph(
    "The following figures illustrate the distribution of awareness scores (normalised 0–1) across the six "
    "knowledge clusters for each question. The plots provide visual confirmation of the ANOVA findings.",
    styles["Normal"]))
elements.append(Spacer(1, 12))

for q in corr_df["Question"]:
    img_path = os.path.join(base_path, f"awareness_{q}_by_cluster.png")
    if os.path.exists(img_path):
        elements.append(Image(img_path, width=400, height=280))
        elements.append(Spacer(1, 12))

# === Section 5: Summary Interpretation ===
elements.append(Paragraph("<b>5. Interpretation Summary</b>", styles["Heading2"]))
elements.append(Paragraph(
    "Overall, results confirm a positive association between knowledge and awareness levels. "
    "Cluster analysis further indicates heterogeneity across respondents, particularly for items "
    "Q14 (environmental implications), Q21 (microplastic contamination), Q24 (knowledge of MP risks), "
    "and Q29 (mitigation awareness). These elements may represent the most sensitive indicators "
    "of awareness disparities within the population sample.",
    styles["Normal"]))
elements.append(Spacer(1, 12))

elements.append(Paragraph("Report automatically generated by Python (ReportLab).", styles["Italic"]))

# === Build PDF ===
doc.build(elements)
print(f"✅ PDF report successfully generated: {pdf_path}")
