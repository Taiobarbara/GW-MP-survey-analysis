from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import pandas as pd
import os

# === File paths ===
base_path = "/Users/bazam/dev/GW-MP-survey-analysis/"
descriptives_csv = os.path.join(base_path, "awareness_analysis_descriptives.csv")
correlations_csv = os.path.join(base_path, "awareness_analysis_knowledge_correlations.csv")
reg1_txt = os.path.join(base_path, "awareness_analysis_factor1_norm_regression.txt")
reg2_txt = os.path.join(base_path, "awareness_analysis_factor2_norm_regression.txt")
reg3_txt = os.path.join(base_path, "awareness_analysis_factor3_norm_regression.txt")
dist_plot = os.path.join(base_path, "awareness_analysis_distributions.png")

output_pdf = os.path.join(base_path, "awareness_analysis_report.pdf")

# === Load data ===
desc = pd.read_csv(descriptives_csv)
corr = pd.read_csv(correlations_csv)

with open(reg1_txt) as f:
    reg1 = f.read().splitlines()[-6:]
with open(reg2_txt) as f:
    reg2 = f.read().splitlines()[-6:]
with open(reg3_txt) as f:
    reg3 = f.read().splitlines()[-6:]

# === PDF setup ===
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Heading", fontSize=14, leading=16, spaceAfter=10, spaceBefore=10, alignment=1))
styles.add(ParagraphStyle(name="SubHeading", fontSize=12, leading=14, spaceAfter=8, textColor=colors.HexColor("#003366")))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=13))

doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                        rightMargin=40, leftMargin=40,
                        topMargin=50, bottomMargin=40)

elements = []

# === Title ===
elements.append(Paragraph("Awareness Analysis Report", styles["Heading"]))
elements.append(Paragraph("DKAP Framework – Awareness Component", styles["SubHeading"]))
elements.append(Spacer(1, 12))

# === Section 1: Descriptive Statistics ===
elements.append(Paragraph("<b>1. Descriptive Statistics</b>", styles["SubHeading"]))
elements.append(Paragraph(
    "The table below summarizes the central tendency and dispersion for each awareness factor "
    "(normalized between 0 and 1).", styles["Body"]))

# Table
desc_table = [desc.columns.to_list()] + desc.values.tolist()
t = Table(desc_table, hAlign='LEFT')
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('ALIGN', (1, 1), (-1, -1), 'CENTER')
]))
elements.append(t)
elements.append(Spacer(1, 12))

elements.append(Paragraph(
    "Factor 1 shows moderate variability (SD≈0.37), Factor 2 is more stable (SD≈0.14), and Factor 3 shows moderate spread (SD≈0.21).",
    styles["Body"]))
elements.append(Spacer(1, 10))

# === Section 2: Distribution Plots ===
elements.append(Paragraph("<b>2. Distributions</b>", styles["SubHeading"]))
elements.append(Paragraph(
    "The figure below displays the distribution of normalized awareness factors. "
    "These plots provide visual insight into the spread and central tendency of awareness scores.",
    styles["Body"]))
elements.append(Spacer(1, 10))
if os.path.exists(dist_plot):
    elements.append(Image(dist_plot, width=400, height=250))
else:
    elements.append(Paragraph("[Distribution plot not found]", styles["Body"]))
elements.append(Spacer(1, 12))

# === Section 3: Correlations ===
elements.append(Paragraph("<b>3. Correlation with Knowledge Score</b>", styles["SubHeading"]))
elements.append(Paragraph(
    "Correlation coefficients indicate the strength of association between knowledge and awareness factors.", styles["Body"]))

corr_table = [corr.columns.to_list()] + corr.values.tolist()
t2 = Table(corr_table, hAlign='LEFT')
t2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('ALIGN', (1, 1), (-1, -1), 'CENTER')
]))
elements.append(t2)
elements.append(Spacer(1, 10))
elements.append(Paragraph(
    "Factor 1 (r≈0.72) and Factor 3 (r≈0.68) show strong positive relationships with Knowledge, "
    "suggesting that higher knowledge corresponds to greater awareness in these domains. "
    "Factor 2 (r≈0.11) exhibits a weak relationship, indicating it may represent a distinct perception component.",
    styles["Body"]))
elements.append(Spacer(1, 12))

# === Section 4: Regression Results ===
elements.append(Paragraph("<b>4. Regression Analyses</b>", styles["SubHeading"]))
elements.append(Paragraph(
    "Simple linear regressions were conducted with knowledge score as a predictor for each awareness factor. "
    "Coefficients (β) represent the expected change in awareness per unit increase in knowledge.", styles["Body"]))
elements.append(Spacer(1, 10))

def add_reg_summary(factor, lines):
    elements.append(Paragraph(f"<b>{factor}</b>", styles["Body"]))
    for line in lines:
        elements.append(Paragraph(line, styles["Body"]))
    elements.append(Spacer(1, 6))

add_reg_summary("Factor 1 Regression Summary", reg1)
add_reg_summary("Factor 2 Regression Summary", reg2)
add_reg_summary("Factor 3 Regression Summary", reg3)

elements.append(Paragraph(
    "Results indicate that knowledge score significantly predicts all awareness components, "
    "with the strongest effects on Factors 1 and 3 (β≈1.28 and β≈0.67).", styles["Body"]))
elements.append(Spacer(1, 12))

# === Section 5: Interpretation ===
elements.append(Paragraph("<b>5. Interpretation and Implications</b>", styles["SubHeading"]))
elements.append(Paragraph(
    "The awareness measures show good variability and strong association with knowledge in two of the three domains. "
    "This suggests that participants with higher factual knowledge tend to report higher awareness of environmental issues and microplastic impacts. "
    "The weaker correlation in Factor 2 indicates that some perceptual aspects of awareness may be less dependent on factual knowledge.",
    styles["Body"]))
elements.append(Spacer(1, 20))
elements.append(Paragraph("End of Report.", styles["Body"]))

# === Build PDF ===
doc.build(elements)
print(f"✅ PDF report successfully saved as: {output_pdf}")
