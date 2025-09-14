import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset (adjust file path)
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey_transformed_3.csv"
df = pd.read_csv(file_path, header=[0,1,2])

# Make respondent_id the index
if ('respondent_id', 'respondent_id', 'respondent_id') in df.columns:
    df = df.set_index(('respondent_id', 'respondent_id', 'respondent_id'))

# Identify Likert scale questions
likert_questions = {"Q6", "Q15", "Q17", "Q26"}

# --- DEMOGRAPHIC SECTION ---
demographics = df.loc[:, df.columns.get_level_values(0) == "demographic"]

print("Available demographic variables:")
print(demographics.columns.get_level_values(1).unique())

# === General function to analyze responses by demographic variable ===
def analyze_by_demographic(demo_keyword: str): 
    """
    demo_keyword: string to match demographic question (e.g., 'age', 'gender', 'education', 'country').
    """
    demo_cols = [col for col in demographics.columns if demo_keyword.lower() in col[1].lower()]
    
    if not demo_cols:
        print(f"No demographic columns found for keyword: {demo_keyword}")
        return

    # Reconstruct single categorical column from one-hot encoding
    demo_groups = demographics[demo_cols].idxmax(axis=1).apply(lambda x: x[2])

    print(f"\n=== Analyzing survey responses by {demo_keyword.title()} ===")
    print("Groups detected:")
    print(demo_groups.value_counts())

    # Loop through survey questions
    for question in df.columns.get_level_values(1).unique():
        if question in likert_questions:
            data = df.loc[:, df.columns.get_level_values(1) == question].squeeze()
            grouped = data.groupby(demo_groups).mean()
            print(f"\n--- {question} by {demo_keyword.title()} (mean Likert score) ---")
            print(grouped)

            grouped.plot(kind="bar")
            plt.title(f"{question} (Mean Likert Score by {demo_keyword.title()})")
            plt.ylabel("Mean score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            data = df.loc[:, df.columns.get_level_values(1) == question]
            counts = data.groupby(demo_groups).sum()
            percentages = (counts.T / counts.T.sum() * 100).T.round(1)
            
            print(f"\n--- {question} by {demo_keyword.title()} (percentage selecting each option) ---")
            print(percentages)

            percentages.plot(kind="bar", stacked=True, figsize=(8,5))
            plt.title(f"{question} (Responses by {demo_keyword.title()})")
            plt.ylabel("% of respondents")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

# === Example usage ===
# Analyze by age groups
analyze_by_demographic("Q32")

# You can repeat for gender, education, country, etc.
# analyze_by_demographic("Q31")
# analyze_by_demographic("Q35")
# analyze_by_demographic("Q33")