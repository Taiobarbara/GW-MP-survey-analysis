import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load the dataset (replace 'survey_data.csv' with your file name)
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey_transformed_3.csv"
# First 3 rows are headers: (section, question, option)
df = pd.read_csv(file_path, header=[0,1,2])

# Make respondent_id the index
if ('respondent_id', 'respondent_id', 'respondent_id') in df.columns:
    df = df.set_index(('respondent_id', 'respondent_id', 'respondent_id'))

# 2. Display first few rows to check structure
print("Preview of dataset:")
print(df.head())

# Identify Likert scale questions (they are numeric, not binary)
likert_questions = {"Q6", "Q15", "Q17", "Q26"}

# 3. Descriptive analysis for each question
print("\nDescriptive statistics for each question:")

for question in df.columns.get_level_values(1).unique():
    q_cols = df.loc[:, df.columns.get_level_values(1) == question]
    
    print(f"\n=== {question} ===")
    
    # Likert scale questions
    if question in likert_questions:
        data = q_cols.squeeze()  # Likert should be a single column
        print(data.describe()[["mean", "std", "min", "max"]])
        
        # Plot histogram (remove the # from the lines below to plot the graphs)
        #plt.figure()
        #data.plot(kind='hist', bins=5, rwidth=0.8)
        #plt.title(question)
        #plt.xlabel("Response scale")
        #plt.ylabel("Frequency")
        #plt.show()
    else:
        # Multiple choice questions (binary encoded)
        counts = q_cols.sum()
        percentages = (counts / len(df) * 100).round(1)
        summary = pd.DataFrame({"Count": counts, "Percentage": percentages})
        print(summary)

        # Plot bar chart (remove the # from the lines below to plot the graphs)
        #plt.figure()
        #summary["Count"].plot(kind='bar')
        #plt.title(question)
        #plt.ylabel("Number of respondents")
        #plt.xlabel("Options")
        #plt.xticks(rotation=45, ha='right')
        #plt.tight_layout()
        #plt.show()
        