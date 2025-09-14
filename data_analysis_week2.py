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
        
        
# 4. Pattern analysis across all binary questions
print("\n=== Pattern Analysis (Frequent Combinations of Responses) ===")


# Keep only binary columns (exclude Likert scale questions)
binary_df = df.drop(columns=df.columns[df.columns.get_level_values(1).isin(likert_questions)])


# Convert all to boolean (True/False)
binary_df = binary_df.astype(bool)


# Apply Apriori algorithm with higher min_support and limited itemset length
frequent_itemsets = apriori(binary_df, min_support=0.2, max_len=3, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)


print("\nTop frequent response combinations:")
print(frequent_itemsets.head(10))


# Plot top 10 frequent itemsets
if not frequent_itemsets.empty:
    top_itemsets = frequent_itemsets.head(10)
    
    # Convert frozensets of tuples into readable strings
    labels = []
    for itemset in top_itemsets["itemsets"]:
        option_names = ["_".join(col) if isinstance(col, tuple) else str(col) for col in itemset]
        labels.append(", ".join(option_names))
    
    values = top_itemsets["support"] * 100  # convert to percentage

    plt.figure(figsize=(8,6))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()  # highest on top
    plt.xlabel("Support (%)")
    plt.title("Top 10 Frequent Response Combinations")
    plt.tight_layout()
    plt.show()


# Association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nExample association rules:")
print(rules.head(10))
