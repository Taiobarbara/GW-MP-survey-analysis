import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from mlxtend.frequent_patterns import apriori, association_rules

# ------------- USER PARAMETERS -------------
# this code runs for the file survey_transformed_3 with binary and numeric (likert) data 
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey_transformed_3.csv"   # adjust
min_support = 0.05              # minimum support (try 0.05 to 0.1)
max_len = 3                     # maximum itemset length (keeps memory manageable)
lift_threshold = 1.0            # association_rules min_threshold on chosen metric
top_n = 10                      # how many top itemsets/rules to show
wrap_width = 45                 # wrap labels at this many characters
output_rules_csv = "simplified_association_rules.csv"
output_itemsets_csv = "frequent_itemsets_2plus.csv"
# -------------------------------------------

# 1) Load dataset (3-level header)
df = pd.read_csv(file_path, header=[0,1,2])

# 2) Set respondent_id as index if present
if ('respondent_id', 'respondent_id', 'respondent_id') in df.columns:
    df = df.set_index(('respondent_id', 'respondent_id', 'respondent_id'))

# 3) Exclude Likert questions from pattern analysis (they are numeric)
likert_questions = {"Q6", "Q15", "Q17", "Q26"}
binary_df = df.drop(columns=df.columns[df.columns.get_level_values(1).isin(likert_questions)])

# 4) Flatten the MultiIndex column names into single strings (safe for mlxtend)
#    e.g., ('water_quality','Q1','Q1_Public_water_supply') -> 'water_quality_Q1_Q1_Public_water_supply'
binary_df.columns = ["_".join(map(str, col)).replace(" ", "_") for col in binary_df.columns]

# 5) Convert to boolean as recommended by mlxtend
binary_df = binary_df.astype(bool)

# 6) Run apriori to get ALL frequent itemsets (including singletons)
print("\nRunning apriori (this may take a bit depending on dataset size)...")
frequent_all = apriori(binary_df, min_support=min_support, max_len=max_len, use_colnames=True)
if frequent_all.empty:
    print("No frequent itemsets found with the current min_support and max_len. Try lowering min_support.")
else:
    print(f"Apriori found {len(frequent_all)} frequent itemsets (support >= {min_support}).")

    # 7) Create a version restricted to 2+ items for plotting/reporting
    frequent_2plus = frequent_all[frequent_all['itemsets'].apply(lambda s: len(s) >= 2)].copy()
    frequent_2plus = frequent_2plus.sort_values(by="support", ascending=False)

    if frequent_2plus.empty:
        print("No multi-item (2+) frequent itemsets found. Try lowering min_support or increasing max_len.")
    else:
        # Save multi-item itemsets to CSV for inspection
        # Convert itemsets to joined strings for CSV readability
        frequent_2plus['itemset_str'] = frequent_2plus['itemsets'].apply(lambda s: ", ".join(sorted(s)))
        frequent_2plus.to_csv(output_itemsets_csv, index=False)
        print(f"Saved frequent 2+ itemsets to '{output_itemsets_csv}' ({len(frequent_2plus)} rows).")

        # 8) Plot top N frequent 2+ itemsets with wrapped labels and larger left margin
        top_itemsets = frequent_2plus.head(top_n)
        labels = [textwrap.fill(", ".join(sorted(it)), width=wrap_width) for it in top_itemsets['itemsets']]
        values = top_itemsets['support'] * 100  # percent

        plt.figure(figsize=(11, 0.8 * max(6, len(labels))))  # height scales with number of labels
        plt.barh(labels, values)
        plt.gca().invert_yaxis()  # highest on top
        plt.xlabel("Support (%)")
        plt.title(f"Top {min(top_n, len(labels))} Frequent Response Combinations (2+ items)")
        plt.subplots_adjust(left=0.35)  # leave room for long labels
        plt.show()
        
        
# --- Generate and save ALL association rules for clarity ---
rules_all = association_rules(frequent_all, metric="lift", min_threshold=1.0)

if rules_all.empty:
    print("No rules generated, try lowering min_support.")
else:
    # Format antecedents and consequents as strings
    rules_all['Antecedent'] = rules_all['antecedents'].apply(lambda x: ", ".join(sorted(x)))
    rules_all['Consequent'] = rules_all['consequents'].apply(lambda x: ", ".join(sorted(x)))

    # Keep only useful columns
    rules_export = rules_all[['Antecedent','Consequent','support','confidence','lift']].copy()

    # Sort by support (highest first) and keep only top 1000
    rules_export = rules_export.sort_values(by="support", ascending=False).head(1000)

    # Save to CSV
    rules_export.to_csv("frequent_itemsets_rules.csv", index=False)
    print("Top 1000 rules by support saved to 'frequent_itemsets_rules.csv'.")
