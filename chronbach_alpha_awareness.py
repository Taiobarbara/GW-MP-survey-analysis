import pandas as pd
import numpy as np

def cronbach_alpha(df):
    """
    Compute Cronbach's Alpha for a given dataframe (columns = items of one scale).
    """
    df = df.dropna(axis=0)
    item_variances = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    n_items = df.shape[1]

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_var)
    return alpha

def cronbach_alpha_by_group(csv_file, group_dict):
    """
    Compute Cronbach's Alpha for each awareness (or other) group.
    csv_file: path to dataset
    group_dict: dictionary with {'group_name': [list of question columns]}
    """

    df = pd.read_csv(csv_file)
    results = []

    for group_name, questions in group_dict.items():
        # Check that all columns exist
        valid_cols = [col for col in questions if col in df.columns]
        if len(valid_cols) < 2:
            print(f"⚠️ Skipping {group_name}: needs ≥2 questions (found {len(valid_cols)})")
            continue

        alpha = cronbach_alpha(df[valid_cols])
        results.append({"Group": group_name, "N_Items": len(valid_cols), "Cronbach_Alpha": round(alpha, 3)})

    results_df = pd.DataFrame(results)
    print("\n=== Cronbach's Alpha by Awareness Group ===\n")
    print(results_df)

    results_df.to_csv("awareness_cronbach_alpha.csv", index=False)
    print("\n✅ Saved as 'awareness_cronbach_alpha.csv'")

    return results_df

groups = {
    "Water_contamination": ["Q8", "Q9", "Q10"],
    "MPs_awareness": ["Q14", "Q19", "Q21", "Q24", "Q29"]
}

cronbach_alpha_by_group("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/database_awareness_questions.csv", groups)
