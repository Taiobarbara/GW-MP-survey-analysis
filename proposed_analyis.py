import pandas as pd
import scipy.stats as stats
import numpy as np

# Load binary survey dataset (already one-hot encoded, Likert transformed to binary)
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey_transformed_4.csv", header=2)

def cramers_v(confusion_matrix):
    """Compute Cramér's V (effect size) for a 2x2 table."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def test_relationships_binary(df, comparisons_csv, alpha=0.05, output_csv="relationship_tests.csv"):
    """
    Run chi-square tests for binary vs binary variables from a comparisons list.

    df: DataFrame with binary survey responses (0/1).
    comparisons_csv: path to CSV file with antecedent and consequent columns.
    alpha: significance threshold.
    output_csv: where to save results.
    """

    # Load comparisons list
    comps = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/antecedents_consequents.csv").fillna("")
    results = []

    
    for _, row in comps.iterrows():
        antecedent = row["Antecedent"].strip()
        consequents = [c.strip() for c in row[1:].tolist() if isinstance(c, str) and c.strip() != ""]

        for cons in consequents:
            if antecedent not in df.columns or cons not in df.columns:
                results.append((antecedent, cons, "Chi-square", None, None, None, False, "Column missing"))
                continue

            # Crosstab
            table = pd.crosstab(df[antecedent], df[cons])

            try:
                chi2, p, dof, expected = stats.chi2_contingency(table)
                v = cramers_v(table)
                results.append((antecedent, cons, "Chi-square", chi2, p, v, p < alpha, "OK"))
            except Exception as e:
                results.append((antecedent, cons, "ERROR", None, None, None, False, str(e)))

    # Save results
    results_df = pd.DataFrame(
        results,
        columns=["Antecedent", "Consequent", "Test", "Chi2", "p-value", "CramersV", "Significant", "Note"]
    )
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return results_df

# Run comparisons + save heatmap
results = test_relationships_binary(df, "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/antecedents_consequents.csv")

# Preview results
print(results.head())