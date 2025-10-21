import pandas as pd
import numpy as np
from itertools import combinations
from factor_analyzer.factor_analyzer import calculate_kmo
from pingouin import cronbach_alpha

# === LOAD DATA ===
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/database_awareness_questions.csv")

# === Define awareness groups ===
groups = {
    "Water_contamination": ["Q8", "Q9", "Q10"],
    "MPs_knowledge": ["Q19", "Q21", "Q29"],
    "MPs_env_implications": ["Q14", "Q24"]
}

# === Helper functions ===
def mcdonald_omega(df_subset):
    """
    Compute McDonald's Omega (ωt) using a correlation-based approximation.
    """
    corr = df_subset.corr()
    eigvals, eigvecs = np.linalg.eig(corr)
    # First eigenvalue = general factor
    omega_total = np.real(eigvals[0] / np.sum(eigvals))
    return omega_total

def item_total_corr(df_subset):
    """
    Compute item-total correlations for each item.
    """
    corrs = {}
    for col in df_subset.columns:
        total = df_subset.drop(columns=[col]).sum(axis=1)
        corrs[col] = df_subset[col].corr(total)
    return corrs

# === Run analysis ===
results = []
for group_name, items in groups.items():
    subset = df[items].dropna()

    # Cronbach's Alpha
    alpha_val, _ = cronbach_alpha(subset)

    # McDonald's Omega
    omega_val = mcdonald_omega(subset)

    # Item-total correlations
    item_corrs = item_total_corr(subset)

    results.append({
        "Group": group_name,
        "N_Items": len(items),
        "Cronbach_Alpha": round(alpha_val, 3),
        "McDonald_Omega": round(omega_val, 3),
        "Mean_Item-Total_Corr": round(np.mean(list(item_corrs.values())), 3)
    })

# === Summary table ===
reliability_df = pd.DataFrame(results)
print("\n=== Internal Consistency Results ===\n")
print(reliability_df)

# === Optional: Detailed item-total correlations ===
print("\n=== Item-Total Correlations by Group ===\n")
for group_name, items in groups.items():
    subset = df[items].dropna()
    item_corrs = item_total_corr(subset)
    print(f"\n{group_name}:")
    for item, corr in item_corrs.items():
        print(f"  {item}: {corr:.3f}")
