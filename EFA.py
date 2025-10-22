# === awareness_efa.py ===
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# === 1. Load your dataset ===
file_path = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/database_awareness_questions.csv"  # replace with your actual path
df = pd.read_csv(file_path)

# Drop respondent_id if present
if 'respondent_id' in df.columns:
    df = df.drop(columns=['respondent_id'])

# === 2. Adequacy checks ===
# KMO (sampling adequacy)
kmo_all, kmo_model = calculate_kmo(df)
print(f"Kaiser-Meyer-Olkin (KMO) overall measure: {kmo_model:.3f}")
if kmo_model < 0.6:
    print("⚠️  KMO is below 0.6 — sample may not be adequate for factor analysis.\n")

# Bartlett’s test of sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(f"Bartlett’s test chi-square: {chi_square_value:.2f}, p-value: {p_value:.4f}")
if p_value >= 0.05:
    print("⚠️  Data may not be suitable for factor analysis (non-significant test).")

# === 3. Check eigenvalues to decide number of factors ===
fa = FactorAnalyzer(rotation=None)
fa.fit(df)

eigen_values, vectors = fa.get_eigenvalues()

# Scree plot
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(eigen_values)+1), eigen_values, "o-", linewidth=2)
plt.title("Scree Plot (Eigenvalues by Factor Number)")
plt.xlabel("Factor Number")
plt.ylabel("Eigenvalue")
plt.axhline(1, color='red', linestyle='--')
plt.tight_layout()
plt.show()

# === 4. Extract 3 factors (adjust if scree plot suggests otherwise) ===
n_factors = 3
fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
fa.fit(df)

# === 5. Factor loadings and variance explained ===
loadings = pd.DataFrame(fa.loadings_, index=df.columns, columns=[f"Factor{i+1}" for i in range(n_factors)])
print("\n=== Factor Loadings ===")
print(loadings.round(3))

# Variance explained
variance = pd.DataFrame({
    "Factor": [f"Factor{i+1}" for i in range(n_factors)],
    "Variance Explained (%)": fa.get_factor_variance()[1] * 100
})
print("\n=== Variance Explained by Each Factor ===")
print(variance.round(2))

# === 6. Save results ===
loadings.to_csv("EFA_factor_loadings.csv", index=True)
variance.to_csv("EFA_variance_explained.csv", index=False)
print("\n✅ Results saved to 'EFA_factor_loadings.csv' and 'EFA_variance_explained.csv'")
