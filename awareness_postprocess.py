# awareness_postprocess.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pathlib import Path

# ---------------- USER PARAMETERS ----------------
# Paths (adjust as needed)
factor_scores_csv = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/awareness_score_EFA.csv"   # your factor scores file (respondent_id, factor1, factor2, factor3)
knowledge_csv = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/knowledge_score_clusters.csv"              # optional: respondent_id, knowledge_score, cluster (set to None if not available)
demographics_csv = "/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/demographic.csv"               # optional: respondent_id, gender, age_group, education (set to None if not available)

output_prefix = "awareness_analysis"  # files produced: awareness_analysis_normalized.csv, ...
# -------------------------------------------------

# 1) Load factor scores
fpath = Path(factor_scores_csv)
if not fpath.exists():
    raise FileNotFoundError(f"Factor scores file not found: {fpath.resolve()}")

df = pd.read_csv(fpath)
# Expect columns: respondent_id, factor1, factor2, factor3 (names may differ -> we'll detect)
print("Loaded factor scores:", df.shape)

# Rename factor columns to consistent names if needed
# Find numeric columns other than respondent_id
cols = [c for c in df.columns if c != "respondent_id"]
if len(cols) < 1:
    raise ValueError("No factor columns detected in factor scores CSV.")
# Map to factor1..factor3 if names differ
if set(cols) != {"factor1", "factor2", "factor3"}:
    # choose first three numeric columns as factor1..3
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 3:
        mapping = {numeric_cols[i]: f"factor{i+1}" for i in range(3)}
        df = df.rename(columns=mapping)
        print("Renamed factor columns:", mapping)
    else:
        # if already named factor1..3, keep them; else try to proceed with existing names
        print("Using existing columns as factors:", cols)

factor_cols = [c for c in ["factor1", "factor2", "factor3"] if c in df.columns]

# 2) Normalize each factor to [0,1] by min-max
def minmax_series(s):
    if s.max() == s.min():
        return s*0.0  # constant column -> return zeros
    return (s - s.min()) / (s.max() - s.min())

for c in factor_cols:
    df[c + "_norm"] = minmax_series(df[c].astype(float))

norm_cols = [c + "_norm" for c in factor_cols]
print("\nNormalized columns:", norm_cols)

# Save normalized scores
out_norm = f"{output_prefix}_normalized_scores.csv"
df.to_csv(out_norm, index=False)
print(f"Saved normalized factor scores to {out_norm}")

# 3) Optionally merge knowledge and demographics if files provided
merged = df.copy()
if knowledge_csv:
    kp = Path(knowledge_csv)
    if kp.exists():
        kdf = pd.read_csv(kp)
        merged = merged.merge(kdf, on="respondent_id", how="left")
        print("Merged knowledge data:", kdf.shape)
    else:
        print("Knowledge file not found, skipping merge:", knowledge_csv)

if demographics_csv:
    dp = Path(demographics_csv)
    if dp.exists():
        ddf = pd.read_csv(dp)
        merged = merged.merge(ddf, on="respondent_id", how="left")
        print("Merged demographics data:", ddf.shape)
    else:
        print("Demographics file not found, skipping merge:", demographics_csv)

# Save merged dataset
out_merged = f"{output_prefix}_merged.csv"
merged.to_csv(out_merged, index=False)
print(f"Saved merged dataset: {out_merged}")

# 4) Descriptive statistics for each normalized awareness factor
desc = merged[norm_cols].agg(["mean", "median", "std", "min", "max"]).T
desc = desc.rename(columns={"std":"sd"})
print("\nDescriptive statistics (normalized awareness factors):")
print(desc.round(3))
desc.to_csv(f"{output_prefix}_descriptives.csv")
print(f"Saved descriptives to {output_prefix}_descriptives.csv")

# 5) Distribution plots: hist + KDE per factor
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set2", n_colors=len(norm_cols))
for color, col in zip(palette, norm_cols):
    sns.histplot(merged[col].dropna(), bins=10, kde=True, stat="density", label=col, color=color, alpha=0.35)
plt.legend()
plt.xlabel("Normalized score (0-1)")
plt.title("Awareness factors distributions (normalized)")
plt.tight_layout()
plt.savefig(f"{output_prefix}_distributions.png", dpi=300)
plt.show()
print(f"Saved distribution plot to {output_prefix}_distributions.png")

# 6) Correlation between Knowledge and Awareness (if knowledge exists)
corr_out = {}
if "knowledge_score" in merged.columns:
    for c in norm_cols:
        r = merged[["knowledge_score", c]].dropna().corr().iloc[0,1]
        corr_out[c] = r
    corr_df = pd.DataFrame.from_dict(corr_out, orient="index", columns=["pearson_r"]).round(3)
    print("\nCorrelation (Knowledge vs Awareness factors):")
    print(corr_df)
    corr_df.to_csv(f"{output_prefix}_knowledge_correlations.csv")
    print(f"Saved knowledge correlations to {output_prefix}_knowledge_correlations.csv")
else:
    print("\nknowledge_score column not found; skipping correlations with knowledge.")

# 7) Compare Awareness across knowledge clusters (if 'cluster' exists)
if "cluster" in merged.columns:
    for col in norm_cols:
        plt.figure(figsize=(7,4))
        sns.boxplot(x="cluster", y=col, data=merged)
        sns.stripplot(x="cluster", y=col, data=merged, color="black", size=3, alpha=0.4)
        plt.title(f"{col} by knowledge cluster")
        plt.tight_layout()
        fname = f"{output_prefix}_{col}_by_cluster.png"
        plt.savefig(fname, dpi=300)
        plt.show()
        print("Saved:", fname)
else:
    print("No 'cluster' column found; skipping cluster comparisons.")

# 8) Simple regression: each awareness factor ~ knowledge_score + demographics (if available)
# Build a formula dynamically; include knowledge_score if present, and up to 3 demographic predictors if available.
dem_vars = []
for demo_col in ["gender_", "age_", "educational_"]:
    if demo_col in merged.columns:
        dem_vars.append(demo_col)

if "knowledge_score" in merged.columns:
    for col in norm_cols:
        # create formula: col ~ knowledge_score + dem1 + dem2
        rhs = ["knowledge_score"] + dem_vars
        formula = f"{col} ~ " + " + ".join(rhs)
        # drop rows with NA in formula vars
        model_df = merged[[col] + rhs].dropna()
        if model_df.shape[0] < 10:
            print(f"Skipping regression for {col} due to small N={model_df.shape[0]}")
            continue
        model = smf.ols(formula=formula, data=model_df).fit()
        print(f"\nRegression results for {col}:")
        print(model.summary().tables[1])
        # Save summary to text
        with open(f"{output_prefix}_{col}_regression.txt", "w") as f:
            f.write(model.summary().as_text())
        print("Saved regression summary to", f"{output_prefix}_{col}_regression.txt")
else:
    print("knowledge_score not found; skipping regression analyses.")

print("\nAll done.")
