# === awareness_cfa.py ===

import pandas as pd
from semopy import Model, semplot, calc_stats

# Load your dataset (replace with your actual file path)
df = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/database_awareness_questions.csv")

# Define the CFA model (Lavaan-like syntax)
model_desc = """
F1_env_implications =~ Q24 + Q29
F2_water_contamination =~ Q9 + Q10
F3_mps_knowledge =~ Q14 + Q19 + Q21
"""

# Create and fit the CFA model
model = Model(model_desc)
model.fit(df)

# Get fit statistics
stats = calc_stats(model)
print(stats)

# Optional: visualize the model (requires graphviz installed)
try:
    semplot(model, "awareness_cfa_model.png")
    print("Model diagram saved as 'awareness_cfa_model.png'")
except:
    print("Graphviz not installed — skipping visualization.")
