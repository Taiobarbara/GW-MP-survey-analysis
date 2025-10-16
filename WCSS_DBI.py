import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import davies_bouldin_score

def evaluate_clusters(data_csv, k_range=[3,4,5,6]):
    """
    Compute Inertia (WCSS) and Davies-Bouldin Index for different cluster numbers.
    """
    # Load dataset (skip first row of section headers, use second row for columns)
    df = pd.read_csv(data_csv, header=1)

    respondent_ids = df["respondent_id"]
    df = df.drop(columns=["respondent_id"])
    df["Knowledge_score"] = df["Knowledge_score"].astype(float)

    # Identify categorical vs numeric
    categorical_cols = [i for i, col in enumerate(df.columns) if set(df[col].unique()) <= {0, 1}]
    numeric_cols = [i for i, col in enumerate(df.columns) if i not in categorical_cols]

    results = {}

    for k in k_range:
        print(f"\n--- Evaluating {k} clusters ---")
        kproto = KPrototypes(n_clusters=k, random_state=42, init='Huang', n_init=10)
        clusters = kproto.fit_predict(df, categorical=categorical_cols)

        # WCSS (Inertia from K-Prototypes)
        inertia = kproto.cost_

        # Prepare encoded version for DBI
        df_for_dbi = df.copy()
        for col_idx in categorical_cols:
            col_name = df.columns[col_idx]
            df_for_dbi[col_name] = df_for_dbi[col_name].astype(str)
        df_encoded = pd.get_dummies(df_for_dbi, drop_first=False)

        # DBI
        dbi = davies_bouldin_score(df_encoded, clusters)

        results[k] = {"Inertia (WCSS)": inertia, "Davies-Bouldin Index": dbi}
        print(f"Inertia (WCSS): {inertia:.2f}")
        print(f"Davies-Bouldin Index: {dbi:.3f}")

    return pd.DataFrame(results).T

results = evaluate_clusters("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/knowledge_database_clean.csv", k_range=[3,4,5,6])
print("\n=== Clustering Evaluation Results ===")
print(results)
