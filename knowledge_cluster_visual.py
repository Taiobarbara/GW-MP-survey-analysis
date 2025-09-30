import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_knowledge(data_csv, n_clusters=4, output_csv="clusters.csv", heatmap_file="clusters_boxplot.png"):
    """
    Cluster respondents based on knowledge score and demographic one-hot data.
    Also generate summary plots.

    data_csv: /Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/knowledge_database_clean.csv
    n_clusters: number of clusters (default 4)
    output_csv: path to save clustered data
    """
    # Load dataset (skip first row of section headers, use second row for columns)
    df = pd.read_csv(data_csv, header=1)

    # Separate respondent_id
    respondent_ids = df["respondent_id"]
    df = df.drop(columns=["respondent_id"])

    # Ensure numeric types are correct
    df["Knowledge_score"] = df["Knowledge_score"].astype(float)

    # Find categorical (binary) vs numeric columns
    categorical_cols = [i for i, col in enumerate(df.columns) if set(df[col].unique()) <= {0, 1}]
    numeric_cols = [i for i, col in enumerate(df.columns) if i not in categorical_cols]

    # K-Prototypes clustering
    kproto = KPrototypes(n_clusters=n_clusters, random_state=42, init='Huang', n_init=10)
    clusters = kproto.fit_predict(df, categorical=categorical_cols)

    # Add cluster labels back to dataframe
    df_out = df.copy()
    df_out["Cluster"] = clusters
    df_out.insert(0, "respondent_id", respondent_ids)

    # Save results
    df_out.to_csv(output_csv, index=False)
    print(f"Clustered data saved to {output_csv}")

    # Cluster summaries
    cluster_summary = df_out.groupby("Cluster")["Knowledge_score"].agg(["mean", "std", "count"])
    print("\n=== Cluster Summaries ===")
    print(cluster_summary)

    # --- Visualization 1: Boxplot of Knowledge Score per Cluster ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Cluster", y="Knowledge_score", data=df_out, palette="Set2")
    sns.stripplot(x="Cluster", y="Knowledge_score", data=df_out, color="black", size=3, alpha=0.5)
    plt.title("Knowledge Score Distribution per Cluster")
    plt.savefig("knowledge_score_clusters.png", dpi=300)
    plt.close()
    print("Boxplot saved as knowledge_score_clusters.png")

    # --- Visualization 2: Heatmap of demographics per cluster ---
    demo_cols = [col for col in df_out.columns if col.startswith("gender_") or col.startswith("age_") or col.startswith("educational_")]
    if demo_cols:
        demo_summary = df_out.groupby("Cluster")[demo_cols].mean()
        plt.figure(figsize=(10, 6))
        sns.heatmap(demo_summary.T, annot=True, cmap="YlGnBu", cbar_kws={"label": "Proportion"})
        plt.title("Demographic Distribution per Cluster")
        plt.tight_layout()
        plt.savefig("demographics_clusters.png", dpi=300)
        plt.close()
        print("Heatmap saved as demographics_clusters.png")

    return df_out, cluster_summary

df_clusters, summary = cluster_knowledge("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/knowledge_database_clean.csv", n_clusters=4)

print(summary)
