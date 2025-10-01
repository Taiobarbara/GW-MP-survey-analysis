import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def cluster_knowledge_with_silhouette(data_csv, n_clusters=4):
    """
    Cluster respondents with K-Prototypes and calculate silhouette score.
    """
    # Load dataset
    df = pd.read_csv(data_csv, header=1)

    respondent_ids = df["respondent_id"]
    df = df.drop(columns=["respondent_id"])
    df["Knowledge_score"] = df["Knowledge_score"].astype(float)

    # Identify categorical vs numeric
    categorical_cols = [i for i, col in enumerate(df.columns) if set(df[col].unique()) <= {0, 1}]
    numeric_cols = [i for i, col in enumerate(df.columns) if i not in categorical_cols]

    # Fit clustering
    kproto = KPrototypes(n_clusters=n_clusters, random_state=42, init='Huang', n_init=10)
    clusters = kproto.fit_predict(df, categorical=categorical_cols)

    # Prepare data for silhouette: convert categorical to string safely
    df_for_sil = df.copy()
    for col_idx in categorical_cols:
        col_name = df.columns[col_idx]
        df_for_sil[col_name] = df_for_sil[col_name].astype(str)

    # One-hot encode categoricals for distance calculation
    df_encoded = pd.get_dummies(df_for_sil, drop_first=False)

    # Compute silhouette score
    sil_score = silhouette_score(df_encoded, clusters, metric="euclidean")
    print(f"Silhouette Score for {n_clusters} clusters: {sil_score:.3f}")

    # Return results
    df_out = df.copy()
    df_out["Cluster"] = clusters
    df_out.insert(0, "respondent_id", respondent_ids)

    return df_out, sil_score

scores = {}
for k in range(2, 7):  # test 2 to 6 clusters
    _, sil = cluster_knowledge_with_silhouette("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/knowledge_database_clean.csv", n_clusters=k)
    scores[k] = sil

print("\nSilhouette scores by number of clusters:")
for k, s in scores.items():
    print(f"{k} clusters: {s:.3f}")


def plot_silhouette_scores(scores, output_file="silhouette_scores.png"):
    """
    Plot silhouette scores for different numbers of clusters.
    
    Parameters:
        scores (dict): dictionary where key = k (clusters), value = silhouette score
        output_file (str): filename to save the plot
    """
    ks = list(scores.keys())
    vals = list(scores.values())

    plt.figure(figsize=(8, 5))
    plt.plot(ks, vals, marker="o", linestyle="-", color="teal")
    plt.xticks(ks)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores vs Number of Clusters")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Highlight best k
    best_k = ks[vals.index(max(vals))]
    best_score = max(vals)
    plt.scatter(best_k, best_score, color="red", s=100, label=f"Best k = {best_k} ({best_score:.3f})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Silhouette score plot saved as {output_file}")

scores = {}
for k in range(2, 7):  # test k=2 to k=6
    _, sil = cluster_knowledge_with_silhouette("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/knowledge_database_clean.csv", n_clusters=k)
    scores[k] = sil

plot_silhouette_scores(scores)
