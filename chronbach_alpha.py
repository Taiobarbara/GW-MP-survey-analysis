import pandas as pd

def cronbach_alpha(df):
    """
    Calculate Cronbach's alpha for a dataframe of items (questions).
    Rows = respondents, Columns = items (scored 0, 1, 2, etc.)
    """
    df = df.dropna(axis=0)  # drop incomplete rows
    k = df.shape[1]  # number of items
    variances = df.var(axis=0, ddof=1)  # variance of each item
    total_var = df.sum(axis=1).var(ddof=1)  # variance of total score
    
    alpha = (k / (k - 1)) * (1 - variances.sum() / total_var)
    return alpha

# Example usage with your dataset
data = pd.read_csv("/Users/bazam/Library/CloudStorage/OneDrive-Personal/Documentos/academia/#PhD PLASTIC UNDERGROUND/7.1_excel/survey/data/database_knowledge_questions.csv")  # replace with your filename

# Select all knowledge question columns (exclude respondent_id)
knowledge_items = data.drop(columns=["respondent_id"])

alpha = cronbach_alpha(knowledge_items)
print(f"Cronbach's alpha: {alpha:.3f}")
