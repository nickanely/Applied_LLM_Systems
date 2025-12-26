import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_interaction(df, expression):
    result = df.eval(expression)
    new_col = f"interaction_{len([c for c in df.columns if c.startswith('interaction_')]) + 1}"
    df[new_col] = result
    return df


def encode_categorical(df, col):
    if col not in df.columns:
        return df

    n_unique = df[col].nunique()

    if n_unique <= 10:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        for dummy_col in dummies.columns:
            df[dummy_col] = dummies[dummy_col]
        df.drop(columns=[col], inplace=True)
    else:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def correlation_analysis(df, target='target'):
    if target not in df.columns:
        return {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric_cols:
        return {}

    features = [c for c in numeric_cols if c != target]
    target_corr = df[features + [target]].corr()[target].drop(target).sort_values(ascending=False)

    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col, row, float(upper.loc[row, col]))
                 for col in upper.columns
                 for row in upper.index
                 if upper.loc[row, col] > 0.8]

    return {
        "target_correlation": {k: float(v) for k, v in target_corr.to_dict().items()},
        "high_intercorrelations": high_corr
    }

def select_top_features(df, k, target):
    """Keep target and k most correlated numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])

    if target not in numeric_df.columns:
        raise ValueError(f"Target '{target}' is not numeric")

    corr = numeric_df.corr()[target].abs().sort_values(ascending=False)

    top_k_features = [col for col in corr.index if col != target][:k]

    # cols_to_keep = top_k_features + [target] + list(df.select_dtypes(exclude=[np.number]).columns)
    # df = df[df.columns.intersection(cols_to_keep)]

    return f"Kept top {k} features: {top_k_features}"