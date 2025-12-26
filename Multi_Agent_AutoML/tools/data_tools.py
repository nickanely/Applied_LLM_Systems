import pandas as pd


def inspect_metadata(df):
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return info


def get_column_stats(df, col):
    if col not in df.columns:
        return f"Column '{col}' not found"

    stats = {'column': col, 'dtype': str(df[col].dtype)}

    if df[col].dtype in ['int64', 'float64']:
        stats.update({
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'null_count': df[col].isnull().sum()
        })
    else:
        stats.update({
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(10).to_dict(),
            'null_count': df[col].isnull().sum()
        })

    return stats


def impute_missing(df, col, strategy='mean'):
    df = df.copy()

    if strategy == 'mean':
        df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mode':
        df[col].fillna(df[col].mode()[0], inplace=True)
    elif strategy == 'zero':
        df[col].fillna(0, inplace=True)

    return df


def drop_column(df, col):
    df = df.copy()
    return df.drop(columns=[col])