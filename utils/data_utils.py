import pandas as pd

def clean_nan(df: pd.DataFrame, drop_threshold: float = 0.3) -> pd.DataFrame:
    """
    Cleans a DataFrame by:
    1. Dropping columns with too many NaNs.
    2. Filling remaining NaNs.
    3. Downcasting to float32.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        drop_threshold (float): Max % of allowed missing values per column.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df.dropna(axis=1, thresh=int((1 - drop_threshold) * len(df)))
    df = df.fillna(0)  # or use forward/backward fill
    df = df.astype({col: 'float32' for col in df.select_dtypes(include=['float', 'int']).columns})
    return df