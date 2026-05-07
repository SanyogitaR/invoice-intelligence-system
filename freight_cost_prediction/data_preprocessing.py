import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path: str) -> pd.DataFrame:
    """
    Load vendor invoice data from an SQLite database.
    """
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM vendor_invoice"
        df = pd.read_sql_query(query, conn)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select features and target variable.
    """
    X = df[["Dollars"]]
    Y = df["Freight"]
    return X, Y


def split_data(X, Y, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into training and testing sets.
    """
    return train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state
    )