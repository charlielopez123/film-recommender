import pandas as pd

def to_datetime_format(date_str: str):
    return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')