import pandas as pd
import os


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Safely loads a CSV file into a Pandas DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found")

    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV '{filepath}': {e}")


def load_txt_data(filepath: str) -> str:
    """Safely loads text data from a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load text file '{filepath}': {e}")

def generate_structured_report(
        client,
        model_name,
        history,
        system_prompt,
        response_format
):
    return client.responses.parse(
        model=model_name,
        input=history + [{"role": "system","content": system_prompt}],
        text_format=response_format,
    )
