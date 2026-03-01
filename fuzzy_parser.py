import pandas as pd
import numpy as np
import streamlit as st

LINGUISTIC_MAPPING = {
    "vp": (0.0, 0.0, 0.2),
    "very poor": (0.0, 0.0, 0.2),
    "p": (0.0, 0.2, 0.4),
    "poor": (0.0, 0.2, 0.4),
    "f": (0.3, 0.5, 0.7),
    "fair": (0.3, 0.5, 0.7),
    "g": (0.6, 0.8, 1.0),
    "good": (0.6, 0.8, 1.0),
    "vg": (0.8, 1.0, 1.0),
    "very good": (0.8, 1.0, 1.0)
}

def parse_tfn_string(val):
    """Parses 'a, b, c' into a tuple (a, b, c)."""
    try:
        parts = [float(x.strip()) for x in str(val).split(',')]
        if len(parts) == 3:
            return tuple(parts)
        elif len(parts) == 1:
            return (parts[0], parts[0], parts[0])
        else:
            return None
    except:
        return None

def parse_fuzzy_matrix(df: pd.DataFrame, method: str):
    """
    Parses an entire dataframe of string TFNs or Linguistic Terms.
    """
    parsed_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        for idx in df.index:
            val = df.at[idx, col]
            if method == "Linguistic Terms":
                clean_val = str(val).strip().lower()
                if clean_val in LINGUISTIC_MAPPING:
                    parsed_df.at[idx, col] = LINGUISTIC_MAPPING[clean_val]
                else:
                    st.error(f"Invalid linguistic term '{val}' at row '{idx}', column '{col}'. Valid terms: Poor, Fair, Good, etc.")
                    return None
            else:
                # Comma-Separated
                tfn = parse_tfn_string(val)
                if tfn is not None:
                    parsed_df.at[idx, col] = tfn
                else:
                    st.error(f"Invalid TFN format '{val}' at row '{idx}', column '{col}'. Expected like '1, 2, 3'.")
                    return None
                    
    return parsed_df

def parse_fuzzy_weights(weights: dict, method: str):
    """Parses the dictionary of string weights."""
    parsed_weights = {}
    for col, val in weights.items():
        if method == "Linguistic Terms":
            clean_val = str(val).strip().lower()
            if clean_val in LINGUISTIC_MAPPING:
                parsed_weights[col] = LINGUISTIC_MAPPING[clean_val]
            else:
                st.error(f"Invalid linguistic weight '{val}' for criterion '{col}'.")
                return None
        else:
            tfn = parse_tfn_string(val)
            if tfn is not None:
                parsed_weights[col] = tfn
            else:
                st.error(f"Invalid TFN weight '{val}' for criterion '{col}'.")
                return None
    return parsed_weights
