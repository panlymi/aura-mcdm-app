import pandas as pd

from mcdm.validation import MCDMValidationError, validate_fuzzy_number

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
    "very good": (0.8, 1.0, 1.0),
    "vl": (0.0, 0.0, 0.2),
    "very low": (0.0, 0.0, 0.2),
    "l": (0.0, 0.2, 0.4),
    "low": (0.0, 0.2, 0.4),
    "m": (0.3, 0.5, 0.7),
    "medium": (0.3, 0.5, 0.7),
    "h": (0.6, 0.8, 1.0),
    "high": (0.6, 0.8, 1.0),
    "vh": (0.8, 1.0, 1.0),
    "very high": (0.8, 1.0, 1.0)
}

def parse_tfn_string(val):
    """Parses 'a, b, c' or '(a, b, c)' into a tuple (a, b, c)."""
    try:
        # Strip common brackets and parenthesis that a user might intuitively type
        clean_val = str(val).strip(' ()[]{}')
        parts = [float(x.strip()) for x in clean_val.split(',')]
        return validate_fuzzy_number(tuple(parts))
    except (TypeError, ValueError, MCDMValidationError):
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
                    raise MCDMValidationError(
                        f"Invalid linguistic term {val!r} at row {idx!r}, column {col!r}. "
                        "Valid terms include Poor, Fair, Good, and Very Good."
                    )
            else:
                # Comma-Separated
                tfn = parse_tfn_string(val)
                if tfn is not None:
                    parsed_df.at[idx, col] = tfn
                else:
                    raise MCDMValidationError(
                        f"Invalid fuzzy number {val!r} at row {idx!r}, column {col!r}. "
                        "Expected ordered values such as '1, 2, 3' or '1, 2, 3, 4'."
                    )
                    
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
                raise MCDMValidationError(
                    f"Invalid linguistic weight {val!r} for criterion {col!r}."
                )
        else:
            tfn = parse_tfn_string(val)
            if tfn is not None:
                parsed_weights[col] = tfn
            else:
                raise MCDMValidationError(
                    f"Invalid fuzzy-number weight {val!r} for criterion {col!r}."
                )
    return parsed_weights
