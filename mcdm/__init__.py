"""Shared infrastructure for the AURA MCDM application."""

from .criteria import (
    CriterionPreference,
    CriterionType,
    METHOD_CAPABILITIES,
    UnsupportedCriterionError,
    normalize_directions,
    validate_method_capabilities,
)
from .validation import MCDMValidationError

__all__ = [
    "CriterionPreference",
    "CriterionType",
    "METHOD_CAPABILITIES",
    "MCDMValidationError",
    "UnsupportedCriterionError",
    "normalize_directions",
    "validate_method_capabilities",
]
