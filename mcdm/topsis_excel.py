"""Public TOPSIS Excel-export interface."""

from .classical_excel import (
    TOPSIS_EXCEL_EXPORT_FILENAME,
    TOPSIS_EXCEL_EXPORT_REVISION,
    build_topsis_excel_workbook,
)

__all__ = [
    "TOPSIS_EXCEL_EXPORT_FILENAME",
    "TOPSIS_EXCEL_EXPORT_REVISION",
    "build_topsis_excel_workbook",
]
