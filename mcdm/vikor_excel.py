"""Public VIKOR Excel-export interface."""

from .classical_excel import (
    VIKOR_EXCEL_EXPORT_FILENAME,
    VIKOR_EXCEL_EXPORT_REVISION,
    build_vikor_excel_workbook,
)

__all__ = [
    "VIKOR_EXCEL_EXPORT_FILENAME",
    "VIKOR_EXCEL_EXPORT_REVISION",
    "build_vikor_excel_workbook",
]
