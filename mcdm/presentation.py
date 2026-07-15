"""Method-specific result metadata used by UI and exports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultPresentation:
    score_column: str
    score_ascending: bool
    format_columns: tuple[str, ...]
    unit: str


RESULT_PRESENTATION: dict[str, ResultPresentation] = {
    "AURA": ResultPresentation("Utility Score", True, ("Utility Score", "D+ (PIS)", "D- (NIS)", "D_avg (AS)"), "Utility"),
    "ARAS": ResultPresentation("K (Utility Degree)", False, ("S (Optimality)", "K (Utility Degree)"), "Degree"),
    "FUZZY ARAS": ResultPresentation("K_i (Utility Degree)", False, ("S_i (Crisp)", "K_i (Utility Degree)"), "Degree"),
    "SYAI": ResultPresentation("Closeness Score (D_i)", False, ("D+ (Dist to Ideal)", "D- (Dist to Anti-Ideal)", "Closeness Score (D_i)"), "Score"),
    "ARIE": ResultPresentation("Relative Closeness (RC_i)", False, ("Sim_best", "Sim_worst", "Relative Closeness (RC_i)"), "Closeness"),
    "MOORA": ResultPresentation("y_i (Assessment Value)", False, ("y_i (Assessment Value)",), "Assessment"),
    "TOPSIS": ResultPresentation("Relative Closeness (C_i)", False, ("D+ (Ideal)", "D- (Anti-Ideal)", "Relative Closeness (C_i)"), "Closeness"),
    "SAW": ResultPresentation("V_i (SAW Score)", False, ("V_i (SAW Score)",), "Score"),
    "VIKOR": ResultPresentation("Q_i (VIKOR Index)", True, ("S_i (Utility)", "R_i (Regret)", "Q_i (VIKOR Index)"), "Index"),
}
