from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import pandas as pd

@dataclass(frozen=True)
class MatchPrediction:
    team_a: str
    team_b: str
    xg_a: float
    xg_b: float
    p_a_win: float
    p_draw: float
    p_b_win: float
    expected_pts_a: float
    expected_pts_b: float
    expected_gd_a: float
    expected_gd_b: float
    expected_gf_a: float
    expected_gf_b: float
    most_likely_score: str

    def to_dict(self) -> dict:
        return asdict(self)

def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_score_matrix(xg_a: float, xg_b: float, max_goals: int = 10) -> pd.DataFrame:
    rows = []
    for a in range(max_goals + 1):
        rows.append([_poisson_pmf(a, xg_a) * _poisson_pmf(b, xg_b) for b in range(max_goals + 1)])
    return pd.DataFrame(rows, index=range(max_goals + 1), columns=range(max_goals + 1))

def predict_match(team_a: str, team_b: str, xg_a: float, xg_b: float, max_goals: int = 10) -> MatchPrediction:
    matrix = poisson_score_matrix(xg_a, xg_b, max_goals=max_goals)
    total = matrix.values.sum()
    if total <= 0:
        raise ValueError("score matrix has zero probability mass")
    matrix = matrix / total
    p_a_win = p_draw = p_b_win = 0.0
    egd_a = egf_a = egf_b = 0.0
    best = (0, 0, -1.0)
    for a in matrix.index:
        for b in matrix.columns:
            p = float(matrix.loc[a, b])
            egf_a += a * p
            egf_b += b * p
            egd_a += (a - b) * p
            if a > b: p_a_win += p
            elif a == b: p_draw += p
            else: p_b_win += p
            if p > best[2]: best = (a, b, p)
    return MatchPrediction(
        team_a=team_a, team_b=team_b, xg_a=round(xg_a, 4), xg_b=round(xg_b, 4),
        p_a_win=round(p_a_win, 6), p_draw=round(p_draw, 6), p_b_win=round(p_b_win, 6),
        expected_pts_a=round(3*p_a_win + p_draw, 4), expected_pts_b=round(3*p_b_win + p_draw, 4),
        expected_gd_a=round(egd_a, 4), expected_gd_b=round(-egd_a, 4),
        expected_gf_a=round(egf_a, 4), expected_gf_b=round(egf_b, 4),
        most_likely_score=f"{best[0]}-{best[1]}",
    )

def expected_goals(row_a: dict, row_b: dict, league_avg_scored: float, cfg: dict) -> tuple[float, float]:
    floor = float(cfg.get("xg_floor", 0.25)); ceiling = float(cfg.get("xg_ceiling", 5.0)); scale = float(cfg.get("scaling_factor", 1.0))
    avg = max(float(league_avg_scored), 0.01)
    attack_a = max(float(row_a["neutral_avg_goals_scored"]) / avg, floor)
    attack_b = max(float(row_b["neutral_avg_goals_scored"]) / avg, floor)
    defensive_weakness_a = max(float(row_a["neutral_avg_goals_conceded"]) / avg, floor)
    defensive_weakness_b = max(float(row_b["neutral_avg_goals_conceded"]) / avg, floor)
    xg_a = attack_a * defensive_weakness_b * avg * scale
    xg_b = attack_b * defensive_weakness_a * avg * scale
    elo_weight = float(cfg.get("elo_weight", 0.0))
    if elo_weight and "elo" in row_a and "elo" in row_b:
        elo_diff = float(row_a["elo"]) - float(row_b["elo"])
        # 400 Elo ~= 10x odds; use a softer exponent for goals so priors help but don't dominate.
        elo_mult_a = 10.0 ** ((elo_diff / 800.0) * elo_weight)
        elo_mult_b = 10.0 ** ((-elo_diff / 800.0) * elo_weight)
        xg_a *= elo_mult_a
        xg_b *= elo_mult_b
    return min(xg_a, ceiling), min(xg_b, ceiling)
