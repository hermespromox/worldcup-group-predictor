from __future__ import annotations
import pandas as pd


def compute_elo_ratings(results: pd.DataFrame, teams: list[str], cutoff: str | pd.Timestamp, base: float = 1500.0, k: float = 24.0) -> dict[str, float]:
    ratings: dict[str, float] = {}
    cutoff = pd.Timestamp(cutoff)
    work = results.copy()
    work["date"] = pd.to_datetime(work["date"])
    df = work[work["date"] < cutoff].sort_values("date")
    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        ratings.setdefault(h, base); ratings.setdefault(a, base)
        rh, ra = ratings[h], ratings[a]
        eh = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
        if m["home_score"] > m["away_score"]: sh = 1.0
        elif m["home_score"] < m["away_score"]: sh = 0.0
        else: sh = 0.5
        # modest goal-difference multiplier, capped so blowouts do not dominate
        gd = abs(int(m["home_score"]) - int(m["away_score"]))
        mult = 1.0 if gd <= 1 else min(1.75, 1.0 + 0.25 * (gd - 1))
        delta = k * mult * (sh - eh)
        ratings[h] = rh + delta; ratings[a] = ra - delta
    return {t: round(ratings.get(t, base), 2) for t in teams}
