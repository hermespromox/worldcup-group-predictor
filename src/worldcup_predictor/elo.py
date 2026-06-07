from __future__ import annotations
import math
import pandas as pd

IMPORTANCE_MULTIPLIERS = {
    "FIFA World Cup": 2.8,
    "FIFA World Cup qualification": 1.7,
    "UEFA Euro": 2.2,
    "Copa América": 2.2,
    "African Cup of Nations": 2.0,
    "AFC Asian Cup": 2.0,
    "CONCACAF Gold Cup": 1.8,
    "OFC Nations Cup": 1.6,
    "UEFA Nations League": 1.3,
    "Friendly": 0.7,
}


def match_importance_k(tournament: str, base_k: float = 24.0) -> float:
    return base_k * IMPORTANCE_MULTIPLIERS.get(str(tournament), 1.0)


def _mov_multiplier(goal_diff: int, elo_diff: float) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    # FiveThirtyEight-style dampened MOV multiplier.
    return min(2.2, math.log(gd + 1.0) * (2.2 / ((abs(elo_diff) * 0.001) + 2.2)))


def _actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for < goals_against:
        return 0.0
    return 0.5


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    results: pd.DataFrame,
    teams: list[str],
    cutoff: str | pd.Timestamp,
    base: float = 1500.0,
    k: float = 24.0,
    home_advantage: float = 65.0,
    decay_half_life_days: float | None = 3650.0,
    variant: str = "advanced",
) -> dict[str, float]:
    ratings: dict[str, float] = {}
    cutoff = pd.Timestamp(cutoff)
    work = results.copy()
    work["date"] = pd.to_datetime(work["date"])
    df = work[work["date"] < cutoff].sort_values("date")
    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        ratings.setdefault(h, base); ratings.setdefault(a, base)
        rh, ra = ratings[h], ratings[a]
        neutral = bool(m.get("neutral", False))
        if variant == "simple":
            eh = expected_score(rh, ra)
            sh = _actual_score(int(m["home_score"]), int(m["away_score"]))
            gd = abs(int(m["home_score"]) - int(m["away_score"]))
            mov = 1.0 if gd <= 1 else min(1.75, 1.0 + 0.25 * (gd - 1))
            delta = k * mov * (sh - eh)
            ratings[h] = rh + delta
            ratings[a] = ra - delta
            continue
        h_adv = 0.0 if neutral else home_advantage
        eh = expected_score(rh + h_adv, ra)
        sh = _actual_score(int(m["home_score"]), int(m["away_score"]))
        gd = int(m["home_score"]) - int(m["away_score"])
        mov = _mov_multiplier(gd, (rh + h_adv) - ra)
        date_weight = 1.0
        if decay_half_life_days:
            age_days = max((cutoff - pd.Timestamp(m["date"])).days, 0)
            date_weight = 0.5 ** (age_days / decay_half_life_days)
        kk = match_importance_k(str(m.get("tournament", "")), k) * mov * date_weight
        delta = kk * (sh - eh)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return {t: round(ratings.get(t, base), 2) for t in teams}


def opponent_adjusted_form(results: pd.DataFrame, teams: list[str], cutoff: str | pd.Timestamp, ratings: dict[str, float], n: int = 10) -> dict[str, float]:
    cutoff = pd.Timestamp(cutoff)
    work = results.copy()
    work["date"] = pd.to_datetime(work["date"])
    out: dict[str, float] = {}
    for team in teams:
        df = work[(work["date"] < cutoff) & ((work["home_team"] == team) | (work["away_team"] == team))].sort_values("date", ascending=False).head(n)
        vals = []
        for _, m in df.iterrows():
            is_home = m["home_team"] == team
            opp = m["away_team"] if is_home else m["home_team"]
            gf = int(m["home_score"] if is_home else m["away_score"])
            ga = int(m["away_score"] if is_home else m["home_score"])
            actual = _actual_score(gf, ga)
            neutral = bool(m.get("neutral", False))
            adv = 0.0 if neutral else (65.0 if is_home else -65.0)
            exp = expected_score(ratings.get(team, 1500.0) + adv, ratings.get(opp, 1500.0))
            gd_bonus = max(min((gf - ga) / 3.0, 1.0), -1.0) * 0.15
            vals.append((actual - exp) + gd_bonus)
        out[team] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out
