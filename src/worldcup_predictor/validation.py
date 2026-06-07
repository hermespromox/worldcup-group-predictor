from __future__ import annotations
from copy import deepcopy
import math
import pandas as pd
from .features import compute_team_features
from .simulate import simulate_group
from .historical import HISTORICAL_GROUPS, CUTOFFS, WINDOWS


def actual_group_matches(results: pd.DataFrame, teams: list[str], start: str, end: str) -> pd.DataFrame:
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    teamset = set(teams)
    df = results[(results["tournament"] == "FIFA World Cup") & (results["date"] >= start) & (results["date"] < end)].copy()
    return df[df["home_team"].isin(teamset) & df["away_team"].isin(teamset)].copy()


def actual_group_table(teams: list[str], matches: pd.DataFrame) -> list[dict]:
    table = {t: {"team": t, "actual_pts": 0, "actual_gd": 0, "actual_gf": 0} for t in teams}
    for _, m in matches.iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, aw = int(m["home_score"]), int(m["away_score"])
        table[h]["actual_gf"] += hs; table[a]["actual_gf"] += aw
        table[h]["actual_gd"] += hs - aw; table[a]["actual_gd"] += aw - hs
        if hs > aw: table[h]["actual_pts"] += 3
        elif hs < aw: table[a]["actual_pts"] += 3
        else:
            table[h]["actual_pts"] += 1; table[a]["actual_pts"] += 1
    rows = sorted(table.values(), key=lambda r: (r["actual_pts"], r["actual_gd"], r["actual_gf"]), reverse=True)
    for i, r in enumerate(rows, 1): r["actual_rank"] = i
    return rows


def _corr(xs, ys):
    if len(xs) < 2: return 0.0
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den if den else 0.0


def validation_metrics(predicted: list[dict], actual: list[dict]) -> dict:
    p_by={r["team"]: r for r in predicted}; a_by={r["team"]: r for r in actual}
    p_top2={r["team"] for r in predicted if r["predicted_rank"] <= 2}
    a_top2={r["team"] for r in actual if r["actual_rank"] <= 2}
    p_winner=next(r["team"] for r in predicted if r["predicted_rank"] == 1)
    a_winner=next(r["team"] for r in actual if r["actual_rank"] == 1)
    teams=list(a_by)
    return {
        "qualification_exact": p_top2 == a_top2,
        "top2_overlap": len(p_top2 & a_top2),
        "winner_correct": p_winner == a_winner,
        "rank_correct": sum(1 for t in teams if p_by[t]["predicted_rank"] == a_by[t]["actual_rank"]),
        "points_corr": _corr([p_by[t]["predicted_pts"] for t in teams], [a_by[t]["actual_pts"] for t in teams]),
    }


def validate_year(results: pd.DataFrame, year: int, cfg: dict) -> dict:
    groups=HISTORICAL_GROUPS[year]; cutoff=CUTOFFS[year]; start,end=WINDOWS[year]
    teams=[t for ts in groups.values() for t in ts]
    features=compute_team_features(results, teams, cutoff, cfg)
    group_reports=[]
    for g,ts in groups.items():
        pred=simulate_group(g, ts, features, cfg)["standings"]
        actual=actual_group_table(ts, actual_group_matches(results, ts, start, end))
        group_reports.append({"group":g,"predicted":pred,"actual":actual,"metrics":validation_metrics(pred, actual)})
    return summarize_year(year, group_reports)


def summarize_year(year: int, group_reports: list[dict]) -> dict:
    n=len(group_reports)
    return {
        "year": year,
        "qualification_accuracy_groups": sum(1 for g in group_reports if g["metrics"]["qualification_exact"]),
        "top2_overlap_avg": round(sum(g["metrics"]["top2_overlap"] for g in group_reports)/n, 3),
        "winners_correct": sum(1 for g in group_reports if g["metrics"]["winner_correct"]),
        "rank_accuracy": round(sum(g["metrics"]["rank_correct"] for g in group_reports)/(n*4), 3),
        "points_correlation": round(sum(g["metrics"]["points_corr"] for g in group_reports)/n, 3),
        "groups": group_reports,
    }


def train_config(results: pd.DataFrame, base_cfg: dict, years=(2014,2018,2022)) -> dict:
    best=None
    candidates = [
        {"elo_variant":"simple", "model_type":"elo_only", "N":10, "scaling_factor":1.0, "xg_floor":0.25, "blend_elo_share":0.5, "elo_goal_slope":800, "base_xg":1.25, "form_elo_weight":0.0},
        {"elo_variant":"advanced", "model_type":"elo_only", "N":10, "scaling_factor":1.0, "xg_floor":0.25, "blend_elo_share":0.5, "elo_goal_slope":800, "base_xg":1.25, "form_elo_weight":0.0},
        {"elo_variant":"advanced", "model_type":"blend", "N":10, "scaling_factor":1.0, "xg_floor":0.25, "blend_elo_share":0.8, "elo_goal_slope":650, "base_xg":1.25, "form_elo_weight":0.0},
        {"elo_variant":"advanced", "model_type":"blend", "N":10, "scaling_factor":1.0, "xg_floor":0.25, "blend_elo_share":0.8, "elo_goal_slope":650, "base_xg":1.25, "form_elo_weight":150.0},
    ]
    for params in candidates:
        cfg=deepcopy(base_cfg)
        cfg.update(params)
        reports=[validate_year(results, y, cfg) for y in years]
        score=sum(r["top2_overlap_avg"] for r in reports) + 0.15*sum(r["winners_correct"] for r in reports) + 0.5*sum(r["rank_accuracy"] for r in reports)
        cand={"score":round(score,4),"config":cfg,"reports":reports}
        if best is None or cand["score"] > best["score"]: best=cand
    return best
