from __future__ import annotations
from itertools import combinations
from .model import expected_goals, predict_match

def simulate_group(group: str, teams: list[str], features, cfg: dict) -> dict:
    by_team={r["team"]:r for r in features.to_dict("records")}
    league_avg=float(features["avg_goals_scored"].mean())
    table={t:{"team":t,"predicted_pts":0.0,"predicted_gd":0.0,"predicted_gf":0.0,"matches_used":int(by_team[t]["matches_used"]),"neutral_matches_used":int(by_team[t]["neutral_matches_used"]),"sparse_data_flag":bool(by_team[t]["sparse_data_flag"])} for t in teams}
    matches=[]
    for a,b in combinations(teams,2):
        xga,xgb=expected_goals(by_team[a],by_team[b],league_avg,cfg)
        pred=predict_match(a,b,xga,xgb,max_goals=int(cfg.get("max_goals_simulated",10)))
        d=pred.to_dict(); matches.append(d)
        table[a]["predicted_pts"] += pred.expected_pts_a; table[b]["predicted_pts"] += pred.expected_pts_b
        table[a]["predicted_gd"] += pred.expected_gd_a; table[b]["predicted_gd"] += pred.expected_gd_b
        table[a]["predicted_gf"] += pred.expected_gf_a; table[b]["predicted_gf"] += pred.expected_gf_b
    standings=sorted(table.values(), key=lambda r:(r["predicted_pts"],r["predicted_gd"],r["predicted_gf"]), reverse=True)
    for i,r in enumerate(standings,1):
        r["predicted_rank"]=i
        for k in ["predicted_pts","predicted_gd","predicted_gf"]: r[k]=round(r[k],3)
    return {"group":group,"standings":standings,"matches":matches}

def apply_2026_qualification_statuses(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    third=[]
    for g, rows in groups.items():
        for r in rows:
            if r["predicted_rank"] <= 2: r["status"]="automatic_qualifier"
            elif r["predicted_rank"] == 3:
                r["status"]="third_place_pending"; third.append((g,r))
            else: r["status"]="eliminated"
    third.sort(key=lambda gr:(gr[1]["predicted_pts"],gr[1]["predicted_gd"],gr[1]["predicted_gf"]), reverse=True)
    qualifiers={(g,r["team"]) for g,r in third[:8]}
    for g,r in third:
        r["status"] = "third_place_qualifier" if (g,r["team"]) in qualifiers else "eliminated"
    return groups

def simulate_2026(groups: dict[str,list[str]], features, cfg: dict) -> dict:
    out={}
    for g,teams in groups.items():
        out[g]=simulate_group(g,teams,features,cfg)["standings"]
    apply_2026_qualification_statuses(out)
    return out
