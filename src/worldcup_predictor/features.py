from __future__ import annotations
import pandas as pd

def _team_matches(results: pd.DataFrame, team: str, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    df = results[(results["date"] < cutoff) & ((results["home_team"] == team) | (results["away_team"] == team))].copy()
    df = df.sort_values("date", ascending=False)
    home = df["home_team"] == team
    df["goals_for"] = df["home_score"].where(home, df["away_score"])
    df["goals_against"] = df["away_score"].where(home, df["home_score"])
    df["result"] = "D"
    df.loc[df["goals_for"] > df["goals_against"], "result"] = "W"
    df.loc[df["goals_for"] < df["goals_against"], "result"] = "L"
    return df

def compute_team_features(results: pd.DataFrame, teams: list[str], cutoff: str | pd.Timestamp, cfg: dict) -> pd.DataFrame:
    rows=[]; n=int(cfg.get("N",20)); min_matches=int(cfg.get("min_matches",10)); neutral_threshold=int(cfg.get("neutral_blend_threshold",5))
    for team in teams:
        allm = _team_matches(results, team, cutoff)
        m = allm.head(n).copy()
        used = len(m)
        sparse = used < min_matches
        if used == 0:
            rows.append({"team": team, "matches_used":0, "neutral_matches_used":0, "sparse_data_flag": True,
                         "win_rate":0.33, "draw_rate":0.34, "loss_rate":0.33, "avg_goals_scored":1.2,
                         "avg_goals_conceded":1.2, "avg_goal_diff":0.0, "form_score":0.0,
                         "neutral_avg_goals_scored":1.2, "neutral_avg_goals_conceded":1.2})
            continue
        wins=(m.result=="W").mean(); draws=(m.result=="D").mean(); losses=(m.result=="L").mean()
        gf=float(m.goals_for.mean()); ga=float(m.goals_against.mean())
        form = m.head(int(cfg.get("form_window",5))).reset_index(drop=True)
        decay=float(cfg.get("recency_decay",0.9)); score=den=0.0
        for i,r in form.iterrows():
            w=decay**i; score += w * ({"W":3,"D":1,"L":0}[r.result]); den += 3*w
        form_score = score/den if den else 0.0
        neutral=m[m.get("neutral",False)==True]
        ngf=float(neutral.goals_for.mean()) if len(neutral) else gf
        nga=float(neutral.goals_against.mean()) if len(neutral) else ga
        if len(neutral) < neutral_threshold:
            alpha=len(neutral)/neutral_threshold
            ngf=alpha*ngf+(1-alpha)*gf; nga=alpha*nga+(1-alpha)*ga
        rows.append({"team":team,"matches_used":used,"neutral_matches_used":len(neutral),"sparse_data_flag":sparse,
                     "win_rate":wins,"draw_rate":draws,"loss_rate":losses,"avg_goals_scored":gf,"avg_goals_conceded":ga,
                     "avg_goal_diff":gf-ga,"form_score":form_score,"neutral_avg_goals_scored":ngf,"neutral_avg_goals_conceded":nga})
    out=pd.DataFrame(rows)
    # shrink sparse teams toward tournament mean
    for col in ["avg_goals_scored","avg_goals_conceded","neutral_avg_goals_scored","neutral_avg_goals_conceded"]:
        mean=float(out[col].mean()) if len(out) else 1.2
        for idx,row in out.iterrows():
            if row["matches_used"] < min_matches:
                weight=row["matches_used"]/min_matches if min_matches else 1
                out.loc[idx,col]=weight*row[col]+(1-weight)*mean
    return out
