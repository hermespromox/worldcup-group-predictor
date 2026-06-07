from __future__ import annotations
import argparse, json
from pathlib import Path
from .config import load_config
from .data import download_kaggle, kaggle_metadata, load_results
from .features import compute_team_features
from .groups import OFFICIAL_2026_GROUPS
from .simulate import simulate_group, apply_2026_qualification_statuses

def predict_2026(args):
    cfg=load_config(args.config)
    results_path=Path(args.results) if args.results else Path("data/raw/results.csv")
    if not results_path.exists():
        results_path=download_kaggle(results_path.parent)
    results=load_results(results_path)
    teams=[t for group in OFFICIAL_2026_GROUPS.values() for t in group]
    features=compute_team_features(results, teams, args.cutoff, cfg)
    group_payload={}; standings_only={}
    for g,ts in OFFICIAL_2026_GROUPS.items():
        payload=simulate_group(g,ts,features,cfg); group_payload[g]=payload; standings_only[g]=payload["standings"]
    apply_2026_qualification_statuses(standings_only)
    for g in group_payload: group_payload[g]["standings"]=standings_only[g]
    out={"cutoff":args.cutoff,"source":"Kaggle martj42 international-football-results + official 2026 groups from Wikipedia/FIFA schedule","groups":group_payload}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"wrote {args.output}")
    for g,payload in group_payload.items():
        print(f"Group {g}: " + ", ".join(f"{r['predicted_rank']}. {r['team']} {r['predicted_pts']}pts {r['status']}" for r in payload["standings"]))

def metadata(_args):
    md=kaggle_metadata()
    print(json.dumps({"title":md.get("titleNullable"),"subtitle":md.get("subtitleNullable"),"version":md.get("currentVersionNumber"),"total_bytes":md.get("totalBytesNullable"),"usability":md.get("usabilityRatingNullable")},indent=2,ensure_ascii=False))

def main(argv=None):
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(required=True)
    m=sub.add_parser("kaggle-metadata"); m.set_defaults(func=metadata)
    p=sub.add_parser("predict-2026")
    p.add_argument("--config",default="config.yaml"); p.add_argument("--results"); p.add_argument("--cutoff",default="2026-06-10"); p.add_argument("--output",default="outputs/predictions_2026.json")
    p.set_defaults(func=predict_2026)
    args=ap.parse_args(argv); args.func(args)
if __name__ == "__main__": main()
