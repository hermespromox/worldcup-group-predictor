# World Cup Group Phase Predictor

Transparent Python predictor for World Cup group stages using historical international results.

## What it predicts

For each group it outputs:
- expected group ranking
- predicted points / goal difference / goals for
- per-match win/draw/loss probabilities and most likely scoreline
- 2026 status: `automatic_qualifier`, `third_place_qualifier`, or `eliminated`

## Data

Primary dataset: Kaggle `martj42/international-football-results-from-1872-to-2017`, currently titled **International football results from 1872 to 2026** via Kaggle API.

Official 2026 groups are included from the final draw / Wikipedia schedule pages.

## Usage

```bash
python -m pip install -e . pytest
python -m worldcup_predictor.cli kaggle-metadata
python -m worldcup_predictor.cli predict-2026 --cutoff 2026-06-10
```

Output is written to `outputs/predictions_2026.json`.

## Status

MVP: 2026 official group prediction engine implemented. Validation against 2014/2018/2022 and FIFA baseline scaffolding is next.
