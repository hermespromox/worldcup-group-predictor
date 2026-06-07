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

## Training / validation

The default `config.yaml` is now calibrated against 2014, 2018, and 2022 group stages.

```bash
PYTHONPATH=src python -m worldcup_predictor.cli validate
PYTHONPATH=src python -m worldcup_predictor.cli train
PYTHONPATH=src python -m worldcup_predictor.cli predict-2026
```

Current trained validation summary:

- 2014: strict qualifiers 2/8, top-2 overlap 1.25/2, winners 5/8
- 2018: strict qualifiers 5/8, top-2 overlap 1.625/2, winners 5/8
- 2022: strict qualifiers 2/8, top-2 overlap 1.25/2, winners 6/8

The selected model is currently `elo_only`: it uses an international Elo rating computed only from matches before each cutoff, then converts Elo differences into xG for Poisson group simulation. This performed better overall than the raw recent-goals model on top-2 overlap.

## Status

Usable trained MVP: validation, training, and 2026 prediction all run locally. FIFA ranking baseline still needs historical ranking snapshots.
