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

- 2014: strict qualifiers 3/8, top-2 overlap 1.375/2, winners 5/8
- 2018: strict qualifiers 5/8, top-2 overlap 1.625/2, winners 5/8
- 2022: strict qualifiers 1/8, top-2 overlap 1.125/2, winners 7/8

The model uses recent form + neutral goal metrics + an Elo prior computed from historical matches before the cutoff date.

## Status

Usable trained MVP: validation, training, and 2026 prediction all run locally. FIFA ranking baseline still needs historical ranking snapshots.
