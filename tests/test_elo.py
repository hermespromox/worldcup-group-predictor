import pandas as pd
from worldcup_predictor.elo import compute_elo_ratings


def test_compute_elo_ratings_rewards_repeated_wins():
    matches = pd.DataFrame([
        {"date":"2020-01-01","home_team":"A","away_team":"B","home_score":2,"away_score":0,"tournament":"Friendly"},
        {"date":"2020-01-02","home_team":"A","away_team":"B","home_score":1,"away_score":0,"tournament":"Friendly"},
    ])
    ratings = compute_elo_ratings(matches, ["A","B"], "2021-01-01")
    assert ratings["A"] > 1500
    assert ratings["B"] < 1500
