import pandas as pd
from worldcup_predictor.elo import compute_elo_ratings, match_importance_k, opponent_adjusted_form


def test_world_cup_match_moves_elo_more_than_friendly():
    assert match_importance_k("FIFA World Cup", 20) > match_importance_k("Friendly", 20)


def test_home_advantage_reduces_reward_for_home_win():
    neutral = pd.DataFrame([
        {"date":"2020-01-01","home_team":"A","away_team":"B","home_score":1,"away_score":0,"tournament":"Friendly","neutral":True},
    ])
    home = pd.DataFrame([
        {"date":"2020-01-01","home_team":"A","away_team":"B","home_score":1,"away_score":0,"tournament":"Friendly","neutral":False},
    ])
    neutral_rating = compute_elo_ratings(neutral, ["A","B"], "2021-01-01", home_advantage=75)["A"]
    home_rating = compute_elo_ratings(home, ["A","B"], "2021-01-01", home_advantage=75)["A"]
    assert neutral_rating > home_rating


def test_opponent_adjusted_form_rewards_beating_strong_opponent_more():
    matches = pd.DataFrame([
        {"date":"2020-01-01","home_team":"Weak","away_team":"Strong","home_score":1,"away_score":0,"tournament":"Friendly","neutral":True},
        {"date":"2020-01-02","home_team":"Peer","away_team":"Other","home_score":1,"away_score":0,"tournament":"Friendly","neutral":True},
    ])
    ratings = {"Weak": 1400, "Strong": 1800, "Peer": 1500, "Other": 1500}
    form = opponent_adjusted_form(matches, ["Weak", "Peer"], "2021-01-01", ratings, n=5)
    assert form["Weak"] > form["Peer"]
