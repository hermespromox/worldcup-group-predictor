from worldcup_predictor.model import expected_goals


def test_elo_only_expected_goals_favors_higher_rated_team():
    high = {"elo": 1800, "neutral_avg_goals_scored": 1.0, "neutral_avg_goals_conceded": 1.0}
    low = {"elo": 1400, "neutral_avg_goals_scored": 3.0, "neutral_avg_goals_conceded": 0.2}
    xg_high, xg_low = expected_goals(high, low, 1.35, {"model_type": "elo_only", "elo_goal_slope": 700, "base_xg": 1.35, "xg_ceiling": 5.0})
    assert xg_high > xg_low
    assert 0 < xg_high <= 5.0
    assert 0 < xg_low <= 5.0


def test_blended_expected_goals_combines_stat_and_elo_components():
    high = {"elo": 1800, "neutral_avg_goals_scored": 1.0, "neutral_avg_goals_conceded": 1.0}
    low = {"elo": 1400, "neutral_avg_goals_scored": 3.0, "neutral_avg_goals_conceded": 0.2}
    xg_high, xg_low = expected_goals(high, low, 1.35, {"model_type": "blend", "blend_elo_share": 0.8, "elo_goal_slope": 700, "base_xg": 1.35, "xg_floor": 0.25, "xg_ceiling": 5.0})
    assert xg_high > xg_low
