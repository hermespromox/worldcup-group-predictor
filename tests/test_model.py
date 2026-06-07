from worldcup_predictor.model import poisson_score_matrix, predict_match


def test_poisson_score_matrix_probability_mass_is_near_one():
    matrix = poisson_score_matrix(1.4, 0.9, max_goals=10)
    assert 0.995 <= matrix.values.sum() <= 1.0


def test_predict_match_returns_probabilities_and_expected_points():
    pred = predict_match("A", "B", 1.5, 1.0, max_goals=10)
    assert pred.team_a == "A"
    assert pred.team_b == "B"
    assert abs(pred.p_a_win + pred.p_draw + pred.p_b_win - 1.0) < 0.01
    assert pred.expected_pts_a > pred.expected_pts_b
    assert isinstance(pred.most_likely_score, str)
