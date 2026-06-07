import pandas as pd
from worldcup_predictor.validation import actual_group_table, validation_metrics


def test_actual_group_table_computes_points_and_rank():
    matches = pd.DataFrame([
        {"home_team":"A","away_team":"B","home_score":1,"away_score":0},
        {"home_team":"C","away_team":"D","home_score":0,"away_score":0},
        {"home_team":"A","away_team":"C","home_score":2,"away_score":2},
        {"home_team":"B","away_team":"D","home_score":3,"away_score":0},
        {"home_team":"A","away_team":"D","home_score":1,"away_score":0},
        {"home_team":"B","away_team":"C","home_score":0,"away_score":0},
    ])
    table = actual_group_table(["A","B","C","D"], matches)
    assert table[0]["team"] == "A"
    assert table[0]["actual_pts"] == 7
    assert table[-1]["team"] == "D"


def test_validation_metrics_gives_top2_overlap():
    predicted = [{"team":"A","predicted_rank":1,"predicted_pts":7},{"team":"B","predicted_rank":2,"predicted_pts":5},{"team":"C","predicted_rank":3,"predicted_pts":2},{"team":"D","predicted_rank":4,"predicted_pts":1}]
    actual = [{"team":"B","actual_rank":1,"actual_pts":7},{"team":"A","actual_rank":2,"actual_pts":5},{"team":"D","actual_rank":3,"actual_pts":2},{"team":"C","actual_rank":4,"actual_pts":1}]
    m = validation_metrics(predicted, actual)
    assert m["top2_overlap"] == 2
    assert m["qualification_exact"] is True
    assert m["winner_correct"] is False
