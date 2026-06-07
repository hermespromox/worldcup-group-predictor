from worldcup_predictor.simulate import apply_2026_qualification_statuses


def test_2026_marks_top_two_and_best_eight_third_placed_teams():
    groups = {}
    for i, letter in enumerate("ABCDEFGHIJKL"):
        groups[letter] = [
            {"team": f"{letter}1", "predicted_pts": 7, "predicted_gd": 3, "predicted_gf": 5, "predicted_rank": 1},
            {"team": f"{letter}2", "predicted_pts": 5, "predicted_gd": 1, "predicted_gf": 3, "predicted_rank": 2},
            {"team": f"{letter}3", "predicted_pts": 12 - i, "predicted_gd": 0, "predicted_gf": 2, "predicted_rank": 3},
            {"team": f"{letter}4", "predicted_pts": 0, "predicted_gd": -4, "predicted_gf": 1, "predicted_rank": 4},
        ]
    out = apply_2026_qualification_statuses(groups)
    third_statuses = [out[l][2]["status"] for l in "ABCDEFGHIJKL"]
    assert third_statuses.count("third_place_qualifier") == 8
    assert third_statuses.count("eliminated") == 4
    assert all(out[l][0]["status"] == "automatic_qualifier" for l in out)
    assert all(out[l][1]["status"] == "automatic_qualifier" for l in out)
