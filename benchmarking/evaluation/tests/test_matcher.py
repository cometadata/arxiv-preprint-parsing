from matcher import StrictMatcher, FuzzyMatcher


def test_strict_matcher_normalizes_and_matches_equal_tokens():
    matcher = StrictMatcher(normalization=True)
    matches, unmatched_gt, unmatched_pred = matcher.match_lists(["José"], ["Jose"])

    assert len(matches) == 1
    assert matches[0].score == 1.0
    assert unmatched_gt == []
    assert unmatched_pred == []


def test_fuzzy_matcher_returns_best_matches_above_threshold():
    matcher = FuzzyMatcher(normalization=True, threshold=0.8)
    matches, unmatched_gt, unmatched_pred = matcher.match_lists(["Alpha", "Beta"], ["Aplha", "Gamma"])

    assert len(matches) == 1
    assert matches[0].ground_truth == "Alpha"
    assert matches[0].prediction == "Aplha"
    assert unmatched_gt == [1]
    assert unmatched_pred == [1]


def test_match_lists_handles_empty_inputs():
    matcher = StrictMatcher()
    matches, unmatched_gt, unmatched_pred = matcher.match_lists([], [])

    assert matches == []
    assert unmatched_gt == []
    assert unmatched_pred == []
