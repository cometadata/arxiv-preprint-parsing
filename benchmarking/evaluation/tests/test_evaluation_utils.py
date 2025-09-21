import math

from evaluation_utils import (
    aggressive_normalize,
    calculate_similarity,
    calculate_f_beta,
    EvaluationMetrics,
    NormalizedCache,
)


def test_aggressive_normalize_removes_diacritics_and_punctuation():
    text = "  Dr. Jõsé  Núñez!  "
    assert aggressive_normalize(text) == "dr jose nunez"


def test_calculate_similarity_handles_empty_inputs():
    assert calculate_similarity("", "alpha") == 0.0
    assert calculate_similarity("beta", "") == 0.0


def test_calculate_f_beta_balances_precision_and_recall():
    precision = 0.75
    recall = 0.5

    f1 = calculate_f_beta(precision, recall, 1.0)
    assert math.isclose(f1, 0.6, rel_tol=1e-9)

    f2 = calculate_f_beta(precision, recall, 2.0)
    assert math.isclose(f2, 0.5357142857, rel_tol=1e-9)


def test_evaluation_metrics_from_counts_uses_f_beta_family():
    metrics = EvaluationMetrics.from_counts(true_positives=3, false_positives=1, false_negatives=2)

    assert math.isclose(metrics.precision, 0.75, rel_tol=1e-9)
    assert math.isclose(metrics.recall, 0.6, rel_tol=1e-9)
    assert math.isclose(metrics.f1_score, 0.6666666667, rel_tol=1e-9)
    assert math.isclose(metrics.f05_score, calculate_f_beta(metrics.precision, metrics.recall, 0.5), rel_tol=1e-9)
    assert math.isclose(metrics.f15_score, calculate_f_beta(metrics.precision, metrics.recall, 1.5), rel_tol=1e-9)


def test_evaluation_metrics_zero_counts_return_all_zeros():
    metrics = EvaluationMetrics.from_counts(true_positives=0, false_positives=0, false_negatives=0)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0
    assert metrics.f05_score == 0.0
    assert metrics.f15_score == 0.0


def test_evaluation_metrics_handles_no_true_positives():
    metrics = EvaluationMetrics.from_counts(true_positives=0, false_positives=2, false_negatives=1)

    assert metrics.precision == 0.0
    assert math.isclose(metrics.recall, 0.0, rel_tol=1e-9)
    assert metrics.f1_score == 0.0
    assert metrics.f05_score == 0.0
    assert metrics.f15_score == 0.0


def test_normalized_cache_tracks_hits_misses_and_can_clear():
    cache = NormalizedCache()

    first = cache.get_normalized("ÁbC")
    second = cache.get_normalized("ÁbC")
    third = cache.get_normalized("Another")

    assert first == "abc"
    assert second == "abc"
    assert third == "another"

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["cache_size"] == 2
    assert math.isclose(stats["hit_rate"], 1 / 3, rel_tol=1e-9)

    cache.clear()
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["cache_size"] == 0
    assert stats["hit_rate"] == 0
