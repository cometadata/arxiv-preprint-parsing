import math

from generic_evaluator import GenericEvaluator
from evaluation_utils import resolve_entity_category_key


def test_generic_evaluator_handles_fuzzy_matching_and_aggregation():
    gt_entities = [
        {
            "name": "Alice B.",
            "affiliations": ["University A", "Lab B"],
            "email": "alice@example.com",
        },
        {
            "name": "Charlie D",
            "affiliations": ["Institute C"],
            "email": "charlie@example.com",
        },
    ]
    pred_entities = [
        {
            "name": "alice b",
            "affiliations": ["University A", "Lab B", "Consortium"],
            "email": "alice@example.com",
        },
        {
            "name": "Charlie D",
            "affiliations": ["Institute C"],
            "email": "charlie@lab.org",
        },
        {
            "name": "Dana",
            "affiliations": ["Independent"],
            "email": "dana@example.com",
        },
    ]
    config = {
        "key_field": "name",
        "field_eval_rules": {
            "name": {"match_type": "fuzzy", "similarity_threshold": 0.6, "normalization": False},
            "affiliations": {"match_type": "fuzzy", "similarity_threshold": 0.8, "normalization": True},
            "email": {"match_type": "strict", "normalization": False},
        },
        "combined_eval": {"harsh_penalty": True},
    }
    schema = {
        "entities_field": "authors",
        "fields": {
            "name": {"type": "string"},
            "affiliations": {"type": "array[string]"},
            "email": {"type": "string"},
        },
    }

    evaluator = GenericEvaluator(
        gt_entities=gt_entities,
        pred_entities=pred_entities,
        doc_id="test-doc",
        config=config,
        schema=schema,
    )

    entity_category = resolve_entity_category_key(schema, config)
    result = evaluator.evaluate()

    entity_metrics = result.metrics[entity_category]
    strict_author = entity_metrics["strict"]
    assert math.isclose(strict_author.precision, 1 / 3, rel_tol=1e-9)
    assert math.isclose(strict_author.recall, 0.5, rel_tol=1e-9)

    fuzzy_author = entity_metrics["fuzzy"]
    assert math.isclose(fuzzy_author.precision, 2 / 3, rel_tol=1e-9)
    assert math.isclose(fuzzy_author.recall, 1.0, rel_tol=1e-9)

    aff_metrics = result.metrics.get("field:affiliations")
    assert aff_metrics is not None
    strict_aff = aff_metrics["strict"]
    assert math.isclose(strict_aff.precision, 1 / 5, rel_tol=1e-9)
    assert math.isclose(strict_aff.recall, 1 / 3, rel_tol=1e-9)

    fuzzy_aff = aff_metrics["fuzzy"]
    assert math.isclose(fuzzy_aff.precision, 3 / 5, rel_tol=1e-9)
    assert math.isclose(fuzzy_aff.recall, 1.0, rel_tol=1e-9)

    combined_metrics = result.metrics["combined"]
    strict_combined = combined_metrics["strict"]
    assert strict_combined.true_positives == 0
    assert strict_combined.false_positives == 3
    assert strict_combined.false_negatives == 2

    fuzzy_combined = combined_metrics["fuzzy"]
    assert fuzzy_combined.true_positives == 0
    assert fuzzy_combined.false_positives == 3
    assert fuzzy_combined.false_negatives == 2

    assert len(result.details["entity_matches"]) == 5
    array_details = result.details.get("array_field_details", {})
    assert len(array_details.get("affiliations", [])) == 2


def test_generic_evaluator_filters_entities_missing_key_field():
    gt_entities = [
        {"name": "Alice"},
        {"name": "Bob"},
        {"identifier": "skip-me"},
    ]
    pred_entities = [
        {"name": "Alice"},
        {"identifier": "skip-me-too"},
        {"name": "Charlie"},
    ]
    config = {
        "key_field": "name",
        "field_eval_rules": {
            "name": {"match_type": "strict", "normalization": True},
        },
    }
    schema = {
        "entities_field": "authors",
        "fields": {
            "name": {"type": "string"},
        },
    }

    evaluator = GenericEvaluator(
        gt_entities=gt_entities,
        pred_entities=pred_entities,
        doc_id="doc-2",
        config=config,
        schema=schema,
    )

    assert len(evaluator.gt_entities) == 2
    assert len(evaluator.pred_entities) == 2

    entity_category = resolve_entity_category_key(schema, config)
    result = evaluator.evaluate()

    strict_entity = result.metrics[entity_category]["strict"]
    assert math.isclose(strict_entity.precision, 0.5, rel_tol=1e-9)
    assert math.isclose(strict_entity.recall, 0.5, rel_tol=1e-9)

    assert "field:affiliations" not in result.metrics

    strict_combined = result.metrics["combined"]["strict"]
    assert strict_combined.true_positives == 1
    assert strict_combined.false_positives == 1
    assert strict_combined.false_negatives == 1


def test_generic_evaluator_combined_metrics_without_harsh_penalty():
    gt_entities = [
        {"name": "Alice", "email": "alice@truth.org"},
    ]
    pred_entities = [
        {"name": "Alice", "email": "alice@prediction.org"},
    ]
    config = {
        "key_field": "name",
        "field_eval_rules": {
            "name": {"match_type": "strict", "normalization": True},
            "email": {"match_type": "strict", "normalization": False},
        },
        "combined_eval": {"harsh_penalty": False},
    }
    schema = {
        "entities_field": "authors",
        "fields": {
            "name": {"type": "string"},
            "email": {"type": "string"},
        },
    }

    evaluator = GenericEvaluator(
        gt_entities=gt_entities,
        pred_entities=pred_entities,
        doc_id="doc-penalty",
        config=config,
        schema=schema,
    )

    entity_category = resolve_entity_category_key(schema, config)
    result = evaluator.evaluate()

    strict_author = result.metrics[entity_category]["strict"]
    assert math.isclose(strict_author.precision, 1.0, rel_tol=1e-9)
    assert math.isclose(strict_author.recall, 1.0, rel_tol=1e-9)

    strict_combined = result.metrics["combined"]["strict"]
    assert strict_combined.true_positives == 0
    assert strict_combined.false_positives == 0
    assert strict_combined.false_negatives == 0
    assert strict_combined.f1_score == 0.0

    fuzzy_combined = result.metrics["combined"]["fuzzy"]
    assert fuzzy_combined.true_positives == 0
    assert fuzzy_combined.false_positives == 0
    assert fuzzy_combined.false_negatives == 0


def test_generic_evaluator_empty_inputs_yield_zero_metrics():
    config = {"key_field": "name", "field_eval_rules": {"name": {"match_type": "strict"}}}
    schema = {
        "entities_field": "authors",
        "fields": {"name": {"type": "string"}},
    }
    evaluator = GenericEvaluator(
        gt_entities=[],
        pred_entities=[],
        doc_id="empty",
        config=config,
        schema=schema,
    )

    entity_category = resolve_entity_category_key(schema, config)
    result = evaluator.evaluate()

    entity_strict = result.metrics[entity_category]["strict"]
    assert entity_strict.true_positives == 0
    assert entity_strict.false_positives == 0
    assert entity_strict.false_negatives == 0
    assert entity_strict.f1_score == 0.0

    combined_strict = result.metrics["combined"]["strict"]
    assert combined_strict.true_positives == 0
    assert combined_strict.false_positives == 0
    assert combined_strict.false_negatives == 0

    assert result.details["entity_matches"] == []
    assert result.details.get("array_field_details", {}) == {}
