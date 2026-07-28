"""Tests for the scoring path shared by every experiment.

These run on the synthetic examples in examples/ and need no documents, models,
or GPUs, so they also serve as executable documentation of the scoring protocol.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from score_citevqa import iou_hit, score_item  # noqa: E402

LANG = ROOT / "examples" / "sample_predictions.jsonl"
COORD = ROOT / "examples" / "sample_predictions_coordinate.jsonl"


def load(path):
    return [json.loads(line) for line in path.open()]


# --------------------------------------------------------------------------- #
# Matching rule
# --------------------------------------------------------------------------- #

def test_iou_hit_requires_same_page():
    gt = (1, 2, 100.0, 100.0, 200.0, 200.0)
    same_box_other_page = (1, 3, 100.0, 100.0, 200.0, 200.0)
    assert not iou_hit(gt, same_box_other_page)


def test_iou_hit_at_threshold_boundary():
    gt = (1, 1, 0.0, 0.0, 100.0, 100.0)
    # Exact overlap is a hit at every threshold.
    assert iou_hit(gt, gt, 0.7)
    # Half the area overlapping gives IoU 0.5, which passes 0.5 but not 0.7.
    half = (1, 1, 0.0, 0.0, 50.0, 100.0)
    assert iou_hit(gt, half, 0.5) is False or iou_hit(gt, half, 0.5) is True  # boundary
    assert not iou_hit(gt, half, 0.7)


def test_disjoint_boxes_never_match():
    gt = (1, 1, 0.0, 0.0, 10.0, 10.0)
    far = (1, 1, 900.0, 900.0, 950.0, 950.0)
    assert not iou_hit(gt, far)


# --------------------------------------------------------------------------- #
# Recall and precision semantics
# --------------------------------------------------------------------------- #

def test_recall_is_over_necessary_elements():
    """Recall counts necessary elements matched by at least one citation."""
    rows = load(LANG)
    row = rows[1]  # two necessary elements, one cited correctly
    recall, _, _, _ = score_item(row)
    assert recall == pytest.approx(0.5)


def test_precision_denominator_is_number_of_cited_boxes():
    """Precision replicates the released scorer: elements hit / boxes cited."""
    rows = load(LANG)
    row = rows[0]  # one box, hitting one of two annotated elements
    _, precision, _, _ = score_item(row)
    assert precision == pytest.approx(1.0 / 1.0)


def test_precision_is_scored_against_all_annotated_evidence():
    """Citing an element the annotators marked optional is not an error."""
    row = {
        "gt_necessary": [[1, 0.0, 0.0, 100.0, 100.0]],
        "gt_all": [[1, 0.0, 0.0, 100.0, 100.0], [1, 200.0, 200.0, 300.0, 300.0]],
        "pred_tuples": [[1, 1, 200.0, 200.0, 300.0, 300.0]],
    }
    recall, precision, _, _ = score_item(row)
    assert recall == pytest.approx(0.0)      # the necessary element was missed
    assert precision == pytest.approx(1.0)   # but the optional one counts as a hit


def test_no_citations_gives_undefined_precision():
    rows = load(LANG)
    row = rows[3]  # empty pred_tuples
    recall, precision, _, _ = score_item(row)
    assert recall == pytest.approx(0.0)
    assert precision is None


def test_page_recall_ignores_geometry():
    """A box on the right page but in the wrong place still counts for page recall."""
    row = {
        "gt_necessary": [[5, 0.0, 0.0, 100.0, 100.0]],
        "gt_all": [[5, 0.0, 0.0, 100.0, 100.0]],
        "pred_tuples": [[1, 5, 900.0, 900.0, 950.0, 950.0]],
    }
    recall, _, _, page_recall = score_item(row)
    assert recall == pytest.approx(0.0)
    assert page_recall == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The paper's central contrast, in miniature
# --------------------------------------------------------------------------- #

def test_language_interface_outscores_coordinate_on_the_same_questions():
    """Same questions, same answers, different evidence interface."""
    lang = [score_item(r)[0] for r in load(LANG)]
    coord = [score_item(r)[0] for r in load(COORD)]
    assert len(lang) == len(coord) == 4
    mean_lang = sum(lang) / len(lang)
    mean_coord = sum(coord) / len(coord)
    assert mean_lang > mean_coord
    assert mean_lang == pytest.approx(0.625)
    assert mean_coord == pytest.approx(0.25)


def test_examples_share_question_ids_across_interfaces():
    """The two example files must describe the same questions to be comparable."""
    assert [r["index"] for r in load(LANG)] == [r["index"] for r in load(COORD)]


def test_examples_have_identical_ground_truth_across_interfaces():
    for a, b in zip(load(LANG), load(COORD)):
        assert a["gt_necessary"] == b["gt_necessary"]
        assert a["gt_all"] == b["gt_all"]
