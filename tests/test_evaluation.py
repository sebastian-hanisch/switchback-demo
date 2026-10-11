"""Zuteilungen, Kennzahlen und Schätzer: von Hand an Mini-Instanzen und als Eigenschaften."""

import numpy as np
import pytest

import swb_constants as C
import swb_evaluation as E
import swb_world as W


def test_complete_assignment_puts_half_of_the_sites_on_b():
    m = E.assign_complete(50, 20, 5)
    assert (m.sum(axis=1) == 10).all()


def test_inpair_assignment_puts_exactly_one_site_of_every_pair_on_b():
    m = E.assign_inpair(50, 20, 6)
    assert (m[:, 0::2] ^ m[:, 1::2]).all()
    assert (m.sum(axis=1) == 10).all()


def test_cluster_assignment_keeps_pairs_together_and_balanced():
    m = E.assign_cluster(50, 20, 7)
    assert (m[:, 0::2] == m[:, 1::2]).all()
    assert (m.sum(axis=1) == 10).all()


def test_assignments_are_random_and_reproducible():
    a, b, c = E.assign_inpair(400, 20, 1), E.assign_inpair(400, 20, 1), E.assign_inpair(400, 20, 2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert 0.42 < a[:, 0].mean() < 0.58


def test_block_layout_uses_an_even_number_of_whole_blocks():
    assert E.block_layout(112, 14) == (8, 112)
    assert E.block_layout(112, 3) == (36, 108)
    assert E.block_layout(112, 28) == (4, 112)
    assert E.block_layout(112, 1) == (112, 112)


def test_site_fill_by_hand(mini_world):
    acc = W.simulate(mini_world["demand"], mini_world["z"], mini_world["q"], burn=0)
    first, second = np.array([[True, False]]), np.array([[False, True]])
    assert E.site_fill(acc, first, "herkunft")[0] == pytest.approx((33 + 3.5) / 40)
    assert E.site_fill(acc, second, "herkunft")[0] == pytest.approx(1.0)
    assert E.site_fill(acc, first, "lokal")[0] == pytest.approx(33 / 40)           # kein Überlauf von Standort 1
    assert E.site_fill(acc, second, "lokal")[0] == pytest.approx((10 + 3.5) / (10 + 3.5))
    assert E.system_fill(acc)[0] == pytest.approx(46.5 / 50)


def test_switchback_estimate_by_hand():
    # zwei Blöcke zu 3 Tagen; Block 0 = B. Bedient je Tag 8, 9, 10 (B) und 6, 7, 8 (A); Nachfrage je Tag 10.
    acc = {"served_day": np.array([[8.0, 9.0, 10.0, 6.0, 7.0, 8.0]]), "dem_day": np.full((1, 6), 10.0),
           "off_day": np.full((1, 6), 12.0)}
    arm_b = np.array([[True, False]])
    # k = 0: B-Block 27/30 = 0,9; A-Block 21/30 = 0,7 -> 0,2; k = 1 (ersten Tag weglassen): 19/20 = 0,95 und 15/20 = 0,75 -> 0,2
    assert E.switchback_estimate(acc, arm_b, 3, 0, "herkunft", 6)[0] == pytest.approx(0.2)
    assert E.switchback_estimate(acc, arm_b, 3, 1, "herkunft", 6)[0] == pytest.approx(0.2)
    acc2 = {"served_day": np.array([[5.0, 9.0, 10.0, 6.0, 7.0, 8.0]]), "dem_day": np.full((1, 6), 10.0), "off_day": np.full((1, 6), 12.0)}
    assert E.switchback_estimate(acc2, arm_b, 3, 0, "herkunft", 6)[0] == pytest.approx(24 / 30 - 21 / 30)
    assert E.switchback_estimate(acc2, arm_b, 3, 1, "herkunft", 6)[0] == pytest.approx(0.95 - 0.75)
    assert E.switchback_estimate(acc, arm_b, 3, 0, "lokal", 6)[0] == pytest.approx(27 / 36 - 21 / 36)


def test_without_overflow_the_two_metrics_agree_and_all_designs_are_unbiased(demand):
    r = E.evaluate(demand, 0.0, 14, 2, "herkunft", 11)
    rl = E.evaluate(demand, 0.0, 14, 2, "lokal", 11)
    for m in C.METHODS:
        assert np.allclose(r["est"][m], rl["est"][m])
    s = E.summarize(r)
    for m in ("complete", "inpair", "cluster"):
        assert abs(s[m]["bias"]) < 3.5 * s[m]["bias_se"] + 0.01
    assert s["_truth"]["mean"] == pytest.approx(3.54, abs=0.15)
    assert s["_truth"]["lost_a"] == 0.0


def test_with_overflow_the_neighbour_design_underestimates_and_the_cluster_does_not(demand):
    s = E.summarize(E.evaluate(demand, 0.5, 14, 2, "herkunft", 12))
    assert s["inpair"]["bias"] < -0.3
    assert s["complete"]["bias"] < -0.15
    assert abs(s["cluster"]["bias"]) < 0.15
    assert s["_truth"]["decomp_a"] > 0 > s["_truth"]["decomp_b"]


def test_the_local_metric_flips_the_direction_of_every_design(demand):
    s = E.summarize(E.evaluate(demand, 0.5, 14, 2, "lokal", 12))
    for m in C.METHODS:
        assert s[m]["bias"] > 0.3, m


def test_the_effect_shrinks_with_the_overflow(demand):
    truths = [E.summarize(E.evaluate(demand[:80], q, 14, 2, "herkunft", 13))["_truth"]["mean"] for q in (0.0, 0.5, 1.0)]
    assert truths[0] > truths[1] > truths[2] > 0


def test_switchback_without_washout_is_biased_and_with_washout_is_not(demand):
    s0 = E.summarize(E.evaluate(demand, 0.3, 7, 0, "herkunft", 14))["switchback"]
    s2 = E.summarize(E.evaluate(demand, 0.3, 10, 2, "herkunft", 14))["switchback"]
    assert s0["bias"] < -0.5
    assert abs(s2["bias"]) < 0.25
    assert abs(s2["carryover"]) < 0.25 and abs(s0["carryover"]) > 0.5


def test_block_of_one_day_cannot_see_the_effect(demand):
    s = E.summarize(E.evaluate(demand, 0.3, 1, 0, "herkunft", 15))["switchback"]
    assert abs(s["mean"]) < 0.2 < s["truth"]


def test_evaluation_is_a_pure_function_of_its_inputs(demand):
    a = E.evaluate(demand[:20], 0.3, 14, 2, "herkunft", 16)
    b = E.evaluate(demand[:20], 0.3, 14, 2, "herkunft", 16)
    assert all(np.array_equal(a["est"][m], b["est"][m]) for m in C.METHODS)


def test_run_one_returns_one_network():
    r = E.run_one({"q": 0.3, "L": 14, "k": 2, "metric": "herkunft", "seed": 35})
    assert r["truth"].shape == (1,) and set(r["est"]) == set(C.METHODS)
    assert r["masks"]["inpair"].shape == (1, C.N_SITES)
