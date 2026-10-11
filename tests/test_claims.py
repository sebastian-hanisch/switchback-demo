"""Jede Zahl des README (Abschnitt "Befunde") wird hier nachgerechnet.

Die Zahlen stammen aus `precomputed_sweeps.json` (2 000 simulierte Netze je Zelle, SplitMix64, Seeds fest); zwei Zellen werden
zusätzlich mit denselben Funktionen wie in der App neu gerechnet, damit die Datei nicht von der Rechnung abweicht."""

import json
from pathlib import Path

import pytest

import swb_constants as C
import swb_evaluation as E

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cells():
    data = json.loads((ROOT / "precomputed_sweeps.json").read_text(encoding="utf-8"))
    return {(c["sweep"], round(c["q"], 1), c["L"], c["k"], c["metric"]): c["rows"] for c in data["cells"]}


def q_row(cells, q, metric="herkunft"):
    return cells[("q", q, C.DEFAULT_L, C.DEFAULT_K, metric)]


def l_row(cells, length, k):
    return cells[("L", C.DEFAULT_Q, length, k, "herkunft")]["switchback"]


def test_the_sweep_file_matches_a_fresh_computation(cells):
    demand = E.demand_for(2000, tag=3)
    for q, metric in ((0.3, "herkunft"), (1.0, "lokal")):
        fresh = E.summarize(E.evaluate(demand, q, C.DEFAULT_L, C.DEFAULT_K, metric, E.seed_for(q, 3)))
        stored = q_row(cells, q, metric)
        for m in C.METHODS:
            for key in ("bias", "sd", "rmse", "mean"):
                assert fresh[m][key] == pytest.approx(stored[m][key], abs=2e-3), (q, metric, m, key)


def test_the_effect_on_the_system_shrinks_with_the_overflow(cells):
    expected = {0.0: 3.54, 0.2: 2.87, 0.3: 2.59, 0.5: 2.14, 1.0: 1.40}
    for q, v in expected.items():
        assert q_row(cells, q)["_truth"]["mean"] == pytest.approx(v, abs=0.01), q


def test_neighbours_move_each_other(cells):
    for q, (a, b) in {0.3: (0.26, -0.16), 1.0: (0.71, -0.44)}.items():
        t = q_row(cells, q)["_truth"]
        assert (t["decomp_a"], t["decomp_b"]) == pytest.approx((a, b), abs=0.01), q
    assert q_row(cells, 0.0)["_truth"]["decomp_a"] == 0.0


def test_bias_by_design_with_origin_metric(cells):
    expected = {
        0.3: {"complete": (-0.22, -9), "inpair": (-0.43, -17), "cluster": (0.0, 0), "switchback": (-0.08, -3)},
        1.0: {"complete": (-0.59, -42), "inpair": (-1.15, -82), "cluster": (-0.01, -1), "switchback": (-0.10, -7)},
    }
    for q, per in expected.items():
        for m, (bias, rel) in per.items():
            r = q_row(cells, q)[m]
            assert r["bias"] == pytest.approx(bias, abs=0.02), (q, m)
            assert 100 * r["rel_bias"] == pytest.approx(rel, abs=1.0), (q, m)


def test_without_overflow_every_design_is_unbiased(cells):
    for m in C.METHODS:
        assert abs(q_row(cells, 0.0)[m]["bias"]) < 0.1, m
    assert abs(q_row(cells, 0.0)["complete"]["bias"]) < 0.02


def test_typical_error_by_design(cells):
    expected = {
        0.0: {"complete": 1.13, "inpair": 1.10, "cluster": 1.10, "switchback": 1.19},
        0.3: {"complete": 0.81, "inpair": 0.84, "cluster": 0.84, "switchback": 0.86},
        1.0: {"complete": 0.78, "inpair": 1.22, "cluster": 0.54, "switchback": 0.57},
    }
    for q, per in expected.items():
        for m, v in per.items():
            assert q_row(cells, q)[m]["rmse"] == pytest.approx(v, abs=0.02), (q, m)


def test_wrong_sign_at_full_overflow(cells):
    row = q_row(cells, 1.0)
    got = {m: 100 * row[m]["wrong_sign"] for m in C.METHODS}
    assert got == pytest.approx({"complete": 4.8, "inpair": 26.8, "cluster": 0.8, "switchback": 0.8}, abs=0.3)


def test_the_local_metric_overestimates_everywhere(cells):
    expected = {
        0.3: {"complete": 0.62, "inpair": 0.41, "cluster": 0.83, "switchback": 0.75},
        1.0: {"complete": 1.47, "inpair": 0.65, "cluster": 2.31, "switchback": 2.17},
    }
    for q, per in expected.items():
        for m, v in per.items():
            assert q_row(cells, q, "lokal")[m]["bias"] == pytest.approx(v, abs=0.02), (q, m)
    assert 100 * q_row(cells, 1.0, "lokal")["cluster"]["rel_bias"] == pytest.approx(165, abs=2)
    assert 100 * q_row(cells, 1.0, "lokal")["complete"]["rel_bias"] == pytest.approx(105, abs=2)


def test_switchback_without_washout_loses_part_of_the_effect(cells):
    truth = 2.59
    assert l_row(cells, 1, 0)["bias"] == pytest.approx(-2.61, abs=0.03)
    assert 100 * l_row(cells, 1, 0)["wrong_sign"] == pytest.approx(50.1, abs=1.0)
    for length, rel in ((7, 37), (14, 19), (28, 10)):
        assert -100 * l_row(cells, length, 0)["bias"] / truth == pytest.approx(rel, abs=1.0), length
    assert l_row(cells, 7, 0)["bias"] == pytest.approx(-0.95, abs=0.02)
    assert l_row(cells, 14, 0)["bias"] == pytest.approx(-0.49, abs=0.02)
    assert l_row(cells, 28, 0)["bias"] == pytest.approx(-0.26, abs=0.02)


def test_washout_removes_the_carryover_and_the_weekday_trap_remains(cells):
    for length in (3, 10, 14, 28):
        assert abs(l_row(cells, length, 2)["carryover"]) < 0.06, length
    assert l_row(cells, 7, 2)["day_selection"] == pytest.approx(-0.27, abs=0.02)
    assert l_row(cells, 14, 2)["day_selection"] == pytest.approx(-0.11, abs=0.02)
    assert l_row(cells, 10, 2)["day_selection"] == pytest.approx(0.01, abs=0.02)
    assert l_row(cells, 28, 2)["day_selection"] == pytest.approx(-0.05, abs=0.02)


def test_block_length_and_error(cells):
    expected = {3: 1.28, 7: 0.87, 10: 0.94, 14: 0.86, 28: 0.86}
    for length, v in expected.items():
        assert l_row(cells, length, 2)["rmse"] == pytest.approx(v, abs=0.02), length


def test_default_seed_network():
    r = E.run_one({"q": C.DEFAULT_Q, "L": C.DEFAULT_L, "k": C.DEFAULT_K, "metric": "herkunft", "seed": C.DEFAULT_SEED})
    est = {m: 100 * float(v[0]) for m, v in r["est"].items()}
    assert 100 * float(r["truth"][0]) == pytest.approx(2.42, abs=0.02)
    assert est["complete"] == pytest.approx(2.33, abs=0.02)
    assert est["inpair"] == pytest.approx(1.91, abs=0.02)
    assert est["cluster"] == pytest.approx(2.41, abs=0.02)
    assert est["switchback"] == pytest.approx(2.12, abs=0.02)


def test_error_at_small_overflow(cells):
    row = q_row(cells, 0.2)
    got = {m: row[m]["rmse"] for m in ("complete", "cluster", "switchback")}
    assert got == pytest.approx({"complete": 0.89, "cluster": 0.93, "switchback": 0.96}, abs=0.02)
