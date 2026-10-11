"""Das Netz: jede Einheit an der Mini-Instanz von Hand, dazu Erhaltung, Wochentage und Reproduzierbarkeit."""

import numpy as np
import pytest

import swb_constants as C
import swb_world as W


def test_partner_index_pairs_neighbours():
    assert W.partner_index(6).tolist() == [1, 0, 3, 2, 5, 4]


def test_site_means_cycle_through_the_size_list():
    mu = C.site_means(10)
    assert mu[:8].tolist() == pytest.approx([12.0, 16.0, 18.0, 20.0, 20.0, 22.0, 26.0, 30.0])
    assert mu[8:].tolist() == pytest.approx([12.0, 16.0])


def test_order_levels_by_hand():
    # mu = 12: sd = sqrt(2 * (12 + 144/5)) = sqrt(81.6) = 9.033; S = round(24 + 9.033) = 33; mu = 16: sd = 11.593, S = round(43.59) = 44
    lvl = W.order_levels(np.array([[1.0]]), n_sites=2)
    assert lvl[:, 0].tolist() == [33.0, 44.0]


def test_one_day_with_overflow_by_hand(mini_world):
    acc = W.simulate(mini_world["demand"], mini_world["z"], mini_world["q"], burn=0)
    assert acc["own"][0].tolist() == [33.0, 10.0]
    assert acc["o"][0].tolist() == [3.5, 0.0]
    assert acc["vin"][0].tolist() == [0.0, 3.5]                  # beim Standort als Überlauf bedient
    assert acc["vorig"][0].tolist() == [3.5, 0.0]                # vom Herkunftsstandort beim Nachbarn bedient
    assert acc["lost"][0].tolist() == [0.0, 0.0]
    assert acc["back"][0].tolist() == [3.5, 0.0]
    assert acc["served_day"][0, 0] == 46.5 and acc["dem_day"][0, 0] == 50.0 and acc["off_day"][0, 0] == 53.5


def test_overflow_that_the_neighbour_cannot_serve_is_lost():
    # Nachbar hat nach eigener Nachfrage nur 1 Einheit übrig: Bestand 33/44 aber Nachfrage 5 und 43 -> Rest 28 und 1
    demand = np.array([[[45], [43]]], dtype=np.int32)       # Standort 0: Fehlmenge 12, Überlauf 12 (q = 1); Standort 1: Rest 1
    acc = W.simulate(demand, 1.0, 1.0, burn=0)
    assert acc["own"][0].tolist() == [33.0, 43.0]
    assert acc["vin"][0].tolist() == [0.0, 1.0]
    assert acc["lost"][0].tolist() == [11.0, 0.0]
    assert acc["back"][0].tolist() == [0.0, 0.0]


def test_conservation_per_origin_site_and_in_the_system(demand):
    for q in (0.0, 0.3, 1.0):
        acc = W.simulate(demand[:40], C.Z_A, q)
        ident = acc["d"] - (acc["own"] + acc["vorig"] + acc["lost"] + acc["back"])
        assert np.abs(ident).max() < 1e-8
        assert acc["vin"].sum() == pytest.approx(acc["vorig"].sum())
        assert (acc["lost"] >= -1e-9).all() and (acc["back"] >= -1e-9).all()


def test_demand_moments_follow_site_size_and_weekday(demand):
    mu = C.site_means()
    dow = C.dow_factors()
    for s in (0, 7, 13):
        for d in (0, 5):
            cell = demand[:, s, C.BURN_DAYS + d::7]
            m = mu[s] * dow[d]
            assert cell.mean() == pytest.approx(m, rel=0.04)
            assert cell.var() == pytest.approx(m + m * m / C.DISP, rel=0.12)


def test_burn_in_starts_on_a_monday():
    assert C.BURN_DAYS % 7 == 0 and C.DAYS % 7 == 0


def test_demand_is_a_pure_function_of_the_seed():
    a = W.demand_worlds(5, 123)
    b = W.demand_worlds(5, 123)
    c = W.demand_worlds(5, 124)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (5, C.N_SITES, C.BURN_DAYS + C.DAYS)
