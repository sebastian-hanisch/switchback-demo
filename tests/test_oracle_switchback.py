"""Orakel-Tests: der Kern gegen einen unabhängigen Rechenweg (ORACLE-PLAYBOOK §2).

1. Nachfrage-Inversion gegen scipy.stats.nbinom.ppf mit denselben Gleichverteilten.
2. Servicegrad ohne Überlauf gegen die exakte Formel der Order-up-to-Regel (unabhängig erzeugte i.i.d. Nachfrage).
3. Der Servicegrad eines Tages hängt nur von der Regel zwei Tage vorher ab (Bestellniveau-Bumps).
4. Erhaltung und Symmetrie des Überlaufs (Summe als Überlauf bedient = Summe beim Nachbarn bedient)."""

import numpy as np
import pytest
from scipy.stats import nbinom

import swb_constants as C
import swb_world as W
from swb_rng import uniform_block


def test_demand_inversion_matches_scipy_ppf():
    worlds, seed = 40, 77
    d = W.demand_worlds(worlds, seed)
    u = uniform_block(seed, worlds * C.N_SITES * (C.BURN_DAYS + C.DAYS)).reshape(worlds, C.N_SITES, C.BURN_DAYS + C.DAYS)
    mu = C.site_means()
    dow = C.dow_factors()
    for s in (2, 9, 17):
        for day in (0, 3, 6):
            m = mu[s] * dow[day]
            ref = nbinom.ppf(u[:, s, day::7], C.DISP, C.DISP / (C.DISP + m)).astype(int)
            assert np.array_equal(d[:, s, day::7], ref)


def fill_rate_formula(mean, disp, lead, z):
    """Exakter Servicegrad (Typ 2) der Order-up-to-Regel mit täglicher Prüfung und i.i.d. negativ-binomialer Nachfrage:
    1 - (E[(D_L - S)+] - E[(D_{L-1} - S)+]) / mean, D_n = Summe von n Tagen = NegBin(n * disp, Mittel n * mean)."""
    sd = np.sqrt(lead * (mean + mean**2 / disp))
    s = int(np.round(lead * mean + z * sd))

    def excess(n):
        if n == 0:
            return max(0.0, -s)
        nn, mm = n * disp, n * mean
        p = nn / (nn + mm)
        k = np.arange(s, s + 2000)
        return float(nbinom.sf(k, nn, p).sum())

    return 1.0 - (excess(lead) - excess(lead - 1)) / mean


def test_service_level_without_overflow_matches_the_exact_formula():
    rng = np.random.default_rng(3)
    worlds, days = 400, 112
    mu = C.site_means(4)
    demand = np.empty((worlds, 4, C.BURN_DAYS + days), dtype=np.int32)
    for s in range(4):
        demand[:, s, :] = nbinom.rvs(C.DISP, C.DISP / (C.DISP + mu[s]), size=(worlds, C.BURN_DAYS + days), random_state=rng)
    for z in (0.0, 1.0, 1.5):
        acc = W.simulate(demand, z, 0.0)
        fill = 1.0 - acc["back"].sum(axis=0) / acc["d"].sum(axis=0)
        for s in range(4):
            assert fill[s] == pytest.approx(fill_rate_formula(mu[s], C.DISP, C.LEAD, z), abs=0.006), (z, s)


def test_a_changed_rule_acts_only_after_the_lead_time():
    """Ein höheres Bestellniveau an Tag 40 ändert den Servicegrad frühestens am Tag 40 + LEAD (Lieferung), kaum noch nach
    Tag 40 + LEAD + 2; die Hauptwirkung liegt am Tag 40 + LEAD. Das ist die Nachwirkung, die der Switchback wegschneiden muss."""
    demand = W.demand_worlds(30, 5)
    base = W.simulate(demand, 1.0, 0.0)
    zd = np.full((30, C.N_SITES, C.DAYS), 1.0)
    zd[:, :, 40] = 2.0
    bump = W.simulate(demand, zd, 0.0)
    diff = np.abs(bump["served_day"] - base["served_day"]).sum(axis=0)
    assert diff[: 40 + C.LEAD].max() < 1e-9                          # Kausalität: nichts vor der Lieferung
    assert diff[40 + C.LEAD] == diff.max() and diff[40 + C.LEAD] > 0
    assert diff[40 + C.LEAD + 3:].max() < 0.01 * diff.max()           # und nach wenigen Tagen ausgelaufen


def test_overflow_totals_are_symmetric_and_bounded():
    demand = W.demand_worlds(60, 8)
    for q in (0.2, 1.0):
        acc = W.simulate(demand, 1.0, q)
        assert acc["vin"].sum() == pytest.approx(acc["vorig"].sum())
        assert (acc["vin"] <= acc["o"][:, acc["partner"]] + 1e-9).all()      # bedient höchstens, was angeboten wurde
        assert acc["o"].sum() == pytest.approx(q * (acc["d"] - acc["own"]).sum())
