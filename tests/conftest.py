"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import swb_evaluation as E  # noqa: E402


@pytest.fixture
def mini_world():
    """Mini-Instanz: zwei Nachbarstandorte (mu = 12 und 16), ein Tag, Nachfrage 40 und 10, z = 1,0, q = 0,5.

    Handrechnung: sd = sqrt(2 * (mu + mu^2 / 5)) = 9,033 und 11,593; Startbestand 24 + 9 = 33 und 32 + 12 = 44.
      bedient eigen: min(33, 40) = 33 und min(44, 10) = 10; Fehlmenge 7 und 0; Überlauf o = 0,5 * 7 = 3,5 und 0
      Rest-Bestand beim Nachbarn: 0 und 44 - 10 = 34; als Überlauf bedient: min(o_Nachbar, Rest) = 0 und min(3,5; 34) = 3,5
      neuer Bestand: 33 - 40 + 3,5 - 0 = -3,5 und 44 - 10 + 0 - 3,5 = 30,5
      Erhaltung Standort 0: 40 = 33 + 3,5 (beim Nachbarn bedient) + 0 verloren + 3,5 Rückstand"""
    return {"demand": np.array([[[40], [10]]], dtype=np.int32), "z": 1.0, "q": 0.5}


@pytest.fixture(scope="session")
def demand():
    return E.demand_for(300, tag=9)
