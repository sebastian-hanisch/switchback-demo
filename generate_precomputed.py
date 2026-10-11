"""Erzeugt `precomputed_sweeps.json`: Sweeps über den Ausweichanteil q, die Blocklänge L und die weggelassenen Tage k mit je
2 000 simulierten Netzen (SplitMix64, Seeds fest). Die Nachfrage ist für alle Zellen dieselbe (gepaarte Vergleiche).

    python generate_precomputed.py            # einige Sekunden bis eine Minute"""

import json
import sys
import time
from pathlib import Path

import swb_constants as C
import swb_evaluation as E

OUT = Path(__file__).resolve().parent / "precomputed_sweeps.json"
WORLDS = 2000
Q_GRID = tuple(round(0.1 * i, 1) for i in range(11))
L_GRID = (1, 2, 3, 4, 5, 7, 10, 14, 21, 28)
K_GRID = (0, 2)


def cell(demand, q, length, k, metric):
    res = E.evaluate(demand, q, length, k, metric, E.seed_for(q, 3))
    return E.summarize(res)


def main():
    t0 = time.time()
    demand = E.demand_for(WORLDS, tag=3)
    cells = []
    for q in Q_GRID:
        for metric in C.METRICS:
            cells.append({"sweep": "q", "q": q, "L": C.DEFAULT_L, "k": C.DEFAULT_K, "metric": metric,
                          "rows": cell(demand, q, C.DEFAULT_L, C.DEFAULT_K, metric)})
    for length in L_GRID:
        for k in K_GRID:
            cells.append({"sweep": "L", "q": C.DEFAULT_Q, "L": length, "k": k, "metric": "herkunft",
                          "rows": cell(demand, C.DEFAULT_Q, length, k, "herkunft")})
    meta = {"worlds": WORLDS, "n_sites": C.N_SITES, "days": C.DAYS, "burn_days": C.BURN_DAYS, "z_a": C.Z_A, "z_b": C.Z_B}
    OUT.write_text(json.dumps({"meta": meta, "cells": cells}, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT.name}: {len(cells)} Zellen à {WORLDS} Netze, {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
