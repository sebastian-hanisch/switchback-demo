"""Auswertung: Zuteilungsverfahren bei Interferenz, mit bekannter Wahrheit.

Die Wahrheit ist die Wirkung auf das System: alle Standorte auf Regel B gegen alle Standorte auf Regel A, gleiche Nachfrage, als
Kundenservicegrad des Systems (bedient / Nachfrage). Die Verfahren sehen nur eine gemischte Welt:
  - "complete": die Hälfte der Standorte zufällig auf B (je Standort randomisiert),
  - "inpair": in jedem Nachbarpaar bekommt ein zufälliger Standort B,
  - "cluster": ganze Nachbarpaare zufällig auf B,
  - "switchback": alle Standorte wechseln gemeinsam blockweise, die ersten k Tage jedes Blocks bleiben außen vor.
Kennzahl: Herkunft (Kunde zählt dort, wo er bedient wird) oder lokal (am Standort bedient durch eigene Nachfrage plus
eingehenden Überlauf)."""

import numpy as np

import swb_constants as C
import swb_world as W
from swb_rng import uniform_block


def assign_complete(worlds, n_sites, seed):
    u = uniform_block(seed, worlds * n_sites).reshape(worlds, n_sites)
    return u.argsort(axis=1).argsort(axis=1) < n_sites // 2


def assign_inpair(worlds, n_sites, seed):
    u = uniform_block(seed, worlds * (n_sites // 2)).reshape(worlds, n_sites // 2)
    pick = (u < 0.5).astype(int)
    mask = np.zeros((worlds, n_sites), dtype=bool)
    mask[np.arange(worlds)[:, None], np.arange(n_sites // 2)[None, :] * 2 + pick] = True
    return mask


def assign_cluster(worlds, n_sites, seed):
    npair = n_sites // 2
    u = uniform_block(seed, worlds * npair).reshape(worlds, npair)
    return np.repeat(u.argsort(axis=1).argsort(axis=1) < npair // 2, 2, axis=1)


ASSIGN = {"complete": assign_complete, "inpair": assign_inpair, "cluster": assign_cluster}


def block_layout(days, length):
    """Zahl der Blöcke (gerade) und genutzte Tage bei Blocklänge `length`."""
    nb = days // length
    nb -= nb % 2
    return nb, nb * length


def site_fill(acc, mask, metric):
    """Servicegrad der Standorte in `mask` (W, S): 'herkunft' = (eigen + beim Nachbarn bedient) / eigene Nachfrage;
    'lokal' = (eigen + als Überlauf bedient) / (eigene Nachfrage + eingehender Überlauf)."""
    w = mask.astype(float)
    if metric == "herkunft":
        return (w * (acc["own"] + acc["vorig"])).sum(axis=1) / (w * acc["d"]).sum(axis=1)
    incoming = acc["o"][:, acc["partner"]]
    return (w * (acc["own"] + acc["vin"])).sum(axis=1) / (w * (acc["d"] + incoming)).sum(axis=1)


def system_fill(acc):
    return (acc["own"] + acc["vin"]).sum(axis=1) / acc["d"].sum(axis=1)


def switchback_assignment(worlds, nb, seed):
    u = uniform_block(seed, worlds * nb).reshape(worlds, nb)
    return u.argsort(axis=1).argsort(axis=1) < nb // 2


def switchback_estimate(acc, arm_b, length, k, metric, used):
    """Differenz der mittleren Block-Servicegrade (B minus A), ohne die ersten k Tage jedes Blocks."""
    nb = arm_b.shape[1]
    k = min(k, length - 1)
    served = acc["served_day"][:, :used].reshape(-1, nb, length)[:, :, k:].sum(axis=2)
    denom = (acc["dem_day"] if metric == "herkunft" else acc["off_day"])[:, :used].reshape(-1, nb, length)[:, :, k:].sum(axis=2)
    fill = served / denom
    nb_b = arm_b.sum(axis=1)
    nb_a = nb - nb_b
    return (fill * arm_b).sum(axis=1) / nb_b - (fill * ~arm_b).sum(axis=1) / nb_a


def seed_for(q, tag, base=40_000_000):
    return base + 10_007 * int(round(q * 100)) + 131 * tag


def evaluate(demand, q, length, k, metric, seed):
    """Alle vier Verfahren auf denselben Netzen. Rückgabe: dict mit Schätzungen je Verfahren (W,), Wahrheit (W,) (System,
    alle Tage) und Wahrheit auf den Tagen des Switchback, dazu die Zerlegung der Wirkung im Nachbarpaar-Design."""
    worlds, n_sites, total = demand.shape
    days = total - C.BURN_DAYS
    acc_a = W.simulate(demand, C.Z_A, q)
    acc_b = W.simulate(demand, C.Z_B, q)
    truth = system_fill(acc_b) - system_fill(acc_a)
    out = {"truth": truth, "est": {}, "kind": metric, "masks": {}}
    for m in ("complete", "inpair", "cluster"):
        mask = ASSIGN[m](worlds, n_sites, seed + 1_000_003 * (1 + ("complete", "inpair", "cluster").index(m)))
        acc = W.simulate(demand, np.where(mask, C.Z_B, C.Z_A)[:, :, None], q)
        out["est"][m] = site_fill(acc, mask, metric) - site_fill(acc, ~mask, metric)
        out["masks"][m] = mask
        if m == "inpair":
            # wie stark verändert die Nachbarschaft die Standorte? Dieselben Standorte in der reinen Welt als Vergleich
            out["decomp_b"] = site_fill(acc, mask, "herkunft") - site_fill(acc_b, mask, "herkunft")
            out["decomp_a"] = site_fill(acc, ~mask, "herkunft") - site_fill(acc_a, ~mask, "herkunft")
    nb, used = block_layout(days, length)
    arm_b = switchback_assignment(worlds, nb, seed + 7_000_003)
    zday = np.full((worlds, days), C.Z_A)
    zday[:, :used] = C.Z_A + (C.Z_B - C.Z_A) * np.repeat(arm_b, length, axis=1)
    acc_s = W.simulate(demand, zday[:, None, :], q)
    out["est"]["switchback"] = switchback_estimate(acc_s, arm_b, length, k, metric, used)
    served_b, dem_b = acc_b["served_day"][:, :used].sum(axis=1), acc_b["dem_day"][:, :used].sum(axis=1)
    served_a, dem_a = acc_a["served_day"][:, :used].sum(axis=1), acc_a["dem_day"][:, :used].sum(axis=1)
    out["truth_switchback"] = served_b / dem_b - served_a / dem_a
    kept = (np.arange(used) % length) >= min(k, length - 1)             # Tage, die der Switchback auswertet
    out["truth_kept"] = (acc_b["served_day"][:, :used][:, kept].sum(axis=1) / acc_b["dem_day"][:, :used][:, kept].sum(axis=1)
                         - acc_a["served_day"][:, :used][:, kept].sum(axis=1) / acc_a["dem_day"][:, :used][:, kept].sum(axis=1))
    out["layout"] = {"blocks": nb, "used": used, "k": min(k, length - 1), "arm_b": arm_b}
    out["fill_a"], out["fill_b"] = system_fill(acc_a), system_fill(acc_b)
    out["lost_a"] = acc_a["lost"].sum(axis=1) / acc_a["d"].sum(axis=1)
    out["lost_b"] = acc_b["lost"].sum(axis=1) / acc_b["d"].sum(axis=1)
    return out


def summarize(result):
    """Kennzahlen je Verfahren in Prozentpunkten: Verzerrung (Mittel der Abweichung von der Wahrheit des Netzes), Streuung,
    RMSE, Anteil der Netze mit Schätzung <= 0, relative Verzerrung."""
    rows = {}
    for m, est in result["est"].items():
        truth = result["truth_switchback"] if m == "switchback" else result["truth"]
        err = 100 * (est - truth)
        n = len(est)
        rows[m] = {
            "bias": float(err.mean()), "bias_se": float(err.std(ddof=1) / np.sqrt(n)),
            "sd": float(100 * est.std(ddof=1)), "rmse": float(np.sqrt((err**2).mean())),
            "wrong_sign": float((est <= 0).mean()), "mean": float(100 * est.mean()), "truth": float(100 * truth.mean()),
            "rel_bias": float(err.mean() / (100 * truth.mean())) if truth.mean() else float("nan"),
        }
        if m == "switchback":
            rows[m]["day_selection"] = float(100 * (result["truth_kept"] - result["truth_switchback"]).mean())
            rows[m]["carryover"] = float(100 * (est - result["truth_kept"]).mean())
    rows["_truth"] = {"mean": float(100 * result["truth"].mean()), "sd": float(100 * result["truth"].std(ddof=1)),
                      "fill_a": float(result["fill_a"].mean()), "fill_b": float(result["fill_b"].mean()),
                      "lost_a": float(result["lost_a"].mean()), "lost_b": float(result["lost_b"].mean()),
                      "decomp_a": float(100 * result["decomp_a"].mean()), "decomp_b": float(100 * result["decomp_b"].mean())}
    return rows


def demand_for(worlds, tag=0):
    return W.demand_worlds(worlds, 30_000_000 + 17 * tag)


def run_one(settings):
    """Das eine Netz der Seitenleiste (ein simuliertes Netz): alle Verfahren, Wahrheit und Zuteilungen."""
    s = settings
    demand = W.demand_worlds(1, 31_000_000 + 7_919 * int(s["seed"]))
    return evaluate(demand, s["q"], s["L"], s["k"], s["metric"], seed_for(s["q"], 1_000 + int(s["seed"])))
