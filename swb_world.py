"""Das Vehikel: ein Netz aus Standorten mit Order-up-to-Bestand und Nachfrageüberlauf zwischen Nachbarn.

Zeitablauf je Tag t und Standort s (alle Standorte und alle simulierten Netze gleichzeitig, numpy):
 1. Wareneingang: was am Ende von Tag t - LEAD bestellt wurde, kommt zum Netto-Bestand.
 2. Nachfrage D (negativ-binomial, Standortgröße, Wochentagsfaktor). Bedient wird aus dem Bestand: own = min(max(NI, 0), D).
 3. Fehlmenge = D - own. Der Anteil q der Fehlmenge versucht es am selben Tag beim Nachbarn (Überlauf o = q * Fehlmenge);
    dort wird bedient, was nach der eigenen Nachfrage des Nachbarn im Bestand übrig ist (vin = min(o_Nachbar, Rest)).
    Der Rest des Überlaufs ist verloren; die übrige Fehlmenge (1 - q) bleibt Rückstand.
 4. Bestellung bis S = round(LEAD * mu + z * sd) über der Bestandsposition; das Zentrallager liefert unbegrenzt.
Erhaltung je Herkunftsstandort: D = own + beim Nachbarn bedient + verloren + Rückstand.

Die Nachfrage der Netze ist vorab gezogen (gleiche Nachfrage in allen Welten: gepaarte Vergleiche); aller Zufall kommt aus
SplitMix64 (`swb_rng.uniform_block`), die negativ-binomiale Inversion über eine Verteilungstabelle je Standort und Wochentag."""

import numpy as np
from scipy.stats import nbinom

import swb_constants as C
from swb_rng import uniform_block

TOTAL_DAYS = C.BURN_DAYS + C.DAYS


def partner_index(n_sites=C.N_SITES):
    """Nachbar jedes Standorts: 0 <-> 1, 2 <-> 3, ..."""
    return np.arange(n_sites) ^ 1


def demand_worlds(worlds, seed, n_sites=C.N_SITES, burn=C.BURN_DAYS, days=C.DAYS):
    """Nachfrage (worlds, n_sites, burn + days) als ganze Zahlen. Tag 0 ist ein Montag. Die ersten `burn` Tage dienen dem
    Einschwingen."""
    total = burn + days
    mu = C.site_means(n_sites)
    dow = C.dow_factors()
    u = uniform_block(seed, worlds * n_sites * total).reshape(worlds, n_sites, total)
    out = np.empty((worlds, n_sites, total), dtype=np.int32)
    for s in range(n_sites):
        for d in range(7):
            mean = mu[s] * dow[d]
            p = C.DISP / (C.DISP + mean)
            kmax = int(nbinom.ppf(1 - 1e-12, C.DISP, p)) + 2
            cdf = nbinom.cdf(np.arange(kmax + 1), C.DISP, p)
            out[:, s, d::7] = np.searchsorted(cdf, u[:, s, d::7], side="left")
    return out


def order_levels(z, n_sites=C.N_SITES):
    """Bestellniveau S je Standort für den Sicherheitsfaktor z (Form (..., n_sites, T) broadcastet)."""
    mu = C.site_means(n_sites)
    sd = np.sqrt(C.LEAD * (mu + mu**2 / C.DISP))
    return np.round(C.LEAD * mu[:, None] + np.asarray(z, dtype=float) * sd[:, None])


def simulate(demand, z, q, burn=C.BURN_DAYS):
    """Simuliert alle Netze. demand: (W, S, burn + days); z: broadcastbar auf (W, S, days) (Einschwingtage nutzen z des ersten
    Tages); q: Anteil der Fehlmenge, der zum Nachbarn ausweicht. Rückgabe: dict mit Summen je Standort (W, S) und je Tag
    (W, days): d, own, vin (hier als Überlauf bedient), vorig (von diesem Herkunftsort woanders bedient), o (Überlauf),
    lost (verloren), back (Rückstand) sowie served_day, dem_day, off_day."""
    w, s, total = demand.shape
    days = total - burn
    partner = partner_index(s)
    zb = np.broadcast_to(np.asarray(z, dtype=float), (w, s, days))
    zfull = np.concatenate([np.repeat(zb[:, :, :1], burn, axis=2), zb], axis=2)
    mu = C.site_means(s)
    sd = np.sqrt(C.LEAD * (mu + mu**2 / C.DISP))
    s_order = np.round(C.LEAD * mu[None, :, None] + zfull * sd[None, :, None])
    ni = np.round(C.LEAD * mu)[None, :].repeat(w, axis=0) + np.round(zfull[:, :, 0] * sd[None, :])
    pipe = [np.zeros((w, s)) for _ in range(C.LEAD)]
    acc = {k: np.zeros((w, s)) for k in ("d", "own", "vin", "vorig", "o", "lost", "back")}
    served_day = np.zeros((w, days))
    dem_day = np.zeros((w, days))
    off_day = np.zeros((w, days))
    for t in range(total):
        ni = ni + pipe[0]
        d = demand[:, :, t].astype(float)
        avail = np.maximum(ni, 0.0)
        own = np.minimum(avail, d)
        short = d - own
        o = q * short
        rest = avail - own
        vin = np.minimum(o[:, partner], rest)
        vorig = vin[:, partner]
        ni = ni - d + o - vin
        order = np.maximum(0.0, s_order[:, :, t] - (ni + sum(pipe[1:], start=np.zeros_like(ni))))
        pipe = pipe[1:] + [order]
        if t >= burn:
            k = t - burn
            acc["d"] += d
            acc["own"] += own
            acc["vin"] += vin
            acc["vorig"] += vorig
            acc["o"] += o
            acc["lost"] += o - vorig
            acc["back"] += short - o
            served_day[:, k] = (own + vin).sum(axis=1)
            dem_day[:, k] = d.sum(axis=1)
            off_day[:, k] = (d + o).sum(axis=1)
    acc.update({"served_day": served_day, "dem_day": dem_day, "off_day": off_day, "partner": partner})
    return acc
