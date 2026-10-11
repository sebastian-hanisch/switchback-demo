"""Konstanten: Vehikel, Regler-Grenzen, Farben, Presets. Das Vehikel (Standorte mit Order-up-to-Bestand) ist das der Linie."""

import numpy as np

# Vehikel: S Standorte, tägliche Nachbestellung (Order-up-to), Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt.
N_SITES = 20              # Standorte; je zwei benachbarte (0-1, 2-3, ...) bilden ein Nachbarpaar
BASE_MEAN = 20.0          # mittlere Tagesnachfrage der Standortgröße 1
SIZES = (0.6, 0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.5)          # Standortgrößen relativ zur Basis (reihum)
DOW = (1.15, 1.10, 1.00, 1.00, 1.05, 0.90, 0.80)          # Wochentagsfaktor (Montag zuerst), wird auf Mittel 1 normiert
DISP = 5                  # Dispersion der negativ-binomialen Tagesnachfrage (Gamma-Poisson)
LEAD = 2                  # Wiederbeschaffungszeit in Tagen
BURN_DAYS = 28            # Einschwingtage vor der Messung (4 volle Wochen)
DAYS = 112                # gemessene Tage (16 Wochen)
Z_A = 1.0                 # Sicherheitsfaktor der bestehenden Regel A
Z_B = 1.4                 # Sicherheitsfaktor der neuen Regel B
ALPHA_NOTE = None

# Stück 10: Interferenz und Switchback
Q_MIN, Q_MAX, DEFAULT_Q, Q_STEP = 0.0, 1.0, 0.3, 0.1      # Anteil der Fehlmenge, der zum Nachbarn ausweicht
L_MIN, L_MAX, DEFAULT_L = 1, 28, 14                       # Blocklänge des Switchback (Tage)
K_MIN, K_MAX, DEFAULT_K = 0, 5, 2                         # weggelassene Tage am Blockanfang (Einschwingen)
METRICS = ("herkunft", "lokal")
METRIC_LABELS = {
    "herkunft": "Kunde zählt, wo er bedient wird (Herkunft)",
    "lokal": "Standort zählt, was bei ihm bedient wird (lokal)",
}
SEED_MAX, DEFAULT_SEED = 999999, 458
WORLDS_LIVE = 1000        # simulierte Netze (Wiederholungen) in der Live-Auswertung

METHODS = ("complete", "inpair", "cluster", "switchback")
METHOD_LABELS = {
    "complete": "Je Standort zufällig (Hälfte der Standorte B)",
    "inpair": "Je Standort zufällig innerhalb jedes Nachbarpaars",
    "cluster": "Nachbarpaare gemeinsam zufällig (Cluster)",
    "switchback": "Switchback: alle Standorte wechseln blockweise",
}
METHOD_SHORT = {"complete": "Je Standort", "inpair": "Im Nachbarpaar", "cluster": "Cluster Paare", "switchback": "Switchback"}
METHOD_COLORS = {"complete": "#c0392b", "inpair": "#d95f02", "cluster": "#2F6B65", "switchback": "#4c78a8"}

COLOR_A = "#4c78a8"
COLOR_B = "#2F6B65"
COLOR_TRUTH = "#d95f02"
COLOR_MISS = "#c0392b"

PRESETS = {
    "Nachbarn teilen Kundschaft (Voreinstellung)": {"q": DEFAULT_Q, "L": DEFAULT_L, "k": DEFAULT_K, "metric": 0, "seed": DEFAULT_SEED},
    "Keine Interferenz": {"q": 0.0, "L": DEFAULT_L, "k": DEFAULT_K, "metric": 0, "seed": DEFAULT_SEED},
    "Starke Interferenz": {"q": 1.0, "L": DEFAULT_L, "k": DEFAULT_K, "metric": 0, "seed": DEFAULT_SEED},
    "Switchback ohne Einschwingen": {"q": DEFAULT_Q, "L": 7, "k": 0, "metric": 0, "seed": DEFAULT_SEED},
}
PRESET_ORDER = tuple(PRESETS)
PRESET_HELP = {
    "Nachbarn teilen Kundschaft (Voreinstellung)": "30 % der Fehlmenge weicht zum Nachbarn aus; Blöcke von 14 Tagen, die ersten 2 Tage jedes Blocks bleiben außen vor.",
    "Keine Interferenz": "Niemand weicht aus: der gewöhnliche Standort-Vergleich ist unverzerrt, und der Switchback bringt nur Nachteile.",
    "Starke Interferenz": "Die ganze Fehlmenge versucht es beim Nachbarn: der Standort-Vergleich zeigt eine andere Wirkung als das System tatsächlich hat.",
    "Switchback ohne Einschwingen": "Blöcke von 7 Tagen und kein Tag weggelassen: die Bestellungen von gestern bestimmen den Servicegrad von heute.",
}


def site_means(n_sites=N_SITES):
    sizes = np.array(SIZES, dtype=float)
    return BASE_MEAN * np.tile(sizes, int(np.ceil(n_sites / len(sizes))))[:n_sites]


def dow_factors():
    d = np.array(DOW, dtype=float)
    return d / d.mean()


def fmt_pct(x, digits=1):
    """Anteil (0..1) als Prozentzahl mit Dezimalkomma, z. B. 0,934 -> '93,4 %'."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def fmt_pp(x, digits=2):
    """Differenz in Prozentpunkten (aus Anteilen 0..1) mit Vorzeichen und Dezimalkomma."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pkt(x, digits=1):
    """Kurzform für enge Kacheln: '+4,0 Pkt.' (Differenz in Prozentpunkten, aus Anteilen 0..1)."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Pkt."


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(x):,}".replace(",", " ")
