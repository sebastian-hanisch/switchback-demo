"""Interferenz und Switchback - wenn Standorte einander beeinflussen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Konzepte-Linie "Experimente und Wirkungsmessung": Der Standort-Vergleich setzt voraus, dass die Regel eines
Standorts nur ihn selbst betrifft. Weichen Kunden zum Nachbarn aus, stimmt das nicht mehr. Vier Zuteilungen (je Standort,
im Nachbarpaar, Cluster, Switchback über die Zeit) treten gegeneinander an, mit bekannter Wirkung auf das System. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

import swb_constants as C
from swb_evaluation import demand_for, evaluate, run_one, seed_for, summarize
from swb_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from swb_visualization import (build_assignment, build_block_sweep, build_estimates, build_q_sweep, build_rmse_sweep,
                               build_spread, build_timeline)

st.set_page_config(page_title="Interferenz und Switchback – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _sweeps():
    data = json.loads((Path(__file__).resolve().parent / "precomputed_sweeps.json").read_text(encoding="utf-8"))
    return data["cells"]


@st.cache_data(show_spinner=False)
def _demand():
    return demand_for(C.WORLDS_LIVE)


@st.cache_data(show_spinner=False)
def _one(q, length, k, metric, seed):
    r = run_one({"q": q, "L": length, "k": k, "metric": metric, "seed": seed})
    return {"est": {m: float(v[0]) for m, v in r["est"].items()}, "truth": float(r["truth"][0]),
            "truth_sb": float(r["truth_switchback"][0]), "masks": {m: v[0] for m, v in r["masks"].items()},
            "arm_b": r["layout"]["arm_b"][0], "decomp_a": float(r["decomp_a"][0]), "decomp_b": float(r["decomp_b"][0]),
            "blocks": r["layout"]["blocks"], "used": r["layout"]["used"], "k": r["layout"]["k"]}


@st.cache_data(show_spinner=False)
def _many(q, length, k, metric):
    return summarize(evaluate(_demand(), q, length, k, metric, seed_for(q, 7)))


def _cell(sweep, metric, q=None, length=None, k=None):
    for c in _sweeps():
        if (c["sweep"] == sweep and c["metric"] == metric and (q is None or abs(c["q"] - q) < 1e-9)
                and (length is None or c["L"] == length) and (k is None or c["k"] == k)):
            return c
    raise KeyError((sweep, metric, q, length, k))


st.title("🔀 Interferenz und Switchback: wenn Standorte einander beeinflussen")
st.markdown(
    """
Der Standort-Vergleich im A/B-Test setzt voraus, dass die Regel eines Standorts **nur diesen Standort** betrifft. Hier ist das
nicht so: Fehlmengen weichen **zum Nachbarn aus**, und was dort aus dem Bestand bedient wird, fehlt dem Nachbarn selbst. Dann
misst der Vergleich zwischen Standorten mit Regel A und B nicht mehr die Wirkung der Regel auf das System. Zwei Auswege sind
üblich: **Nachbarn gemeinsam zuteilen** (Cluster) oder **alle Standorte gemeinsam über die Zeit wechseln** (Switchback). Beide
haben ihren Preis.
"""
)
st.caption(
    "Zehntes Stück der Linie „Experimente und Wirkungsmessung“, aufbauend auf "
    "[ratio-metric-demo](https://sebastianhanisch-ratio-metric-demo.streamlit.app/) (Einheit der Auswertung) und "
    "[ab-test-demo](https://sebastianhanisch-ab-test-demo.streamlit.app/) (gleiches Vehikel: Servicegrad aus der "
    "Bestandsregel). Neu: ein **Netz** aus 20 Standorten mit Nachfrageüberlauf zwischen Nachbarn."
)

with st.expander("So ist das Netz aufgebaut", expanded=True):
    st.markdown(
        """
- **20 Standorte in 10 Nachbarpaaren**, tägliche Nachbestellung (Order-up-to), Wiederbeschaffungszeit 2 Tage, 16 Wochen
  Messung. Regel A fährt Sicherheitsfaktor 1,0, Regel B 1,4.
- **Ausweichanteil q:** der Anteil der Fehlmenge, der es am selben Tag beim Nachbarn versucht. Was dort nach dessen eigener
  Nachfrage noch im Bestand ist, wird bedient; der Rest geht verloren.
- **Wahrheit:** alle Standorte auf B gegen alle auf A, gleiche Nachfrage: der Kundenservicegrad des Systems.
- **Vier Zuteilungen:** *je Standort* (die Hälfte aller Standorte zufällig auf B), *im Nachbarpaar* (in jedem Paar bekommt einer
  B), *Cluster* (ganze Paare gemeinsam) und *Switchback* (alle Standorte gleichzeitig, in Blöcken von L Tagen; die ersten k
  Tage jedes Blocks werden nicht ausgewertet, weil die Bestellungen von gestern noch wirken).
- **Zwei Kennzahlen:** *Herkunft* zählt jeden Kunden dort, wo er bedient wird (das ist die Wirkung auf das System); *lokal*
  teilt je Standort das dort Bedienten durch eigene Nachfrage plus eingehenden Überlauf (der Nenner ändert sich mit).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{i}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])
st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Wahrheit**")
    q = st.slider("Ausweichanteil q der Fehlmenge", *bounds("q_slider"), step=C.Q_STEP, key="q_slider", format="%.1f",
                  help="0 = niemand weicht aus (keine Interferenz); 1 = die ganze Fehlmenge versucht es beim Nachbarn.")
    metric_index = st.radio("Kennzahl", options=list(range(len(C.METRICS))), key="metric_radio",
                            format_func=lambda i: C.METRIC_LABELS[C.METRICS[i]],
                            help="Wie aus den Bedienungen ein Servicegrad wird. Nur Herkunft misst die Wirkung auf das System.")
    st.markdown("**Switchback**")
    length = st.slider("Blocklänge L (Tage)", *bounds("l_slider"), step=1, key="l_slider",
                       help="So lange gilt eine Regel, bevor alle Standorte wechseln.")
    k = st.slider("Weggelassene Tage k je Block", *bounds("k_slider"), step=1, key="k_slider",
                  help="Die ersten Tage nach dem Wechsel zählen nicht: die Bestellungen vom Vortag laufen noch mit der alten Regel.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt das eine Netz unten.")
    st.button("🎲 Neues Netz würfeln", on_click=randomize_seed)

q, length, k, metric_index, seed = round(float(q), 1), int(length), int(k), int(metric_index), int(seed)
metric = C.METRICS[metric_index]
sync_query_params({"q_slider": q, "l_slider": length, "k_slider": k, "metric_radio": metric_index, "seed_input": seed})
k_eff = min(k, length - 1)

# --- Ein Netz -------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 Ein Netz")
one = _one(q, length, k, metric, seed)
tag = f"{q}_{length}_{k}_{metric}_{seed}"
st.caption(
    f"20 Standorte, 112 Tage, Ausweichanteil {C.fmt_pct(q, 0)}; Kennzahl: {C.METRIC_LABELS[metric]}. Wahre Wirkung von B auf das "
    f"System: **{C.fmt_pp(one['truth'])}**."
)
st.plotly_chart(build_assignment(one["masks"]), width="stretch", key=f"assign_{tag}")
st.plotly_chart(build_timeline(one["arm_b"], length, k_eff, 112), width="stretch", key=f"timeline_{tag}")
st.plotly_chart(build_estimates(one["est"], one["truth"]), width="stretch", key=f"est_{tag}")
rows_md = ["| Verfahren | Geschätzte Wirkung | Abstand zur Wahrheit |", "|---|---|---|"]
for m in C.METHODS:
    ref = one["truth_sb"] if m == "switchback" else one["truth"]
    rows_md.append(f"| {C.METHOD_LABELS[m]} | {C.fmt_pp(one['est'][m])} | {C.fmt_pp(one['est'][m] - ref)} |")
st.markdown("\n".join(rows_md))
if q > 0:
    st.caption(
        f"Im Nachbarpaar-Design verändert der Nachbar die Standorte: die Standorte mit Regel B liegen um "
        f"{C.fmt_pp(one['decomp_b'])} anders als in einer Welt, in der alle B fahren, die mit Regel A um "
        f"{C.fmt_pp(one['decomp_a'])} anders als in einer Welt, in der alle A fahren (Kennzahl Herkunft). B-Standorte "
        "verlieren, weil ihr schlechter versorgter Nachbar zu ihnen ausweicht; A-Standorte gewinnen, weil ihr Überlauf beim "
        "B-Nachbarn bedient wird. Der Vergleich beider Gruppen ist deshalb kleiner als die Wirkung auf das System."
    )
if k >= length:
    st.caption(f"k = {k} ist nicht kleiner als L = {length}: gewertet wird nur der letzte Tag jedes Blocks.")

# --- Viele Netze ----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 Tausend Netze: wer trifft die Wirkung auf das System?")
with st.spinner("Rechne …"):
    many = _many(q, length, k, metric)
tr = many["_truth"]
st.plotly_chart(build_spread(many), width="stretch", key=f"spread_{q}_{length}_{k}_{metric}")
tab = ["| Verfahren | Mittlere Schätzung | Verzerrung | Streuung | RMSE | Netze mit Wirkung ≤ 0 |", "|---|---|---|---|---|---|"]
for m in C.METHODS:
    r = many[m]
    rel = f" ({C.fmt_num(100 * r['rel_bias'], 0)} %)" if abs(r["truth"]) > 0.05 else ""
    tab.append(f"| {C.METHOD_SHORT[m]} | {C.fmt_num(r['mean'], 2)} Pkt. | {C.fmt_num(r['bias'], 2)} Pkt.{rel} | "
               f"{C.fmt_num(r['sd'], 2)} | {C.fmt_num(r['rmse'], 2)} | {C.fmt_pct(r['wrong_sign'])} |")
st.markdown("\n".join(tab))
st.caption(
    f"Wahrheit im Mittel {C.fmt_num(tr['mean'], 2)} Prozentpunkte (Servicegrad alle A {C.fmt_pct(tr['fill_a'], 1)}, alle B "
    f"{C.fmt_pct(tr['fill_b'], 1)}). Verzerrung = mittlere Abweichung von der Wirkung des jeweiligen Netzes; RMSE = typischer "
    "Fehler in einem Netz. Der Switchback wird gegen die Wahrheit auf den Tagen seines Plans gemessen."
)
if q == 0:
    st.success(
        "**Ohne Ausweichen sind alle vier Verfahren im Mittel richtig.** Der Switchback streut mehr "
        f"(Streuung {C.fmt_num(many['switchback']['sd'], 2)} gegen {C.fmt_num(many['complete']['sd'], 2)} beim Vergleich je "
        "Standort): er tauscht Teststärke gegen einen Schutz, den es hier nicht braucht."
    )
else:
    worst = max(("complete", "inpair"), key=lambda m: abs(many[m]["bias"]))
    st.error(
        f"**Je Standort zugeteilt, schätzt der Vergleich {C.fmt_num(many['complete']['mean'], 2)} statt "
        f"{C.fmt_num(tr['mean'], 2)} Punkte, im Nachbarpaar {C.fmt_num(many['inpair']['mean'], 2)}** "
        f"({C.fmt_num(100 * many[worst]['rel_bias'], 0)} % im ungünstigeren Fall); in {C.fmt_pct(many['inpair']['wrong_sign'])} "
        f"der Netze zeigt der Vergleich im Nachbarpaar keine oder eine negative Wirkung. Der Cluster nach Nachbarpaaren liegt "
        f"bei {C.fmt_num(many['cluster']['mean'], 2)}, der Switchback bei {C.fmt_num(many['switchback']['mean'], 2)}."
        + (" **Mit der lokalen Kennzahl kippt die Richtung** (alle Verfahren zu hoch): der Nenner enthält fremde Nachfrage."
           if metric == "lokal" else "")
    )

# --- q-Sweep --------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Je mehr Ausweichen, desto mehr zählt das Design")
qs = [round(0.1 * i, 1) for i in range(11)]
qcells = [_cell("q", metric, q=x) for x in qs]
series = {m: [c["rows"][m]["mean"] for c in qcells] for m in C.METHODS}
st.plotly_chart(
    build_q_sweep(qs, series, [c["rows"]["_truth"]["mean"] for c in qcells],
                  "Mittlere Schätzung je Verfahren (L = 14, k = 2, Kennzahl: " + ("Herkunft" if metric == "herkunft" else "lokal") + ")"),
    width="stretch", key=f"qsweep_{metric}")
st.plotly_chart(
    build_rmse_sweep(qs, {m: [c["rows"][m]["rmse"] for c in qcells] for m in C.METHODS}, "Typischer Fehler in einem Netz (RMSE)"),
    width="stretch", key=f"qrmse_{metric}")
c0, c2, c10 = _cell("q", metric, q=0.0), _cell("q", metric, q=0.2), _cell("q", metric, q=1.0)
st.info(
    f"Bei q = 0 sind alle Verfahren richtig, und der Vergleich je Standort hat den kleinsten Fehler "
    f"({C.fmt_num(c0['rows']['complete']['rmse'], 2)} gegen {C.fmt_num(c0['rows']['switchback']['rmse'], 2)} beim Switchback). "
    f"Bei q = 0,2 ist er noch gleichauf ({C.fmt_num(c2['rows']['complete']['rmse'], 2)} gegen "
    f"{C.fmt_num(c2['rows']['cluster']['rmse'], 2)} beim Cluster). Bei q = 1 liegt der Fehler beim Vergleich je Standort bei "
    f"{C.fmt_num(c10['rows']['complete']['rmse'], 2)}, beim Cluster bei {C.fmt_num(c10['rows']['cluster']['rmse'], 2)}, beim "
    f"Switchback bei {C.fmt_num(c10['rows']['switchback']['rmse'], 2)}."
)

# --- Switchback: Blocklänge ---------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⏱️ Switchback: Blocklänge und Einschwingen")
lengths = [1, 2, 3, 4, 5, 7, 10, 14, 21, 28]
lc0 = [_cell("L", "herkunft", q=C.DEFAULT_Q, length=x, k=0)["rows"]["switchback"] for x in lengths]
lc2 = [_cell("L", "herkunft", q=C.DEFAULT_Q, length=x, k=2)["rows"]["switchback"] for x in lengths]
st.plotly_chart(build_block_sweep(lengths, [r["bias"] for r in lc0], [r["bias"] for r in lc2], [r["rmse"] for r in lc0],
                                  [r["rmse"] for r in lc2], 2), width="stretch", key="block_sweep")
by_len = dict(zip(lengths, lc2))
by_len0 = dict(zip(lengths, lc0))
st.info(
    f"Ohne weggelassene Tage hängt der Servicegrad eines Tages an der Regel von vorgestern: bei Blöcken von 1 Tag schätzt "
    f"der Switchback {C.fmt_num(by_len0[1]['mean'], 2)} statt {C.fmt_num(by_len0[1]['truth'], 2)} Punkten, bei 14 Tagen "
    f"{C.fmt_num(by_len0[14]['bias'], 2)} Punkte Verzerrung. Mit 2 weggelassenen Tagen (= Wiederbeschaffungszeit) verschwindet "
    f"die Nachwirkung. **Wochentagsfalle:** ist L ein Vielfaches von 7, fallen immer dieselben Wochentage weg; bei L = 7 "
    f"entsteht dadurch {C.fmt_num(by_len[7]['day_selection'], 2)} Punkte Verzerrung, bei L = 10 praktisch keine "
    f"({C.fmt_num(by_len[10]['day_selection'], 2)})."
)

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Interferenz bleibt innerhalb des Clusters** | Weichen Kunden auch zu Standorten außerhalb des Paars aus, bleibt ein Rest-Fehler; größere Cluster kosten Standorte und Teststärke. | (dieses Stück) |
| **Man kennt die Nachbarschaft** | Der Cluster-Ansatz braucht die Paare im Voraus; der Switchback braucht sie nicht, dafür die Wiederbeschaffungszeit. | (dieses Stück) |
| **Die Kennzahl misst das System** | Eine Kennzahl je Standort mit fremder Nachfrage im Nenner verzerrt auch den Cluster, der Herkunftsservicegrad nicht. | (dieses Stück) |
| **Die Wirkung ist nach dem Einschwingen konstant** | Wirkt die Regel erst über Wochen (Gewöhnung, Bestandsaufbau), misst ein kurzer Block sie nicht. | Differenz-von-Differenzen und synthetische Kontrolle |
| **Es gibt viele Einheiten** | Mit wenigen Standorten hilft keine Zuteilung; die Wirkung muss aus Vorher-Nachher und Kontrollen geschlossen werden. | Differenz-von-Differenzen und synthetische Kontrolle |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen oder Varianten vervielfachen die Tests. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
"""
)
st.caption(
    "Die Zahlen stammen aus dem erfundenen Vehikel dieses Stücks (20 Standorte, Überlauf nur zum festen Nachbarn, "
    "Zentrallager unbegrenzt); wie stark Kunden in einem Betrieb ausweichen, ist eine Annahme und kein Messwert."
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Wirkung auf das System.** Mit allen Standorten auf Regel $r$ ist der Servicegrad $F_r = \sum_i \mathrm{bedient}_i /
\sum_i D_i$. Die Wahrheit ist $\Delta = F_B - F_A$ bei gleicher Nachfrage.

**Warum der Vergleich je Standort daneben liegt.** Ein Standort mit Regel B bekommt Überlauf von seinem Nachbarn (mit Regel A,
der öfter Fehlmengen hat) und hat dadurch weniger Bestand für die eigene Kundschaft; der Nachbar mit A wird entlastet, weil
sein Überlauf bedient wird. Der Vergleich beider Gruppen $\hat\Delta_{\text{Standort}} = F_B^{(B)} - F_A^{(A)}$ ist in der
gemischten Welt kleiner als $\Delta$, solange die Kennzahl die Herkunft zählt. Mit der lokalen Kennzahl
$\frac{\mathrm{bedient}_i}{D_i + \mathrm{eingehend}_i}$ ändert sich der Nenner mit, und die Richtung kippt.

**Cluster.** Wird das ganze Paar gemeinsam zugeteilt, bleibt jede Interferenz im Paar; in der Herkunftskennzahl gilt
$\hat\Delta_{\text{Cluster}} \approx \Delta$. Preis: nur 10 statt 20 unabhängige Einheiten.

**Switchback.** Alle Standorte wechseln in Blöcken von $L$ Tagen gemeinsam. Weil die Bestellung vom Vortag noch mit der alten Regel
getroffen wurde, wirkt am Tag $r$ eines Blocks die neue Regel erst voll, wenn $r \ge$ Wiederbeschaffungszeit. Ohne Weglassen
fehlt der Schätzung ein Teil der Wirkung, der mit der Blocklänge sinkt (gemessen bei $q = 0{,}3$: 37 % der Wirkung bei
$L = 7$, 19 % bei $L = 14$, 10 % bei $L = 28$). Der Schätzer vergleicht den mittleren Block-Servicegrad der B-Blöcke mit dem der
A-Blöcke.

Implementiert in `swb_world.py` (Netz, Tagesablauf), `swb_evaluation.py` (Zuteilungen, Schätzer, Zusammenfassungen) und
`generate_precomputed.py` (Sweeps).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html)."
)
