# Interferenz und Switchback: wenn Standorte einander beeinflussen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-switchback-demo.streamlit.app/)** -->

Interaktive Demo zu **Interferenz zwischen Standorten und Zuteilung über die Zeit**. **Zehntes Stück der Konzepte-Linie
„Experimente und Wirkungsmessung“** im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research
und Machine Learning). Das Vehikel ist die Bestandsregel der Vorgänger (tägliche Nachbestellung, Servicegrad), aber nicht mehr
ein Standort allein, sondern ein **Netz aus 20 Standorten** mit Nachfrageüberlauf zwischen Nachbarn.

## Kernfrage

Wenn Kunden einer Filiale mit Fehlmenge zur Nachbarfiliale ausweichen: was misst der Vergleich zwischen Standorten mit Regel A
und B noch, und was leisten die zwei üblichen Auswege, Nachbarn gemeinsam zuteilen (Cluster) und alle Standorte über die Zeit
wechseln lassen (Switchback)?

## Modell und Methodik

- **Netz** (`swb_world.py`): 20 Standorte in 10 Nachbarpaaren, Standortgrößen reihum (Nachfrage 12 bis 30 je Tag),
  Wochentagsfaktor, negativ-binomiale Nachfrage (Gamma-Poisson), tägliche Bestellung bis zum Niveau
  `S = round(2 · mu + z · sd)`, Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt. 28 Einschwingtage, 112 Messtage.
- **Überlauf:** der Anteil q der Fehlmenge versucht es am selben Tag beim Nachbarn; bedient wird, was dort nach dessen eigener
  Nachfrage im Bestand ist, der Rest ist verloren, die übrige Fehlmenge (1 − q) bleibt Rückstand. Erhaltung je Herkunftsstandort:
  Nachfrage = eigen bedient + beim Nachbarn bedient + verloren + Rückstand.
- **Wahrheit:** alle Standorte auf B (z = 1,4) gegen alle auf A (z = 1,0), gleiche Nachfrage: Kundenservicegrad des Systems.
- **Vier Zuteilungen** (`swb_evaluation.py`): je Standort zufällig (die Hälfte der Standorte B), im Nachbarpaar (in jedem Paar
  bekommt einer B), Cluster (Paare gemeinsam), Switchback (alle Standorte in Blöcken von L Tagen, die ersten k Tage jedes Blocks
  nicht ausgewertet).
- **Zwei Kennzahlen:** *Herkunft* (Kunde zählt dort, wo er bedient wird) und *lokal* (am Standort bedient / eigene Nachfrage plus
  eingehender Überlauf).
- **Simulationen:** 2 000 Netze je Zelle in `precomputed_sweeps.json`, 1 000 live in der App; die Nachfrage kommt aus SplitMix64,
  die negativ-binomiale Inversion über eine Verteilungstabelle je Standort und Wochentag.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Verzerrung = mittlere Abweichung der Schätzung von der Wirkung des jeweiligen
Netzes in Prozentpunkten Servicegrad; Kennzahl Herkunft, wenn nicht anders genannt; Switchback mit L = 14 und k = 2.

| Frage | Befund |
|---|---|
| Wie groß ist die Wirkung auf das System? | Sie **schrumpft mit dem Ausweichen**: **3,54** Punkte bei q = 0, **2,87** bei 0,2, **2,59** bei 0,3, **2,14** bei 0,5, **1,40** bei q = 1 (A-Standorte profitieren vom B-Nachbarn). |
| Wie verändert der Nachbar die Standorte? | Bei q = 0,3 liegen B-Standorte um **−0,16** Punkte unter der Welt „alle B“, A-Standorte um **+0,26** über der Welt „alle A“ (q = 1: **−0,44** und **+0,71**). |
| Ohne Ausweichen? | Alle vier Verfahren im Mittel richtig (Verzerrung unter 0,1). Typischer Fehler (RMSE) je Standort **1,13**, Nachbarpaar 1,10, Cluster 1,10, Switchback **1,19**. |
| Bei q = 0,3? | Verzerrung je Standort **−0,22** (−9 %), im Nachbarpaar **−0,43** (−17 %), Cluster **0,00**, Switchback **−0,08** (−3 %); RMSE **0,81 / 0,84 / 0,84 / 0,86**. |
| Bei q = 1? | Je Standort **−0,59** (−42 %), im Nachbarpaar **−1,15** (−82 %), Cluster **−0,01**, Switchback **−0,10** (−7 %); RMSE **0,78 / 1,22 / 0,54 / 0,57**. |
| Wie oft sieht es aus, als wäre B nutzlos? | Bei q = 1 zeigt der Vergleich im Nachbarpaar in **26,8 %** der Netze eine Wirkung ≤ 0, je Standort **4,8 %**, Cluster und Switchback **0,8 %**. |
| Und mit der lokalen Kennzahl? | **Alle Verfahren zu hoch.** Bei q = 0,3: **+0,62 / +0,41 / +0,83 / +0,75** Punkte; bei q = 1 beim Cluster **+2,31** (**+165 %**), je Standort **+1,47** (+105 %). Der Cluster ist hier am schlechtesten, weil er die Nachbarn sauber trennt und der Nenner fremde Nachfrage trotzdem mitzählt. |
| Switchback ohne weggelassene Tage? | Bei Blöcken von 1 Tag **−2,61** Punkte (die Wirkung verschwindet, **50,1 %** der Netze ≤ 0); bei q = 0,3 gehen **37 %** der Wirkung bei L = 7, **19 %** bei L = 14 und **10 %** bei L = 28 verloren (−0,95 / −0,49 / −0,26). |
| Mit 2 weggelassenen Tagen? | Die Nachwirkung verschwindet (unter 0,06 Punkte). Es bleibt die **Wochentagsfalle**: bei L = 7 **−0,27**, bei L = 14 **−0,11**, bei L = 28 **−0,05**, bei L = 10 **+0,01** Punkte, weil bei Vielfachen von 7 immer dieselben Wochentage wegfallen. |
| Welche Blocklänge hat den kleinsten Fehler? | Bei k = 2: L = 3 **1,28**, L = 7 **0,87**, L = 10 **0,94**, L = 14 **0,86**, L = 28 **0,86**. |
| Default-Seed 458 (q = 0,3) | Wahre Wirkung **2,42**; je Standort **2,33**, im Nachbarpaar **1,91**, Cluster **2,41**, Switchback **2,12**. |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Mechanismus ist nicht das Zentrallager.** Die Annahme „ein höheres z zieht mehr aus dem Zentrallager“ stimmt für die
  Order-up-to-Regel nicht: die Bestellung ist Nachfrage von gestern plus Rückstand, unabhängig von z, und die Lieferungen sind in
  allen Welten gleich (Vorab-Messreihe Phase 0: Interferenz exakt 0). Echte Interferenz entsteht erst durch den **Nachfrageüberlauf
  zwischen Nachbarn**, den dieses Stück modelliert; der Ausweichanteil q ist eine Modellannahme, kein Betriebsmesswert.
- **Der Standort-Vergleich ist bei kleinem q gut genug.** Bis q = 0,2 ist sein Fehler gleichauf oder kleiner als bei Cluster
  und Switchback (0,89 gegen 0,93 und 0,96 bei q = 0,2): Der Schutz kostet Teststärke und lohnt erst bei starkem Ausweichen.
- **Die Richtung hängt an der Kennzahl.** Der Herkunfts-Servicegrad unterschätzt, der lokale überschätzt; wer ein Dashboard
  je Standort liest, bekommt eine Wirkung mit anderem Vorzeichen des Fehlers als der Systemeffekt.
- **Der Cluster rettet die Herkunftskennzahl, nicht die lokale.** Selbst bei sauberer Trennung bleibt der Nenner
  (eingehender Überlauf) eine Größe, die an der Regel hängt.
- **Switchback und Cluster sind praktisch gleich gut**, aber mit verschiedenen Voraussetzungen: der Switchback braucht die
  Wiederbeschaffungszeit (k ≥ 2) und eine passende Blocklänge, der Cluster die Nachbarschaft und die Herkunftskennzahl.
- **Keine Teststärke- oder Konfidenzrechnung:** Das Stück zeigt Verzerrung, Streuung und Fehler; Intervalle und Tests je
  Verfahren sind nicht gebaut (siehe Vorgänger für den Welch-Test).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Interferenz bleibt innerhalb des Clusters** | Weichen Kunden auch zu Standorten außerhalb des Paars aus, bleibt ein Rest-Fehler; größere Cluster kosten Standorte und Teststärke. | (dieses Stück) |
| **Man kennt die Nachbarschaft** | Der Cluster-Ansatz braucht die Paare im Voraus; der Switchback braucht sie nicht, dafür die Wiederbeschaffungszeit. | (dieses Stück) |
| **Die Kennzahl misst das System** | Eine Kennzahl je Standort mit fremder Nachfrage im Nenner verzerrt auch den Cluster. | (dieses Stück) |
| **Die Wirkung ist nach dem Einschwingen konstant** | Wirkt die Regel erst über Wochen, misst ein kurzer Block sie nicht. | Differenz-von-Differenzen und synthetische Kontrolle |
| **Es gibt viele Einheiten** | Mit wenigen Standorten hilft keine Zuteilung. | Differenz-von-Differenzen und synthetische Kontrolle |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen oder Varianten vervielfachen die Tests. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |

Die Zahlen gelten für dieses erfundene Netz (feste Nachbarpaare, Überlauf nur am selben Tag, Zentrallager unbegrenzt).

## Bezug zu OR

Jede Filialen- oder Depotstruktur hat Nachbarn, die einander Kundschaft abnehmen. Ein Pilot, der Regeln je Standort verteilt,
misst dann die Wirkung auf einen Standort im Verbund, nicht die auf das Netz. Bei Netzentscheidungen zählt die Wirkung auf das
System; deshalb muss Zuteilung und Kennzahl zur Frage passen.

## Verwandte Demos im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo): Vehikel und Welch-Test für einen Standort.
- [ratio-metric-demo](https://github.com/sebastian-hanisch/ratio-metric-demo): die Einheit der Auswertung (Cluster = Standort).
- [bandit-vs-ab-demo](https://github.com/sebastian-hanisch/bandit-vs-ab-demo): Zuteilung, die von den Ergebnissen abhängt.
- [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo): die Base-Stock-Regel des Vehikels.

## Verifikation

- **Netz** (`tests/test_world.py`): von Hand an einer Mini-Instanz mit zwei Standorten (Bestand 33/44, Nachfrage 40/10, q = 0,5:
  Überlauf 3,5, neuer Bestand −3,5 und 30,5, Erhaltung 40 = 33 + 3,5 + 0 + 3,5); Überlauf, den der Nachbar nicht bedienen kann,
  geht verloren; Erhaltung und Symmetrie auf 20 Standorten; Nachfragemomente je Standort und Wochentag; Reproduzierbarkeit.
- **Orakel** (`tests/test_oracle_switchback.py`): Nachfrage-Inversion gegen `scipy.stats.nbinom.ppf` mit denselben Gleichverteilten;
  Servicegrad ohne Überlauf gegen die exakte Formel der Order-up-to-Regel (unabhängig erzeugte Nachfrage); ein geändertes
  Bestellniveau wirkt frühestens nach der Wiederbeschaffungszeit; Summe als Überlauf bedient = Summe beim Nachbarn bedient.
- **Zuteilung und Schätzer** (`tests/test_evaluation.py`): Hälfte B je Design, genau ein B je Paar, Paare gemeinsam;
  Kennzahlen und Switchback-Schätzer von Hand; ohne Überlauf sind alle Verfahren unverzerrt; mit Überlauf unterschätzt das
  Nachbarpaar-Design, die lokale Kennzahl kippt die Richtung; ohne Weglassen ist der Switchback verzerrt.
- **Befunde:** jede Zahl dieses READMEs in `tests/test_claims.py`; zwei Zellen werden live nachgerechnet.
- **Presets, Oberfläche, Footer:** vollständig, Permalink eingerastet, AppTest-Rauchtest, Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: ein Netz, tausend Netze, Sweeps über q und Blocklänge, Grenzen |
| `swb_constants.py` | Vehikel-Konstanten, Regler-Grenzen, Verfahren, Presets |
| `swb_rng.py` | SplitMix64 |
| `swb_world.py` | Nachfrage (Gamma-Poisson per Inversion), Tagesablauf des Netzes mit Überlauf |
| `swb_evaluation.py` | Zuteilungen, Kennzahlen, Switchback-Schätzer, Zusammenfassungen, ein Netz |
| `swb_presets.py`, `swb_visualization.py` | Permalink, Presets; Plotly-Abbildungen (Achsen gesperrt) |
| `generate_precomputed.py`, `precomputed_sweeps.json` | Sweeps (42 Zellen à 2 000 Netze) |
| `tests/` | Netz, Zuteilung, Orakel, Presets, Befunde, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Konfidenzintervalle und Tests** je Verfahren: nicht gebaut.
- **Zentrallager mit Knappheit** als Interferenzmechanismus: in der Vorab-Messreihe ohne Wirkung bei der verwendeten
  Order-up-to-Regel; deshalb nicht Teil dieses Stücks.
- **Größere Cluster** (mehr als zwei Standorte) und **zufällige Nachbarschaft**: nicht gebaut.
- **Wochenschocks** und andere gemeinsame Störungen: nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (Verteilungsfunktionen).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
