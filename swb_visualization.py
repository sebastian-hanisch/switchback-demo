"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Scrollen auf dem Handy nicht zoomt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import swb_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=360, legend=dict(orientation="h", y=-0.25))


def _finish(fig, title, xtitle=None, ytitle=None, **extra):
    fig.update_layout(title=dict(text=title, x=0), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), **{**_LAYOUT, **extra})
    return fig


def build_assignment(masks):
    """Welche Standorte bekommen Regel B? Je Verfahren eine Zeile über die 20 Standorte (Nachbarpaare sind benachbart)."""
    labels = [C.METHOD_SHORT[m] for m in ("complete", "inpair", "cluster")]
    z = np.array([masks[m].astype(int) for m in ("complete", "inpair", "cluster")])
    n = z.shape[1]
    text = [["B" if v else "A" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(1, n + 1)), y=labels, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#c9d7e8"], [1, "#2F6B65"]], showscale=False, xgap=2, ygap=2,
                               hovertemplate="Standort %{x}: Regel %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    fig.update_xaxes(dtick=1, fixedrange=True, title="Standort (Nachbarpaare: 1-2, 3-4, ...)")
    fig.update_layout(title=dict(text="Wer bekommt Regel B? (ein Netz)", x=0), height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_timeline(arm_b, length, k, days):
    """Der Switchback-Plan über die Tage: Regel B / A, weggelassene Tage grau, nicht genutzte Tage hell."""
    nb = len(arm_b)
    used = nb * length
    z = np.full(days, -1.0)
    for b in range(nb):
        for d in range(length):
            z[b * length + d] = (1.0 if arm_b[b] else 0.0) if d >= k else 0.5
    fig = go.Figure(go.Heatmap(z=[z], x=list(range(1, days + 1)), y=["Switchback"], zmin=-1, zmax=1,
                               colorscale=[[0, "#eeeeee"], [0.25, "#eeeeee"], [0.5, "#c9d7e8"], [0.75, "#bbbbbb"], [1, "#2F6B65"]],
                               showscale=False, xgap=0,
                               hovertemplate="Tag %{x}<extra></extra>"))
    fig.update_xaxes(title=f"Tag (dunkelgrün: B, blau: A, grau: weggelassen; {used} von {days} Tagen im Plan)", fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(title=dict(text="Switchback-Plan über die Tage", x=0), height=170, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_estimates(est, truth):
    """Geschätzte Wirkung je Verfahren in einem Netz (Prozentpunkte) gegen die wahre Wirkung auf das System."""
    names = [m for m in C.METHODS if m in est]
    fig = go.Figure(go.Bar(x=[C.METHOD_SHORT[m] for m in names], y=[100 * est[m] for m in names],
                           marker_color=[C.METHOD_COLORS[m] for m in names],
                           text=[f"{100 * est[m]:+.2f}".replace(".", ",") for m in names], textposition="outside"))
    fig.add_hline(y=100 * truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="wahre Wirkung auf das System",
                  annotation_position="top left")
    return _finish(fig, "Geschätzte Wirkung von Regel B in diesem Netz", None, "Prozentpunkte Servicegrad")


def build_spread(rows):
    """Mittel ± Standardabweichung der Schätzung je Verfahren über viele Netze, mit der wahren Wirkung (Linie)."""
    fig = go.Figure()
    for m in C.METHODS:
        r = rows[m]
        fig.add_trace(go.Scatter(x=[C.METHOD_SHORT[m]], y=[r["mean"]], mode="markers", showlegend=False,
                                 marker=dict(color=C.METHOD_COLORS[m], size=12),
                                 error_y=dict(type="data", array=[r["sd"]], thickness=3, width=9, color=C.METHOD_COLORS[m])))
    fig.add_hline(y=rows["_truth"]["mean"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="wahre Wirkung",
                  annotation_position="top left")
    return _finish(fig, "Geschätzte Wirkung: Mittel ± Standardabweichung über die Netze", None, "Prozentpunkte Servicegrad")


def build_q_sweep(q, series, truth, title):
    """Mittlere Schätzung je Verfahren gegen den Ausweichanteil, mit der wahren Wirkung."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=truth, mode="lines", name="wahre Wirkung auf das System",
                             line=dict(color=C.COLOR_TRUTH, width=4, dash="dash")))
    for m, y in series.items():
        fig.add_trace(go.Scatter(x=q, y=y, mode="lines+markers", name=C.METHOD_SHORT[m],
                                 line=dict(color=C.METHOD_COLORS[m], width=2.5)))
    return _finish(fig, title, "Anteil der Fehlmenge, der zum Nachbarn ausweicht", "Prozentpunkte Servicegrad")


def build_rmse_sweep(q, series, title):
    fig = go.Figure()
    for m, y in series.items():
        fig.add_trace(go.Scatter(x=q, y=y, mode="lines+markers", name=C.METHOD_SHORT[m],
                                 line=dict(color=C.METHOD_COLORS[m], width=2.5)))
    fig.update_yaxes(rangemode="tozero")
    return _finish(fig, title, "Anteil der Fehlmenge, der zum Nachbarn ausweicht", "Prozentpunkte (RMSE)")


def build_block_sweep(lengths, bias_k0, bias_k2, rmse_k0, rmse_k2, k):
    """Switchback: Verzerrung und RMSE gegen die Blocklänge, ohne und mit Einschwingen."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Verzerrung", "RMSE"))
    for col, (y0, y2) in enumerate(((bias_k0, bias_k2), (rmse_k0, rmse_k2)), start=1):
        fig.add_trace(go.Scatter(x=lengths, y=y0, mode="lines+markers", name="kein Tag weggelassen", legendgroup="k0",
                                 showlegend=(col == 1), line=dict(color=C.COLOR_MISS, width=2.5)), row=1, col=col)
        fig.add_trace(go.Scatter(x=lengths, y=y2, mode="lines+markers", name=f"{k} Tage weggelassen", legendgroup="k2",
                                 showlegend=(col == 1), line=dict(color=C.METHOD_COLORS["switchback"], width=2.5)), row=1, col=col)
    fig.add_hline(y=0, line=dict(color="gray", width=1), row=1, col=1)
    fig.update_xaxes(title_text="Blocklänge in Tagen", fixedrange=True, type="log", tickvals=[1, 2, 3, 5, 7, 10, 14, 21, 28],
                     ticktext=["1", "2", "3", "5", "7", "10", "14", "21", "28"])
    fig.update_yaxes(title_text="Prozentpunkte", fixedrange=True)
    fig.update_layout(title=dict(text="Switchback: Blocklänge und Einschwingen", x=0), height=400,
                      margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", y=-0.38))
    return fig
