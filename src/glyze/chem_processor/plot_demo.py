import plotly.graph_objects as go
from glyze.glyceride_mix import GlycerideMix


def plot(self, initial_mix: GlycerideMix, final_moles: dict, total_final: float):
    """
    Plot a grouped bar chart of initial vs final moles for each component.

    Parameters
    ----------
    initial_mix  : GlycerideMix
        The mix before deodorization. Iterated with .items() -> (component, moles).
    final_moles  : dict
        Mole-fraction dict returned by single_pass(), keyed by component objects.
    total_final  : float
        Total moles after deodorization (used to convert fractions -> moles).
    """

    # Build (name, initial_moles, final_moles) tuples, sorted by initial moles descending
    rows = []
    initial_lookup = dict(initial_mix.items())

    for component, init_mol in initial_lookup.items():
        fin_mol = final_moles.get(component, 0.0) * total_final
        rows.append((str(component), init_mol, fin_mol))

    rows.sort(key=lambda r: r[1], reverse=True)

    names = [r[0] for r in rows]
    y_initial = [r[1] for r in rows]
    y_final = [r[2] for r in rows]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Initial",
            x=names,
            y=y_initial,
            marker_color="#1a3a5c",
            marker_line=dict(color="#0d1f33", width=1),
            hovertemplate="<b>%{x}</b><br>Initial: %{y:.5f} mol<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Final",
            x=names,
            y=y_final,
            marker_color="#c0392b",
            marker_line=dict(color="#7a1f1a", width=1),
            hovertemplate="<b>%{x}</b><br>Final: %{y:.5f} mol<extra></extra>",
        )
    )

    # Annotate % removed above each bar group
    for name, init, fin in rows:
        if init > 0:
            pct = (init - fin) / init * 100
            fig.add_annotation(
                x=name,
                y=max(init, fin),
                text=f"−{pct:.1f}%",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="#666", family="monospace"),
            )

    total_initial = sum(y_initial)
    total_removed = total_initial - total_final
    efficiency = total_removed / total_initial * 100 if total_initial else 0

    fig.update_layout(
        title=dict(
            text="Deodorizer — Initial vs Final Mole Composition",
            font=dict(size=20, family="Arial Black, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        xaxis=dict(
            title="Component",
            tickangle=-35,
            tickfont=dict(size=11, family="monospace"),
            gridcolor="#e8e3da",
            linecolor="#d8d3c8",
        ),
        yaxis=dict(
            title="Moles [mol]",
            gridcolor="#e8e3da",
            linecolor="#d8d3c8",
            tickfont=dict(family="monospace"),
        ),
        plot_bgcolor="#fdfaf5",
        paper_bgcolor="#f5f2eb",
        font=dict(family="Arial, sans-serif", color="#1a1a1a"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=100, b=140, l=70, r=40),
        annotations=[
            # per-bar % annotations are added above; this is the summary footer
            dict(
                text=(
                    f"Total initial: <b>{total_initial:.4f} mol</b> │ "
                    f"Total final: <b>{total_final:.4f} mol</b> │ "
                    f"Removed: <b>{total_removed:.4f} mol ({efficiency:.1f}%)</b>"
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.28,
                showarrow=False,
                font=dict(size=11, family="monospace", color="#4a4540"),
                align="center",
            )
        ],
    )

    fig.show()
