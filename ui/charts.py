"""Shared Plotly theme for OSMOSE charts."""

import plotly.graph_objects as go
import plotly.io as pio

# Register a custom OSMOSE template based on plotly_dark
_OSMOSE_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(15, 25, 35, 0)",
        plot_bgcolor="rgba(15, 25, 35, 0.6)",
        font=dict(
            family="Plus Jakarta Sans, -apple-system, sans-serif",
            color="#e2e8f0",
            size=12,
        ),
        title=dict(
            font=dict(size=15, color="#e2e8f0"),
            x=0.02,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            linecolor="rgba(255, 255, 255, 0.1)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(size=11),
        ),
        colorway=[
            "#e8a838",  # amber
            "#38c9b1",  # teal
            "#3498db",  # blue
            "#e74c3c",  # red
            "#9b59b6",  # purple
            "#2ecc71",  # green
            "#f39c12",  # orange
            "#1abc9c",  # turquoise
        ],
        legend=dict(
            bgcolor="rgba(0, 0, 0, 0.2)",
            bordercolor="rgba(255, 255, 255, 0.06)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=40, b=40),
    ),
)

pio.templates["osmose"] = _OSMOSE_TEMPLATE

# Template name for use in update_layout calls
PLOTLY_TEMPLATE = "osmose"
