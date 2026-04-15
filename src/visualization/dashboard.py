"""Plotly dashboard for downloaded document metrics."""

import statistics
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

from src.models.schemas import DocumentMetrics, MetricsReport

console = Console()

_ACCENT = "#636EFA"
_GREEN = "#00CC96"
_ORANGE = "#EF553B"


def _histogram(
    values: List[int],
    title: str,
    x_label: str,
    color: str,
) -> go.Histogram:
    return go.Histogram(
        x=values,
        name=title,
        marker_color=color,
        opacity=0.85,
        xbins={"size": 1} if max(values, default=1) <= 50 else {},
        hovertemplate=f"{x_label}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    )


def _stats_table(docs: List[DocumentMetrics], tokenizer: str) -> go.Table:
    """Summary statistics table shown alongside the token histogram."""

    def _stats(values: List[int], label: str) -> List[str]:
        if not values:
            return [label, "-", "-", "-", "-", "-"]
        return [
            label,
            str(min(values)),
            str(max(values)),
            f"{statistics.mean(values):.1f}",
            f"{statistics.median(values):.1f}",
            str(len(values)),
        ]

    token_vals = [d.token_count for d in docs if d.token_count > 0]
    ref_vals = [d.reference_count for d in docs]
    img_vals = [d.image_count for d in docs]

    headers = ["Metric", "Min", "Max", "Mean", "Median", "Docs"]
    rows = [
        _stats(token_vals, "Tokens (excl. 0)"),
        _stats(ref_vals, "References"),
        _stats(img_vals, "Images"),
    ]

    return go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#2d2d2d",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[[r[i] for r in rows] for i in range(len(headers))],
            fill_color=[["#1a1a2e", "#16213e"] * 3],
            font=dict(color="white", size=11),
            align="left",
        ),
        name=f"Tokenizer: {tokenizer}",
    )


def render(report: MetricsReport, open_browser: bool = True) -> go.Figure:
    """Build and display the metrics dashboard.

    Layout:
        [Token histogram        |  Summary stats table ]
        [References histogram   |  Images histogram    ]
    """
    docs = report.documents
    token_vals = [d.token_count for d in docs if d.token_count > 0]
    ref_vals = [d.reference_count for d in docs]
    img_vals = [d.image_count for d in docs]

    if not token_vals:
        console.print("[red]No documents with token data found. Run 'ingest' first.[/red]")
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Token Distribution ({len(token_vals)} docs, 0 excluded)",
            "Summary Statistics",
            f"References per Document ({len(ref_vals)} docs)",
            f"Images per Document ({len(img_vals)} docs)",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "table"}],
            [{"type": "histogram"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(_histogram(token_vals, "Tokens", "Tokens", _ACCENT), row=1, col=1)
    fig.add_trace(_stats_table(docs, report.tokenizer), row=1, col=2)
    fig.add_trace(_histogram(ref_vals, "References", "References", _GREEN), row=2, col=1)
    fig.add_trace(_histogram(img_vals, "Images", "Images", _ORANGE), row=2, col=2)

    # Vertical stat lines on token histogram (row=1, col=1 → xaxis="x", yaxis="y")
    token_mean = statistics.mean(token_vals)
    token_median = statistics.median(token_vals)
    for value, label, dash in [
        (min(token_vals), f"Min: {min(token_vals)}", "dot"),
        (max(token_vals), f"Max: {max(token_vals)}", "dot"),
        (token_mean, f"Mean: {token_mean:.0f}", "dash"),
        (token_median, f"Median: {token_median:.0f}", "dashdot"),
    ]:
        fig.add_shape(
            type="line",
            x0=value, x1=value,
            y0=0, y1=1,
            xref="x", yref="y domain",
            line=dict(color="white", dash=dash, width=1),
            opacity=0.6,
        )
        fig.add_annotation(
            x=value,
            y=1.02,
            xref="x", yref="y domain",
            text=label,
            showarrow=False,
            font=dict(size=9, color="white"),
            textangle=-45,
            xanchor="left",
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Document Metrics Dashboard</b>"
                f"<br><sup>{report.total_documents} documents · {report.tokenizer}</sup>"
            ),
            x=0.5,
            font_size=18,
        ),
        template="plotly_dark",
        showlegend=False,
        height=750,
        margin=dict(t=100, b=40, l=50, r=50),
    )

    fig.update_xaxes(title_text="Token count", row=1, col=1)
    fig.update_yaxes(title_text="Documents", row=1, col=1)
    fig.update_xaxes(title_text="Reference count", row=2, col=1)
    fig.update_yaxes(title_text="Documents", row=2, col=1)
    fig.update_xaxes(title_text="Image count", row=2, col=2)
    fig.update_yaxes(title_text="Documents", row=2, col=2)

    if open_browser:
        fig.show()

    return fig
