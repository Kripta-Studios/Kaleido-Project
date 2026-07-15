from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def process_report_html(payload: dict[str, Any], *, title: str, watermark: str) -> str:
    discovery = payload["discovery"]
    activities = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{count}</td></tr>"
        for name, count in list(discovery["activities"].items())[:20]
    )
    edges = "".join(
        "<tr>"
        f"<td>{html.escape(str(edge['source']))}</td>"
        f"<td>{html.escape(str(edge['target']))}</td>"
        f"<td>{edge['count']}</td>"
        "</tr>"
        for edge in discovery["directly_follows"][:30]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:Inter,Segoe UI,sans-serif;background:#f4f7f8;color:#17262d;margin:0}}
main{{max-width:1080px;margin:auto;padding:40px 24px}}
.watermark{{background:#fff2ce;border:1px solid #e0b24b;padding:12px 16px;border-radius:10px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}
.card{{background:white;border:1px solid #dbe4e7;border-radius:14px;padding:20px}}
.value{{font-size:30px;font-weight:750;color:#063b49}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #e7edef}}
th{{color:#52646b;font-size:12px;text-transform:uppercase}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<p class="watermark"><strong>{html.escape(watermark)}</strong> — public/synthetic data is
pipeline evidence only, not Kaleido business evidence.</p>
<h1>{html.escape(title)}</h1>
<div class="grid">
<div class="card"><div class="value">{discovery["events"]:,}</div>events</div>
<div class="card"><div class="value">{discovery["operations"]:,}</div>object traces</div>
<div class="card"><div class="value">{len(discovery["activities"]):,}</div>activities</div>
</div>
<h2>Top activities</h2><table><tr><th>Activity</th><th>Events</th></tr>{activities}</table>
<h2>Directly-follows edges</h2>
<table><tr><th>Source</th><th>Target</th><th>Count</th></tr>{edges}</table>
<h2>What this does not prove</h2>
<p>No accuracy, early warning, causal effect, ROI, realized saving or production deployment
claim can be made from this report.</p>
</main></body></html>"""


def write_evidence_pdf(path: Path, title: str, metrics: dict[str, Any]) -> None:
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42
    )
    story: list[Any] = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "SMOKE_ONLY — public data validates the pipeline, not Kaleido value.",
            styles["Heading2"],
        ),
        Spacer(1, 12),
    ]
    rows = [["Field", "Value"]]
    for key in (
        "dataset_id",
        "source_file_sha256",
        "split_protocol",
        "number_of_seeds",
        "claim_state",
    ):
        rows.append([key, str(metrics.get(key))])
    selected = metrics.get("selected_model_test", {})
    rows.extend(
        [
            ["selected_model", str(metrics.get("model_selection", {}).get("selected_model"))],
            ["test_mae_minutes", str(selected.get("mae_minutes"))],
            ["p90_interval_coverage", str(selected.get("p90_interval_coverage"))],
            [
                "threshold_selection",
                str(metrics.get("threshold_selection", {}).get("selection_method")),
            ],
            [
                "test_influenced_choice",
                str(metrics.get("model_selection", {}).get("test_influenced_choice")),
            ],
        ]
    )
    table = Table(rows, colWidths=[165, 325])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#083b49")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d5d9")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f7f8")]),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 18),
            Paragraph("What this does not prove", styles["Heading2"]),
            Paragraph(
                "The dataset is not a Kaleido port export. The result does not establish "
                "Kaleido accuracy, early warning, ROI, savings, causal action value or "
                "production readiness.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value
