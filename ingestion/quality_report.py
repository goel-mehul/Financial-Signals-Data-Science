import pandas as pd
from pathlib import Path
from jinja2 import Template

REPORT_DIR = Path("data/quality_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>FinSignal Data Quality Report</title>
  <style>
    body { font-family: monospace; padding: 2rem; background: #0f0f0f; color: #e0e0e0; }
    h1 { color: #4fc3f7; }
    h2 { color: #81d4fa; margin-top: 2rem; }
    p  { color: #aaa; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th { background: #1e1e1e; color: #4fc3f7; padding: 8px 12px; text-align: left; border: 1px solid #333; }
    td { padding: 6px 12px; border: 1px solid #222; }
    tr:nth-child(even) { background: #1a1a1a; }
    .good  { color: #81c784; }
    .warn  { color: #ffb74d; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  </style>
</head>
<body>
  <h1>FinSignal — data quality report</h1>
  <p>Generated: {{ generated_at }}</p>

  <h2>Summary</h2>
  <p>Tickers: <strong>{{ n_tickers }}</strong> &nbsp;|&nbsp;
     Total rows: <strong>{{ total_rows }}</strong> &nbsp;|&nbsp;
     Date range: <strong>{{ date_min }}</strong> to <strong>{{ date_max }}</strong>
  </p>

  <h2>Per-ticker coverage</h2>
  <table>
    <tr>
      <th>Ticker</th>
      <th>Rows</th>
      <th>Start</th>
      <th>End</th>
      <th>Null close %</th>
      <th>Status</th>
    </tr>
    {% for row in rows %}
    <tr>
      <td>{{ row.ticker }}</td>
      <td>{{ row.n }}</td>
      <td>{{ row.start }}</td>
      <td>{{ row.end }}</td>
      <td>{{ row.null_pct }}</td>
      <td>
        {% if row.ok %}
          <span class="badge good">ok</span>
        {% else %}
          <span class="badge warn">sparse</span>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

def generate_report() -> None:
    df = pd.read_parquet("data/processed/ohlcv_clean.parquet")

    summary = []
    for ticker, grp in df.groupby("ticker"):
        null_pct = grp["close"].isna().mean() * 100
        summary.append({
            "ticker":   ticker,
            "n":        len(grp),
            "start":    str(grp["date"].min())[:10],
            "end":      str(grp["date"].max())[:10],
            "null_pct": f"{null_pct:.2f}%",
            "ok":       len(grp) >= 252 and null_pct < 5,
        })

    html = Template(TEMPLATE).render(
        generated_at = pd.Timestamp.now().isoformat()[:19],
        n_tickers    = len(summary),
        total_rows   = f"{len(df):,}",
        date_min     = str(df["date"].min())[:10],
        date_max     = str(df["date"].max())[:10],
        rows         = summary,
    )

    out_path = REPORT_DIR / "report.html"
    out_path.write_text(html)
    print(f"Report written to {out_path}")
    print(f"Tickers: {len(summary)} | Total rows: {len(df):,}")

if __name__ == "__main__":
    generate_report()