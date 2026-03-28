import pandas as pd
import numpy as np
from pathlib import Path
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

REPORT_DIR = Path("data/quality_report")
ARTIFACTS  = Path("artifacts")

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>FinSignal Weekly Insight Report</title>
  <style>
    body { font-family: monospace; background: #0f0f0f; color: #e0e0e0;
           padding: 2rem; max-width: 900px; margin: 0 auto; }
    h1   { color: #4fc3f7; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }
    h2   { color: #81d4fa; margin-top: 2rem; }
    h3   { color: #b0bec5; margin-top: 1.5rem; }
    p    { color: #aaa; line-height: 1.7; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.75rem; }
    th { background: #1e1e1e; color: #4fc3f7; padding: 8px 12px;
         text-align: left; border: 1px solid #333; font-size: 13px; }
    td { padding: 6px 12px; border: 1px solid #222; font-size: 13px; }
    tr:nth-child(even) { background: #1a1a1a; }
    .good    { color: #81c784; }
    .warn    { color: #ffb74d; }
    .bad     { color: #e57373; }
    .neutral { color: #aaa; }
    .card  { background: #1a1a2e; border: 1px solid #2a2a4e;
             border-radius: 6px; padding: 1rem 1.25rem; margin: 0.75rem 0; }
    .signal-up   { color: #81c784; font-weight: bold; }
    .signal-down { color: #e57373; font-weight: bold; }
    .signal-flat { color: #ffb74d; font-weight: bold; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 11px; margin: 2px; }
    .tag-bull { background: #1b3a1f; color: #81c784; }
    .tag-bear { background: #3a1b1b; color: #e57373; }
    .tag-neutral { background: #2a2a1b; color: #ffb74d; }
  </style>
</head>
<body>
  <h1>FinSignal — weekly insight report</h1>
  <p>Generated: <strong>{{ generated_at }}</strong> &nbsp;|&nbsp;
     Pipeline version: 1.0 &nbsp;|&nbsp;
     Models: XGBoost, LightGBM + Optuna</p>

  <h2>Market regime</h2>
  <div class="card">
    <p><strong>VIX:</strong>
      <span class="{{ 'bad' if macro.vix > 25 else 'good' if macro.vix < 15 else 'warn' }}">
        {{ macro.vix }}</span>
      &nbsp;
      {% if macro.vix > 25 %}
        <span class="tag tag-bear">high fear</span>
      {% elif macro.vix < 15 %}
        <span class="tag tag-bull">low fear</span>
      {% else %}
        <span class="tag tag-neutral">neutral</span>
      {% endif %}
    </p>
    <p><strong>Fed Funds Rate:</strong> {{ macro.fed_funds }}%</p>
    <p><strong>10yr Yield:</strong> {{ macro.yield_10yr }}%</p>
    <p><strong>CPI (latest):</strong> {{ macro.cpi }}</p>
    <p><strong>Unemployment:</strong> {{ macro.unemployment }}%</p>
  </div>

  <h2>Model signals — top tickers</h2>
  <p>Current model signal for each ticker based on latest features.
     Signal = predicted 5-day forward return direction.</p>
  <table>
    <tr>
      <th>Ticker</th>
      <th>Close</th>
      <th>RSI</th>
      <th>Vol 21d</th>
      <th>LGBM signal</th>
      <th>Regime</th>
    </tr>
    {% for row in signals %}
    <tr>
      <td>{{ row.ticker }}</td>
      <td>${{ row.close }}</td>
      <td class="{{ 'bad' if row.rsi > 70 else 'good' if row.rsi < 30 else 'neutral' }}">
        {{ row.rsi }}</td>
      <td>{{ row.vol }}</td>
      <td>
        {% if row.signal == 'LONG' %}
          <span class="signal-up">▲ LONG</span>
        {% elif row.signal == 'SHORT' %}
          <span class="signal-down">▼ SHORT</span>
        {% else %}
          <span class="signal-flat">— FLAT</span>
        {% endif %}
      </td>
      <td>
        {% if row.vix_regime %}
          <span class="tag tag-bear">high vol</span>
        {% else %}
          <span class="tag tag-bull">low vol</span>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
  </table>

  <h2>Model performance summary</h2>
  <table>
    <tr>
      <th>Ticker</th><th>Model</th><th>Sharpe</th>
      <th>IC</th><th>Hit rate</th><th>Assessment</th>
    </tr>
    {% for row in performance %}
    <tr>
      <td>{{ row.ticker }}</td>
      <td>{{ row.model }}</td>
      <td class="{{ 'good' if row.sharpe > 0.3 else 'warn' if row.sharpe > 0 else 'bad' }}">
        {{ row.sharpe }}</td>
      <td class="{{ 'good' if row.ic > 0.05 else 'warn' if row.ic > 0 else 'bad' }}">
        {{ row.ic }}</td>
      <td>{{ row.hit_rate }}</td>
      <td>
        {% if row.sharpe > 0.3 and row.ic > 0.05 %}
          <span class="tag tag-bull">strong signal</span>
        {% elif row.sharpe > 0 %}
          <span class="tag tag-neutral">weak signal</span>
        {% else %}
          <span class="tag tag-bear">no signal</span>
        {% endif %}
      </td>
    </tr>
    {% endfor %}
  </table>

  <h2>Anomalies detected</h2>
  {% if anomalies %}
  <table>
    <tr><th>Date</th><th>Ticker</th><th>Log return</th><th>Method</th></tr>
    {% for row in anomalies %}
    <tr>
      <td>{{ row.date }}</td>
      <td>{{ row.ticker }}</td>
      <td class="{{ 'bad' if row.log_return < 0 else 'good' }}">
        {{ row.log_return }}</td>
      <td>{{ row.method }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p class="neutral">No anomalies detected in the current window.</p>
  {% endif %}

  <h2>Methodology</h2>
  <div class="card">
    <p>
      <strong>Data:</strong> 5yr daily OHLCV for 80 S&P 500 tickers +
      FRED macro indicators (CPI, Fed Funds, Unemployment, 10yr Yield, M2, VIX).
    </p>
    <p>
      <strong>Features:</strong> 24 features including RSI, MACD, Bollinger Bands,
      ATR, rolling z-score, momentum (5/10/21/63d), volatility (5/21/63d),
      and lagged macro indicators.
    </p>
    <p>
      <strong>Models:</strong> XGBoost (baseline) and LightGBM with Optuna
      Bayesian HPO (30 trials). Both evaluated via 5-fold walk-forward
      validation with no lookahead bias.
    </p>
    <p>
      <strong>Target:</strong> 5-day forward return. Macro features lagged
      by 1 period to prevent data leakage.
    </p>
  </div>
</body>
</html>
"""

def get_latest_macro() -> dict:
    try:
        macro = pd.read_parquet("data/raw/macro/macro_indicators.parquet")
        macro["date"] = pd.to_datetime(macro["date"])
        macro = macro.set_index("date").ffill()
        latest = macro.iloc[-1]
        return {
            "vix":          round(float(latest.get("VIX", 0)), 2),
            "fed_funds":    round(float(latest.get("FedFunds", 0)), 2),
            "yield_10yr":   round(float(latest.get("Yield10yr", 0)), 2),
            "cpi":          round(float(latest.get("CPI", 0)), 2),
            "unemployment": round(float(latest.get("Unemployment", 0)), 2),
        }
    except Exception as e:
        print(f"Macro load error: {e}")
        return {
            "vix": 0, "fed_funds": 0,
            "yield_10yr": 0, "cpi": 0, "unemployment": 0,
        }

def get_current_signals(
    df: pd.DataFrame,
    tickers: list[str],
) -> list[dict]:
    """
    Get the latest row for each ticker and generate a signal
    based on the most recent model prediction direction.
    """
    signals = []
    for ticker in tickers:
        sub = df[df["ticker"] == ticker].sort_values("date")
        if sub.empty:
            continue
        latest = sub.iloc[-1]

        # Load latest prediction if available
        pred_path = ARTIFACTS / f"lgbm_{ticker}_preds.parquet"
        signal    = "FLAT"
        if pred_path.exists():
            preds = pd.read_parquet(pred_path)
            if not preds.empty:
                last_pred = preds.iloc[-1]["predicted"]
                if last_pred > 0.002:
                    signal = "LONG"
                elif last_pred < -0.002:
                    signal = "SHORT"

        signals.append({
            "ticker":     ticker,
            "close":      round(float(latest["close"]), 2),
            "rsi":        round(float(latest.get("rsi_14", 0)), 1),
            "vol":        f"{float(latest.get('vol_21d', 0))*100:.2f}%",
            "signal":     signal,
            "vix_regime": bool(latest.get("vix_high_regime", 0)),
        })

    return signals

def generate_report() -> None:
    print("Generating insight report...")

    df = pd.read_parquet("data/processed/full_features_v2.parquet")

    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "XLF", "XLK"]
    tickers = [t for t in tickers if t in df["ticker"].values]

    # Macro context
    macro = get_latest_macro()
    print(f"VIX: {macro['vix']} | Fed Funds: {macro['fed_funds']}%")

    # Current signals
    signals = get_current_signals(df, tickers)

    # Model performance
    comp_path = ARTIFACTS / "model_comparison.parquet"
    performance = []
    if comp_path.exists():
        comp = pd.read_parquet(comp_path)
        comp = comp[comp["ticker"].isin(tickers)]
        performance = comp.to_dict("records")
        for row in performance:
            row["sharpe"]   = round(row["sharpe"], 3)
            row["ic"]       = round(row["ic"], 3)
            row["hit_rate"] = f"{row['hit_rate']:.1%}"

    # Recent anomalies
    anom_path = Path("data/processed/anomaly_dates.parquet")
    anomalies = []
    if anom_path.exists():
        anom = pd.read_parquet(anom_path)
        anom["date"] = pd.to_datetime(anom["date"])
        recent = anom[
            anom["date"] >= pd.Timestamp.now() - pd.Timedelta(days=30)
        ].sort_values("date", ascending=False).head(10)
        for _, row in recent.iterrows():
            anomalies.append({
                "date":       str(row["date"])[:10],
                "ticker":     row["ticker"],
                "log_return": round(float(row["log_return"]), 4),
                "method":     row.get("method", "isolation_forest"),
            })

    # Render HTML
    html = Template(TEMPLATE).render(
        generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        macro        = type("M", (), macro)(),
        signals      = signals,
        performance  = performance,
        anomalies    = anomalies,
    )

    out_path = REPORT_DIR / "insight_report.html"
    out_path.write_text(html)
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    generate_report()