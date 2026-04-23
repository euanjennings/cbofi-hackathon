import os
import glob
import joblib
import pandas as pd
import warnings
from flask import Flask, render_template_string, request, redirect, url_for

warnings.filterwarnings("ignore")

app = Flask(__name__)


MODEL_PATH = os.environ.get("MODEL_PATH", "fraud_classifier.pkl")
CSV_PATH   = os.environ.get("CSV_PATH",   "risk_ai_data_v3_3.csv")


# Load model + data (done once at startup)
model = joblib.load(MODEL_PATH)

# Discover feature columns expected by the model's ColumnTransformer
_ct           = model.named_steps["preprocessing"]
_cat_cols     = list(_ct.transformers_[0][2])
_num_cols     = list(_ct.transformers_[1][2])
FEATURE_COLS  = _cat_cols + _num_cols

df_data = pd.read_csv(CSV_PATH)

if "fraud_label" not in df_data.columns:
    _suspicious = {"High_Value_Online", "Geographic_Anomaly",
                   "Social_Engineering_Indicators", "Unusual_Merchant_Category"}
    def _derive(row):
        if row["risk_score"] >= 0.65:
            return 1
        if row["risk_score"] >= 0.45 and row.get("trigger_reason") in _suspicious:
            return 1
        return 0
    df_data["fraud_label"] = df_data.apply(_derive, axis=1)

_missing = [c for c in FEATURE_COLS if c not in df_data.columns]
if _missing:
    raise ValueError(f"CSV is missing columns required by the model: {_missing}")

current_index      = 0
prediction_history = []


# Helpers

def get_features(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([row[FEATURE_COLS]])


def session_stats():
    total   = len(prediction_history)
    correct = sum(1 for p in prediction_history if p["correct"] is True)
    wrong   = sum(1 for p in prediction_history if p["correct"] is False)
    accuracy = correct / total if total else 0

    tp = sum(1 for p in prediction_history if p["prediction"] == 1 and p["actual"] == 1)
    fp = sum(1 for p in prediction_history if p["prediction"] == 1 and p["actual"] == 0)
    fn = sum(1 for p in prediction_history if p["prediction"] == 0 and p["actual"] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_conf  = sum(p["confidence"] for p in prediction_history) / total if total else 0.0

    return total, correct, wrong, accuracy, precision, recall, f1, avg_conf


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection System</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0e1a; --surface: #111827; --surface2: #1a2236;
    --border: #1e2d45; --accent: #00d4ff; --accent2: #7c3aed;
    --danger: #ef4444; --safe: #10b981; --warn: #f59e0b;
    --text: #e2e8f0; --muted: #64748b;
    --mono: 'IBM Plex Mono', monospace; --sans: 'IBM Plex Sans', sans-serif;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--bg); color: var(--text); font-family: var(--sans);
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse at 20% 20%, rgba(0,212,255,.04) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 80%, rgba(124,58,237,.04) 0%, transparent 50%);
  }
  header {
    border-bottom: 1px solid var(--border); padding: 16px 32px;
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(17,24,39,.8); backdrop-filter: blur(10px);
    position: sticky; top: 0; z-index: 100;
  }
  .logo { font-family: var(--mono); font-size:13px; font-weight:600;
          color: var(--accent); letter-spacing:.1em; text-transform:uppercase; }
  .logo span { color: var(--muted); font-weight:400; }
  .header-meta { font-family: var(--mono); font-size:11px; color: var(--muted); }
  .header-meta strong { color: var(--text); }
  .layout { display: grid; grid-template-columns: 1fr 380px; gap:0; min-height: calc(100vh - 57px); }
  .main-panel { padding: 32px; border-right: 1px solid var(--border); }
  .side-panel { padding: 24px; background: var(--surface); display: flex; flex-direction: column; }
  .section-label {
    font-family: var(--mono); font-size:10px; letter-spacing:.15em;
    text-transform:uppercase; color: var(--muted); margin-bottom:16px;
    display:flex; align-items:center; gap:8px; flex-shrink: 0;
  }
  .section-label::after { content:''; flex:1; height:1px; background: var(--border); }
  .tx-card { background: var(--surface); border: 1px solid var(--border); border-radius:8px; overflow:hidden; margin-bottom:20px; }
  .tx-card-header {
    padding: 14px 20px; background: var(--surface2); border-bottom: 1px solid var(--border);
    display:flex; justify-content:space-between; align-items:center;
  }
  .tx-id { font-family: var(--mono); font-size:13px; font-weight:600; color: var(--accent); }
  .tx-index { font-family: var(--mono); font-size:11px; color: var(--muted); }
  .source-tag {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; border-radius: 4px; font-family: var(--mono);
    font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  }
  .source-csv  { background: rgba(124,58,237,.15); color: #a78bfa; border: 1px solid rgba(124,58,237,.3); }
  .source-model{ background: rgba(0,212,255,.1);   color: var(--accent); border: 1px solid rgba(0,212,255,.25); }
  .tx-fields { display: grid; grid-template-columns: 1fr 1fr; gap:0; }
  .field { padding:14px 20px; border-bottom:1px solid var(--border); border-right:1px solid var(--border); }
  .field:nth-child(even) { border-right:none; }
  .field:nth-last-child(-n+2) { border-bottom:none; }
  .field-label { font-family: var(--mono); font-size:10px; letter-spacing:.08em; color: var(--muted); text-transform:uppercase; margin-bottom:6px; }
  .field-value { font-family: var(--mono); font-size:14px; font-weight:500; color: var(--text); }
  .field-value.highlight { color: var(--accent); }
  .risk-bar-wrap { padding:16px 20px; border-bottom:1px solid var(--border); background:rgba(0,212,255,.02); }
  .risk-bar-label { display:flex; justify-content:space-between; font-family: var(--mono); font-size:11px; color: var(--muted); margin-bottom:8px; }
  .risk-val { color: var(--text); font-weight:600; }
  .risk-bar { height:6px; background: var(--border); border-radius:3px; overflow:hidden; }
  .risk-fill { height:100%; border-radius:3px; transition: width .6s ease; }
  .btn-analyze {
    width:100%; padding:14px; font-family: var(--mono); font-size:13px; font-weight:600;
    letter-spacing:.08em; text-transform:uppercase; background:transparent;
    color: var(--accent); border: 1px solid var(--accent); cursor:pointer;
    transition: all .2s; border-radius:4px; margin-bottom:16px;
  }
  .btn-analyze:hover { background:rgba(0,212,255,.1); box-shadow:0 0 20px rgba(0,212,255,.2); }
  .result-banner {
    border-radius:6px; padding:20px 24px; margin-top:4px; border:1px solid;
    animation: slideUp .3s ease;
  }
  @keyframes slideUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  .result-banner.fraud { background:rgba(239,68,68,.08); border-color:rgba(239,68,68,.3); }
  .result-banner.legit { background:rgba(16,185,129,.08); border-color:rgba(16,185,129,.3); }
  .result-title { font-family: var(--mono); font-size:15px; font-weight:600; margin-bottom:8px; }
  .result-title.fraud { color: var(--danger); }
  .result-title.legit { color: var(--safe); }
  .result-meta { font-family: var(--mono); font-size:11px; color: var(--muted); display:flex; gap:20px; flex-wrap:wrap; }
  .result-meta span { display:flex; gap:6px; }
  .result-meta strong { color: var(--text); }
  .correct-badge { display:inline-block; padding:2px 8px; border-radius:3px; font-size:10px; font-weight:600; font-family:var(--mono); letter-spacing:.05em; }
  .correct-badge.correct   { background:rgba(16,185,129,.2); color:var(--safe); }
  .correct-badge.incorrect { background:rgba(239,68,68,.2);  color:var(--danger); }
  .truth-block {
    margin-top: 10px; padding: 12px 16px; border-radius: 6px;
    border: 1px dashed rgba(255,255,255,.1);
    font-family: var(--mono); font-size: 11px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }
  .truth-label { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
  .truth-val   { font-size: 13px; font-weight: 600; }
  .truth-val.fraud { color: var(--danger); }
  .truth-val.legit { color: var(--safe); }
  .nav-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:20px; }
  .btn-nav {
    padding:10px; font-family:var(--mono); font-size:11px; font-weight:500;
    letter-spacing:.06em; text-transform:uppercase; background:transparent;
    color:var(--muted); border:1px solid var(--border); cursor:pointer;
    border-radius:4px; transition:all .2s; width:100%;
  }
  .btn-nav:hover { color:var(--text); border-color:var(--muted); }
  .stat-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; flex-shrink:0; }
  .stat-item { background:var(--surface2); border:1px solid var(--border); border-radius:6px; padding:14px; text-align:center; }
  .stat-num { font-family:var(--mono); font-size:26px; font-weight:600; color:var(--accent); line-height:1; }
  .stat-num.stat-sm { font-size:20px; }
  .stat-num.good { color:var(--safe); } .stat-num.bad { color:var(--danger); }
  .stat-lbl { font-family:var(--mono); font-size:9px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-top:6px; }
  .acc-wrap { margin-bottom:16px; flex-shrink:0; }
  .acc-header { display:flex; justify-content:space-between; font-family:var(--mono); font-size:10px; color:var(--muted); margin-bottom:6px; }
  .acc-pct { color:var(--accent); font-weight:600; }
  .acc-bar { height:4px; background:var(--border); border-radius:2px; overflow:hidden; }
  .acc-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,var(--accent2),var(--accent)); transition:width .5s ease; }

  /* Scrollable history table */
  .history-wrap {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 8px;
  }
  .history-wrap::-webkit-scrollbar { width: 4px; }
  .history-wrap::-webkit-scrollbar-track { background: transparent; }
  .history-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .history-wrap::-webkit-scrollbar-thumb:hover { background: var(--muted); }
  .history-table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:11px; }
  .history-table thead th {
    position: sticky; top: 0; z-index: 1;
    background: var(--surface2);
    text-align:left; padding:8px 10px; font-size:9px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted);
    border-bottom:1px solid var(--border);
  }
  .history-table td { padding:8px 10px; border-bottom:1px solid rgba(30,45,69,.5); color:var(--text); vertical-align:middle; }
  .history-table tbody tr:last-child td { border-bottom: none; }
  .history-count {
    font-family: var(--mono); font-size: 10px; color: var(--muted);
    text-align: right; padding: 4px 0 8px;
    flex-shrink: 0;
  }
  .history-count strong { color: var(--text); }
  .badge { display:inline-block; padding:2px 6px; border-radius:3px; font-size:9px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }
  .badge-fraud { background:rgba(239,68,68,.2); color:var(--danger); }
  .badge-safe  { background:rgba(16,185,129,.2); color:var(--safe); }
  .dot { font-size:14px; }
  .dot.ok { color:var(--safe); } .dot.fail { color:var(--danger); }
  .btn-reset {
    width:100%; padding:9px; font-family:var(--mono); font-size:10px; letter-spacing:.08em;
    text-transform:uppercase; background:transparent; color:var(--muted);
    border:1px solid var(--border); cursor:pointer; border-radius:4px; margin-top:8px;
    transition:all .2s; flex-shrink:0;
  }
  .btn-reset:hover { color:var(--text); border-color:var(--muted); }
  .empty-state { font-family:var(--mono); font-size:11px; color:var(--muted); text-align:center; padding:20px 0; }
  @media(max-width:900px){
    .layout{grid-template-columns:1fr;}
    .tx-fields{grid-template-columns:1fr;}
    .field{border-right:none;}
    .side-panel { min-height: 400px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">Fraud<span>Detect</span> / AI Risk Engine</div>
  <div class="header-meta">
    Transaction <strong>{{ current_index + 1 }}</strong> of <strong>{{ total_transactions }}</strong>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Session accuracy:
    <strong style="color:{% if accuracy >= 0.7 %}#10b981{% elif accuracy >= 0.5 %}#f59e0b{% else %}#ef4444{% endif %}">
      {{ "%.0f"|format(accuracy * 100) }}%
    </strong>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <span title="Model file">{{ model_name }}</span>
    &nbsp;&middot;&nbsp;
    <span title="Data file">{{ csv_name }}</span>
  </div>
</header>

<div class="layout">
  <!-- Main Panel -->
  <div class="main-panel">
    <div class="section-label">Transaction Analysis</div>

    {% if data %}
    <div class="tx-card">
      <div class="tx-card-header">
        <span class="tx-id">TXN-{{ "%04d"|format(current_index + 1) }}</span>
        <span class="tx-index">Record {{ current_index + 1 }} / {{ total_transactions }}</span>
      </div>

      {% set rs = data.get('risk_score', 0)|float %}
      <div class="risk-bar-wrap">
        <div class="risk-bar-label">
          <span>Risk Score</span>
          <span class="risk-val">{{ "%.2f"|format(rs) }}</span>
        </div>
        <div class="risk-bar">
          <div class="risk-fill" style="width:{{ rs * 100 }}%; background:{% if rs >= 0.65 %}#ef4444{% elif rs >= 0.45 %}#f59e0b{% else %}#10b981{% endif %};"></div>
        </div>
      </div>

      <div class="tx-fields">
        {% for key, value in data.items() %}
          {% if key not in ['fraud_label', 'risk_score'] %}
          <div class="field">
            <div class="field-label">{{ key.replace('_', ' ') }}</div>
            <div class="field-value {% if key == 'amount_at_risk' %}highlight{% endif %}">
              {% if key == 'amount_at_risk' %}${{ value }}{% else %}{{ value }}{% endif %}
            </div>
          </div>
          {% endif %}
        {% endfor %}
      </div>
    </div>

    <form action="/analyze" method="POST">
      <button type="submit" class="btn-analyze">▶ Run Model Analysis</button>
    </form>

    {% if prediction is not none %}
    <div class="result-banner {% if prediction == 1 %}fraud{% else %}legit{% endif %}">
      <div class="result-title {% if prediction == 1 %}fraud{% else %}legit{% endif %}">
        {% if prediction == 1 %}⚠ Fraudulent Transaction Detected{% else %}✓ Transaction Appears Legitimate{% endif %}
      </div>
      <div class="result-meta">
        <span>Model confidence <strong>{{ [1, (probability * 100)|round|int]|max }}%</strong></span>
        <span>
          {% if is_correct %}
            <span class="correct-badge correct">✓ Correct</span>
          {% else %}
            <span class="correct-badge incorrect">✗ Incorrect</span>
          {% endif %}
        </span>
      </div>

      <div class="truth-block" style="margin-top:14px;">
        <div>
          <div class="truth-label">CSV Label</div>
          <div class="truth-val {% if csv_label == 1 %}fraud{% else %}legit{% endif %}">
            {% if csv_label == 1 %}Fraud{% else %}Legitimate{% endif %}
          </div>
        </div>
        <div style="color:var(--border);">|</div>
        <div>
          <div class="truth-label">Model Prediction</div>
          <div class="truth-val {% if prediction == 1 %}fraud{% else %}legit{% endif %}">
            {% if prediction == 1 %}Fraud{% else %}Legitimate{% endif %}
          </div>
        </div>
        <div style="color:var(--border);">|</div>
        <div>
          <div class="truth-label">Label Source</div>
          {% if label_from_csv %}
            <span class="source-tag source-csv">CSV column</span>
          {% else %}
            <span class="source-tag source-model">Derived</span>
          {% endif %}
        </div>
      </div>
    </div>
    {% endif %}

    {% else %}
    <div class="empty-state" style="padding:60px 0;">
      <p style="font-family:var(--mono);color:var(--muted);font-size:12px;">No transaction loaded</p>
    </div>
    {% endif %}

    <div class="nav-row">
      <form action="/previous" method="GET">
        <button type="submit" class="btn-nav">← Previous</button>
      </form>
      <form action="/next" method="GET">
        <button type="submit" class="btn-nav">Next →</button>
      </form>
    </div>
  </div>

  <!-- Side Panel -->
  <div class="side-panel">
    <div class="section-label">Model Performance</div>

    <div class="stat-row">
      <div class="stat-item">
        <div class="stat-num">{{ total_analyzed }}</div>
        <div class="stat-lbl">Analyzed</div>
      </div>
      <div class="stat-item">
        <div class="stat-num {% if accuracy >= 0.7 %}good{% elif accuracy < 0.5 %}bad{% endif %}">
          {{ "%.0f"|format(accuracy * 100) }}%
        </div>
        <div class="stat-lbl">Accuracy</div>
      </div>
    </div>

    <div class="stat-row">
      <div class="stat-item">
        <div class="stat-num good">{{ correct_predictions }}</div>
        <div class="stat-lbl">Correct</div>
      </div>
      <div class="stat-item">
        <div class="stat-num bad">{{ incorrect_predictions }}</div>
        <div class="stat-lbl">Incorrect</div>
      </div>
    </div>

    <div class="stat-row">
      <div class="stat-item">
        <div class="stat-num stat-sm {% if precision >= 0.7 %}good{% elif precision < 0.4 %}bad{% endif %}">
          {% if total_analyzed > 0 %}{{ "%.0f"|format(precision * 100) }}%{% else %}—{% endif %}
        </div>
        <div class="stat-lbl">Precision</div>
      </div>
      <div class="stat-item">
        <div class="stat-num stat-sm {% if recall >= 0.7 %}good{% elif recall < 0.4 %}bad{% endif %}">
          {% if total_analyzed > 0 %}{{ "%.0f"|format(recall * 100) }}%{% else %}—{% endif %}
        </div>
        <div class="stat-lbl">Recall</div>
      </div>
    </div>

    <div class="acc-wrap">
      <div class="acc-header">
        <span>Session Accuracy</span>
        <span class="acc-pct">{{ "%.0f"|format(accuracy * 100) }}%</span>
      </div>
      <div class="acc-bar">
        <div class="acc-fill" style="width:{{ accuracy * 100 }}%;"></div>
      </div>
    </div>

    <div class="section-label">Predictions</div>

    {% if history|length > 0 %}
    <div class="history-count">
      <strong>{{ total_analyzed }}</strong> total &nbsp;·&nbsp;
      <strong style="color:var(--safe);">{{ correct_predictions }}</strong> correct &nbsp;·&nbsp;
      <strong style="color:var(--danger);">{{ incorrect_predictions }}</strong> wrong
    </div>
    <div class="history-wrap" style="max-height:260px;overflow-y:auto;">
      <table class="history-table">
        <thead>
          <tr><th>TXN</th><th>Pred</th><th>Label</th><th>Conf.</th><th>OK?</th></tr>
        </thead>
        <tbody>
          {% for item in history %}
          <tr>
            <td>{{ item.id }}</td>
            <td>
              {% if item.prediction == 1 %}
                <span class="badge badge-fraud">Fraud</span>
              {% else %}
                <span class="badge badge-safe">Safe</span>
              {% endif %}
            </td>
            <td>
              {% if item.actual == 1 %}
                <span class="badge badge-fraud">Fraud</span>
              {% else %}
                <span class="badge badge-safe">Safe</span>
              {% endif %}
            </td>
            <td>{{ [1, (item.confidence * 100)|round|int]|max }}%</td>
            <td>
              {% if item.correct is none %}
                <span style="color:var(--muted);">—</span>
              {% elif item.correct %}
                <span class="dot ok">&#10003;</span>
              {% else %}
                <span class="dot fail">&#10007;</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <div class="empty-state">No predictions yet</div>
    {% endif %}

    <form action="/reset" method="GET">
      <button type="submit" class="btn-reset">Reset Session</button>
    </form>
  </div>
</div>

</body>
</html>
"""


#----------------------------------------------------------------------------



_label_from_csv = "fraud_label" in pd.read_csv(CSV_PATH).columns

def base_context():
    total, correct, wrong, accuracy, precision, recall, f1, avg_conf = session_stats()
    return dict(
        model_name=os.path.basename(MODEL_PATH),
        csv_name=os.path.basename(CSV_PATH),
        total_transactions=len(df_data),
        current_index=current_index,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        avg_conf=avg_conf,
        total_analyzed=total,
        correct_predictions=correct,
        incorrect_predictions=wrong,
        history=prediction_history,
        prediction=None,
        probability=0,
        csv_label=None,
        is_correct=None,
        label_from_csv=_label_from_csv,
    )


def render_transaction():
    row = df_data.iloc[current_index]
    ctx = base_context()
    ctx["data"] = row.to_dict()
    return render_template_string(HTML, **ctx)


# ---------------------------------------------------------------------------



@app.route("/")
def home():
    return redirect(url_for("next_transaction"))


@app.route("/next")
def next_transaction():
    global current_index
    current_index = (current_index + 1) % len(df_data)
    return render_transaction()


@app.route("/previous")
def previous_transaction():
    global current_index
    current_index = (current_index - 1) % len(df_data)
    return render_transaction()


@app.route("/analyze", methods=["POST"])
def analyze():
    global current_index, prediction_history

    try:
        row       = df_data.iloc[current_index]
        csv_label = int(row["fraud_label"])

        X          = get_features(row)
        pred       = int(model.predict(X)[0])
        prob       = float(model.predict_proba(X)[0][1])
        confidence = prob if pred == 1 else 1 - prob   # certainty of the actual prediction
        is_correct = (pred == csv_label)

        prediction_history.insert(0, {
            "id":         f"T{current_index + 1}",
            "prediction": pred,
            "confidence": confidence,
            "correct":    is_correct,
            "actual":     csv_label,
        })
      

        total, correct, wrong, accuracy, precision, recall, f1, avg_conf = session_stats()

        ctx = dict(
            model_name=os.path.basename(MODEL_PATH),
            csv_name=os.path.basename(CSV_PATH),
            data=row.to_dict(),
            prediction=pred,
            probability=confidence,
            csv_label=csv_label,
            is_correct=is_correct,
            label_from_csv=_label_from_csv,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            avg_conf=avg_conf,
            total_analyzed=total,
            correct_predictions=correct,
            incorrect_predictions=wrong,
            history=prediction_history,
            current_index=current_index,
            total_transactions=len(df_data),
        )
        return render_template_string(HTML, **ctx)

    except Exception as e:
        return f"<pre style='color:red;padding:20px'>Error: {e}</pre>"


@app.route("/reset")
def reset():
    global prediction_history, current_index
    prediction_history.clear()
    current_index = 0
    return render_transaction()


if __name__ == "__main__":
    app.run(debug=True)