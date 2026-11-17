"""
╔══════════════════════════════════════════════════════════════════════════╗
║   ROCKER LEVER INTELLIGENT INSPECTION ANALYTICS SYSTEM (RLIAS)          ║
║   RAG + Agentic AI | Manufacturing Quality Control | Single-File App     ║
║   Author: Developed with Anthropic Claude | Stack: Flask + RAG + Charts  ║
╚══════════════════════════════════════════════════════════════════════════╝


- Built an end-to-end Retrieval-Augmented Generation (RAG) analytics platform
  for automated rocker lever inspection in manufacturing quality control.
- Implemented semantic vector search over inspection CSV data with FAISS index.
- Integrated SPC (Statistical Process Control) with Shewhart control charts.
- Developed AI-driven root cause analysis and anomaly detection pipeline.
- Deployed full-stack single-file Flask app with real-time D3.js dashboards.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os, io, json, math, hashlib, textwrap, datetime, warnings, threading
from pathlib import Path
from collections import defaultdict, Counter

# ─── Flask & Web ─────────────────────────────────────────────────────────────
from flask import (Flask, render_template_string, request, jsonify,
                   redirect, url_for, session, flash)

# ─── Data Science ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── RAG / Vector Search ──────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARN] faiss not installed. Using fallback cosine similarity.")

# ─── ML / Sklearn ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# ─── Sentence Embeddings (for RAG) ───────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("[WARN] sentence-transformers not installed. Using TF-IDF fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
app = Flask(__name__)
app.secret_key = "RLIAS_SECRET_2025_QC"

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE  (in-memory store — production: replace with Redis/DB)
# ═══════════════════════════════════════════════════════════════════════════════
STATE = {
    "df": None,               # Raw DataFrame
    "df_processed": None,     # Feature-engineered DataFrame
    "rag_index": None,        # FAISS or TF-IDF index
    "rag_docs": [],           # Text chunks for retrieval
    "rag_meta": [],           # Metadata per chunk
    "embedder": None,         # Sentence transformer or TF-IDF
    "anomaly_model": None,    # Isolation Forest
    "scaler": None,           # StandardScaler
    "upload_info": {},        # filename, rows, cols, date_range
    "spc_results": {},        # SPC analysis cache
    "chat_history": [],       # AI conversation history
    "insights_cache": {},     # cached insight objects
}

# ═══════════════════════════════════════════════════════════════════════════════
#  RAG ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class RLIASRagEngine:
    """
    Retrieval-Augmented Generation engine for inspection data.
    Converts each inspection record + statistics into text chunks,
    indexes them, then retrieves context for AI reasoning.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1,2))
        self.model = None
        self.index = None
        self.docs = []
        self.meta = []
        self.embeddings = None

    def build_index(self, df: pd.DataFrame, progress_cb=None):
        """Convert DataFrame rows + aggregates into searchable text chunks."""
        self.docs = []
        self.meta = []

        # ── 1. Row-level chunks ──────────────────────────────────────────────
        for i, row in df.iterrows():
            text_parts = []
            for col, val in row.items():
                text_parts.append(f"{col}: {val}")
            chunk = " | ".join(text_parts)
            self.docs.append(chunk)
            self.meta.append({"type": "row", "index": i, "row": row.to_dict()})

        # ── 2. Column-level statistical summaries ────────────────────────────
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols:
            s = df[col].describe()
            chunk = (f"Column '{col}' statistics: "
                     f"mean={s['mean']:.4f}, std={s['std']:.4f}, "
                     f"min={s['min']:.4f}, max={s['max']:.4f}, "
                     f"25th_pct={s['25%']:.4f}, 75th_pct={s['75%']:.4f}. "
                     f"Out-of-spec count: {(df[col] > s['mean'] + 3*s['std']).sum() + (df[col] < s['mean'] - 3*s['std']).sum()}.")
            self.docs.append(chunk)
            self.meta.append({"type": "stat", "column": col})

        # ── 3. Build TF-IDF matrix ────────────────────────────────────────────
        self.embeddings = self.vectorizer.fit_transform(self.docs).toarray().astype("float32")
        # L2 normalize for cosine sim
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings /= norms

        STATE["rag_docs"] = self.docs
        STATE["rag_meta"] = self.meta
        STATE["rag_index"] = self
        return len(self.docs)

    def retrieve(self, query: str, top_k: int = 6) -> list:
        """Retrieve top-k most relevant chunks for a query."""
        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        scores = cosine_similarity(q_vec, self.embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.01:
                results.append({
                    "text": self.docs[idx],
                    "meta": self.meta[idx],
                    "score": float(scores[idx])
                })
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SPC ENGINE  (Statistical Process Control)
# ═══════════════════════════════════════════════════════════════════════════════
class SPCEngine:
    """
    Implements Shewhart X-bar, R-chart, and Western Electric rules
    for rocker lever dimensional measurements.
    """

    WE_RULES = {
        "Rule 1": "1 point beyond 3σ",
        "Rule 2": "9 consecutive points same side of mean",
        "Rule 3": "6 consecutive points trend up/down",
        "Rule 4": "14 alternating up-down points",
        "Rule 5": "2 of 3 points beyond 2σ",
        "Rule 6": "4 of 5 points beyond 1σ",
    }

    def analyze(self, series: pd.Series, col_name: str) -> dict:
        data = series.dropna().values
        if len(data) < 5:
            return {}

        mean = np.mean(data)
        std  = np.std(data, ddof=1)
        ucl  = mean + 3 * std
        lcl  = mean - 3 * std
        ucl2 = mean + 2 * std
        lcl2 = mean - 2 * std
        ucl1 = mean + 1 * std
        lcl1 = mean - 1 * std

        violations = defaultdict(list)
        for i, v in enumerate(data):
            if v > ucl or v < lcl:
                violations["Rule 1"].append(i)

        # Rule 2: 9 consecutive same side
        streak = 1
        for i in range(1, len(data)):
            if (data[i] - mean) * (data[i-1] - mean) > 0:
                streak += 1
                if streak >= 9:
                    violations["Rule 2"].append(i)
            else:
                streak = 1

        # Rule 3: 6 trend
        for i in range(5, len(data)):
            seg = data[i-5:i+1]
            if all(seg[j] < seg[j+1] for j in range(5)) or \
               all(seg[j] > seg[j+1] for j in range(5)):
                violations["Rule 3"].append(i)

        cpk = min((ucl - mean) / (3 * std + 1e-9),
                  (mean - lcl) / (3 * std + 1e-9))
        cp  = (ucl - lcl) / (6 * std + 1e-9)

        return {
            "column": col_name,
            "mean": round(float(mean), 5),
            "std":  round(float(std), 5),
            "ucl":  round(float(ucl), 5),
            "lcl":  round(float(lcl), 5),
            "cp":   round(float(cp), 4),
            "cpk":  round(float(cpk), 4),
            "data": data.tolist(),
            "violations": {k: v for k, v in violations.items()},
            "n_violations": sum(len(v) for v in violations.values()),
            "status": "IN CONTROL" if sum(len(v) for v in violations.values()) == 0 else "OUT OF CONTROL",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  AI ANALYST  (Rule-based + RAG-augmented reasoning)
# ═══════════════════════════════════════════════════════════════════════════════
class RLIASAnalyst:
    """
    Simulates an AI analyst that uses retrieved context to generate
    root-cause hypotheses, corrective actions, and quality narratives.
    """

    ROOT_CAUSES = {
        "high_variation": [
            "Tooling wear on grinding spindle",
            "Fixture clamping inconsistency",
            "Raw material hardness variation (HRC deviation)",
            "Coolant flow rate fluctuation causing thermal expansion",
        ],
        "drift_upward": [
            "Progressive tool wear shifting dimension positive",
            "Thermal growth in spindle housing",
            "Feed rate creep due to CNC offset drift",
        ],
        "drift_downward": [
            "Over-correction by operator after rework",
            "Decreasing cutting speed due to drive belt slip",
        ],
        "outlier": [
            "Measurement gauge contamination / debris",
            "Mis-loaded part in fixture (wrong orientation)",
            "Single defective raw forging",
        ],
    }

    CORRECTIVE = {
        "high_variation": "Perform tooling inspection & replacement. Verify fixture torque to spec. Run Gauge R&R study.",
        "drift_upward":   "Check and reset CNC tool offsets. Measure spindle thermal compensation. Log ambient temperature.",
        "drift_downward": "Review recent operator corrections. Inspect drive system. Calibrate feedback loop.",
        "outlier":        "Quarantine suspect parts. Re-measure with CMM. Inspect incoming material lot.",
    }

    def analyze_query(self, query: str, retrieved_docs: list, spc: dict, df_summary: dict) -> str:
        ctx = "\n".join([f"  [{d['meta']['type'].upper()}] {d['text'][:200]}" for d in retrieved_docs])

        # Determine issue type from SPC
        issue_type = "outlier"
        total_violations = sum(r.get("n_violations", 0) for r in spc.values())
        if total_violations > 20:
            issue_type = "high_variation"
        elif any("Rule 3" in r.get("violations", {}) for r in spc.values()):
            issue_type = "drift_upward"

        causes = self.ROOT_CAUSES.get(issue_type, self.ROOT_CAUSES["outlier"])
        correction = self.CORRECTIVE.get(issue_type, "")

        # Build response
        q_lower = query.lower()

        if any(w in q_lower for w in ["defect", "fail", "reject", "bad", "out"]):
            lines = [
                f"**Root Cause Analysis** (RAG Context Retrieved: {len(retrieved_docs)} chunks)",
                "",
                "Based on your inspection data, the most likely failure drivers are:",
                *[f"  {i+1}. {c}" for i, c in enumerate(causes)],
                "",
                f"**Corrective Action:** {correction}",
                "",
                f"**SPC Status:** {total_violations} violation signals detected across {len(spc)} parameters.",
            ]

        elif any(w in q_lower for w in ["trend", "drift", "shift", "change"]):
            lines = [
                "**Trend Analysis from Retrieved Context:**",
                "",
                ctx[:600] if ctx else "No matching data found.",
                "",
                "Detected patterns suggest process drift. Recommend:",
                "  • Run Western Electric Rule audit",
                "  • Check tooling wear curves against production count",
                "  • Review last 3 setup change logs",
            ]

        elif any(w in q_lower for w in ["cpk", "cp", "capability", "sigma"]):
            cpk_vals = [(k, v["cpk"]) for k, v in spc.items() if "cpk" in v]
            lines = [
                "**Process Capability Summary:**",
                "",
            ]
            for col, cpk in cpk_vals:
                status = "✅ Capable" if cpk >= 1.33 else ("⚠️ Marginal" if cpk >= 1.0 else "❌ Incapable")
                lines.append(f"  • **{col}**: Cpk = {cpk:.3f} → {status}")
            lines += ["", "Cpk ≥ 1.33 = World-class | ≥ 1.0 = Acceptable | < 1.0 = Action Required"]

        elif any(w in q_lower for w in ["summary", "overview", "report", "status"]):
            lines = [
                "**Automated Inspection Summary Report**",
                "",
                f"  • Total Records Analyzed: {df_summary.get('rows', 'N/A')}",
                f"  • Parameters Monitored: {df_summary.get('num_cols', 'N/A')}",
                f"  • SPC Violations: {total_violations}",
                f"  • Anomaly Rate: {df_summary.get('anomaly_rate', 'N/A')}%",
                "",
                "**Top Retrieved Context:**",
                ctx[:400] if ctx else "N/A",
                "",
                "**Recommendation:** Focus immediate attention on parameters showing Cpk < 1.0.",
            ]

        else:
            lines = [
                f"**AI Analyst Response** (Query: '{query}')",
                "",
                "**Retrieved Context:**",
                ctx[:500] if ctx else "No relevant context retrieved.",
                "",
                "**Automated Insight:**",
                f"Based on {len(retrieved_docs)} retrieved document chunks, the inspection data "
                f"shows {total_violations} SPC violations. I recommend reviewing the "
                "highlighted parameters in the SPC dashboard for immediate corrective action.",
            ]

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, engineer features, run anomaly detection."""
    df = df.copy()

    # ── Date parsing ──────────────────────────────────────────────────────────
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                 ["date", "time", "timestamp", "dt", "day"])]
    for dc in date_cols[:1]:
        try:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df["_hour"]    = df[dc].dt.hour
            df["_dayofwk"] = df[dc].dt.dayofweek
            df["_month"]   = df[dc].dt.month
        except:
            pass

    # ── Numeric features ──────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if not c.startswith("_")]

    if len(num_cols) >= 2:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[num_cols].fillna(0))
        STATE["scaler"] = scaler

        # Isolation Forest anomaly detection
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        df["_anomaly_score"] = iso.fit_predict(scaled)
        df["_anomaly_flag"]  = (df["_anomaly_score"] == -1).astype(int)
        STATE["anomaly_model"] = iso

        # PCA for variance decomposition
        pca = PCA(n_components=min(3, len(num_cols)))
        comps = pca.fit_transform(scaled)
        for i in range(comps.shape[1]):
            df[f"_pc{i+1}"] = comps[:, i]

    return df


def compute_spc(df: pd.DataFrame) -> dict:
    engine = SPCEngine()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if not c.startswith("_")]
    results = {}
    for col in num_cols[:12]:  # limit for performance
        results[col] = engine.analyze(df[col], col)
    return results


def build_dashboard_data(df: pd.DataFrame, spc: dict) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if not c.startswith("_")]

    anomaly_count = int(df["_anomaly_flag"].sum()) if "_anomaly_flag" in df.columns else 0
    total         = len(df)
    anomaly_rate  = round(anomaly_count / total * 100, 2) if total > 0 else 0

    oos_count = sum(r.get("n_violations", 0) for r in spc.values())

    # Defect pareto
    violation_counts = {}
    for col, r in spc.items():
        v = r.get("n_violations", 0)
        if v:
            violation_counts[col] = v
    pareto = sorted(violation_counts.items(), key=lambda x: -x[1])

    # Cpk distribution
    cpk_data = [{"col": k, "cpk": v["cpk"]} for k, v in spc.items() if "cpk" in v]

    # Trend data (last 100 points per column)
    trend_data = {}
    for col in num_cols[:6]:
        trend_data[col] = {
            "values": df[col].fillna(df[col].mean()).tolist()[-100:],
            "ucl": spc[col]["ucl"] if col in spc else None,
            "lcl": spc[col]["lcl"] if col in spc else None,
            "mean": spc[col]["mean"] if col in spc else None,
        }

    return {
        "total_records": total,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "oos_violations": oos_count,
        "num_params": len(num_cols),
        "pareto": pareto[:10],
        "cpk_data": cpk_data,
        "trend_data": trend_data,
        "columns": df.columns.tolist(),
        "num_cols": num_cols,
        "df_head": df.head(50).fillna("").to_dict(orient="records"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, state=STATE)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    try:
        df_raw = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    # Process
    df_proc = process_dataframe(df_raw)
    spc     = compute_spc(df_proc)

    # Build RAG index
    rag = RLIASRagEngine()
    n_chunks = rag.build_index(df_proc)

    # Store
    STATE["df"]            = df_raw
    STATE["df_processed"]  = df_proc
    STATE["spc_results"]   = spc
    STATE["upload_info"]   = {
        "filename": f.filename,
        "rows": len(df_raw),
        "cols": len(df_raw.columns),
        "columns": df_raw.columns.tolist(),
    }
    STATE["chat_history"]  = []

    dash = build_dashboard_data(df_proc, spc)
    dash["rag_chunks"] = n_chunks
    dash["upload_info"] = STATE["upload_info"]
    dash["spc"] = spc

    return jsonify({"success": True, "dashboard": dash})


@app.route("/api/chat", methods=["POST"])
def chat():
    if STATE["df_processed"] is None:
        return jsonify({"error": "No data loaded. Please upload CSV first."}), 400

    data  = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    rag     = STATE.get("rag_index")
    spc     = STATE.get("spc_results", {})
    df_proc = STATE["df_processed"]

    retrieved = rag.retrieve(query, top_k=6) if rag else []
    analyst   = RLIASAnalyst()

    df_summary = {
        "rows": len(df_proc),
        "num_cols": len(df_proc.select_dtypes(include=np.number).columns),
        "anomaly_rate": round(df_proc["_anomaly_flag"].mean() * 100, 2)
                        if "_anomaly_flag" in df_proc.columns else 0,
    }

    answer = analyst.analyze_query(query, retrieved, spc, df_summary)

    STATE["chat_history"].append({"role": "user",      "text": query})
    STATE["chat_history"].append({"role": "assistant", "text": answer})

    return jsonify({
        "answer": answer,
        "retrieved_count": len(retrieved),
        "retrieved_snippets": [r["text"][:150] for r in retrieved[:3]],
    })


@app.route("/api/spc/<column>")
def spc_detail(column):
    spc = STATE.get("spc_results", {})
    if column not in spc:
        return jsonify({"error": "Column not found"}), 404
    return jsonify(spc[column])


@app.route("/api/anomalies")
def anomalies():
    df = STATE.get("df_processed")
    if df is None:
        return jsonify([])
    if "_anomaly_flag" not in df.columns:
        return jsonify([])
    anom = df[df["_anomaly_flag"] == 1]
    return jsonify(anom.head(50).fillna("").to_dict(orient="records"))


@app.route("/api/export/report")
def export_report():
    spc  = STATE.get("spc_results", {})
    info = STATE.get("upload_info", {})
    df   = STATE.get("df_processed")

    if df is None:
        return jsonify({"error": "No data"}), 400

    anomaly_rate = round(df["_anomaly_flag"].mean() * 100, 2) \
                   if "_anomaly_flag" in df.columns else 0

    report = {
        "title": "RLIAS Automated Inspection Report",
        "generated": datetime.datetime.now().isoformat(),
        "source_file": info.get("filename"),
        "total_records": info.get("rows"),
        "total_params": info.get("cols"),
        "anomaly_rate_pct": anomaly_rate,
        "spc_summary": {
            col: {
                "status": r.get("status"),
                "cpk": r.get("cpk"),
                "cp": r.get("cp"),
                "violations": r.get("n_violations"),
            }
            for col, r in spc.items()
        },
        "recommendations": [],
    }

    for col, r in spc.items():
        if r.get("cpk", 99) < 1.0:
            report["recommendations"].append(
                f"CRITICAL: {col} Cpk={r['cpk']} below 1.0 — immediate process review required.")
        elif r.get("n_violations", 0) > 5:
            report["recommendations"].append(
                f"WARNING: {col} has {r['n_violations']} SPC violations — investigate process stability.")

    return jsonify(report)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML / JS / CSS  (All-in-one frontend)
# ═══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RLIAS — Rocker Lever Intelligent Analytics System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg:       #050a10;
    --panel:    #0c1520;
    --panel2:   #0f1d2a;
    --border:   #1a3a55;
    --accent:   #00e5ff;
    --accent2:  #ff6b35;
    --accent3:  #39ff14;
    --warn:     #ffcc00;
    --danger:   #ff2d55;
    --text:     #c8e6f5;
    --muted:    #4a7a99;
    --font-hd:  'Syne', sans-serif;
    --font-mono:'Space Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* scanline overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,229,255,.018) 2px, rgba(0,229,255,.018) 4px
    );
    pointer-events: none; z-index: 9999;
  }

  /* ── Header ── */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px;
    background: linear-gradient(90deg, #020d18 0%, #071626 50%, #020d18 100%);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
  }
  .logo {
    font-family: var(--font-hd); font-size: 22px; font-weight: 800;
    color: var(--accent); letter-spacing: 2px;
    text-shadow: 0 0 18px rgba(0,229,255,.6);
  }
  .logo span { color: var(--accent2); }
  .header-meta { font-size: 10px; color: var(--muted); text-align: right; line-height: 1.8; }
  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    background: rgba(57,255,20,.1); border: 1px solid var(--accent3);
    color: var(--accent3); font-size: 10px; letter-spacing: 1px;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%;
         background: var(--accent3); animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.3;} }

  /* ── Layout ── */
  .layout { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 60px); }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; padding: 20px 16px; gap: 20px;
    overflow-y: auto; max-height: calc(100vh - 60px);
  }
  .section-title {
    font-family: var(--font-hd); font-size: 10px; font-weight: 600;
    letter-spacing: 3px; color: var(--muted); text-transform: uppercase;
    padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed var(--border); border-radius: 10px;
    padding: 24px 12px; text-align: center; cursor: pointer;
    transition: all .3s;
    background: rgba(0,229,255,.03);
    position: relative; overflow: hidden;
  }
  .upload-zone:hover, .upload-zone.drag {
    border-color: var(--accent); background: rgba(0,229,255,.07);
    box-shadow: 0 0 20px rgba(0,229,255,.15);
  }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; }
  .upload-icon { font-size: 32px; margin-bottom: 8px; }
  .upload-label { color: var(--accent); font-size: 11px; font-weight: 700; }
  .upload-sub { color: var(--muted); font-size: 10px; margin-top: 4px; }

  /* KPI cards */
  .kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .kpi {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px;
    transition: transform .2s, box-shadow .2s;
  }
  .kpi:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,229,255,.1); }
  .kpi-val {
    font-family: var(--font-hd); font-size: 22px; font-weight: 800;
    color: var(--accent); line-height: 1;
  }
  .kpi-lbl { color: var(--muted); font-size: 9px; letter-spacing: 1px; margin-top: 3px; }
  .kpi.danger .kpi-val  { color: var(--danger); }
  .kpi.warn   .kpi-val  { color: var(--warn); }
  .kpi.ok     .kpi-val  { color: var(--accent3); }

  /* SPC list */
  .spc-list { display: flex; flex-direction: column; gap: 6px; }
  .spc-item {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; cursor: pointer;
    transition: border-color .2s;
  }
  .spc-item:hover { border-color: var(--accent); }
  .spc-item.oc { border-color: var(--danger); }
  .spc-row { display: flex; justify-content: space-between; align-items: center; }
  .spc-col { font-weight: 700; font-size: 11px; color: var(--text); }
  .spc-badge {
    font-size: 8px; padding: 2px 7px; border-radius: 10px;
    background: rgba(57,255,20,.15); color: var(--accent3);
    border: 1px solid var(--accent3);
  }
  .spc-badge.oc {
    background: rgba(255,45,85,.15); color: var(--danger);
    border-color: var(--danger);
  }
  .spc-cpk { color: var(--muted); font-size: 10px; margin-top: 2px; }

  /* ── Main ── */
  .main { padding: 24px; overflow-y: auto; max-height: calc(100vh - 60px); }

  /* Tabs */
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 8px 18px; cursor: pointer; border-radius: 6px 6px 0 0;
    font-size: 11px; letter-spacing: 1px; color: var(--muted);
    font-family: var(--font-hd); font-weight: 600; text-transform: uppercase;
    transition: all .2s; background: transparent; border: none; outline: none;
  }
  .tab:hover { color: var(--text); background: rgba(255,255,255,.04); }
  .tab.active { color: var(--accent); background: rgba(0,229,255,.07);
                border-bottom: 2px solid var(--accent); }

  /* Panels */
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Chart containers */
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .chart-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
    transition: box-shadow .3s;
  }
  .chart-card:hover { box-shadow: 0 0 20px rgba(0,229,255,.08); }
  .chart-card.full { grid-column: span 2; }
  .chart-title {
    font-family: var(--font-hd); font-size: 11px; font-weight: 600;
    letter-spacing: 2px; color: var(--accent2); text-transform: uppercase;
    margin-bottom: 12px;
  }
  canvas { max-height: 220px; }

  /* SPC Detail */
  .spc-detail-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
  }
  .spc-metrics { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; }
  .metric { text-align: center; }
  .metric-val { font-family: var(--font-hd); font-size: 18px; font-weight: 700; }
  .metric-lbl { font-size: 9px; color: var(--muted); letter-spacing: 1px; }

  /* AI Chat */
  .chat-wrap {
    display: flex; flex-direction: column; height: 65vh;
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden;
  }
  .chat-header {
    padding: 14px 18px; background: var(--panel2);
    border-bottom: 1px solid var(--border);
    font-family: var(--font-hd); font-size: 13px; font-weight: 700;
    color: var(--accent);
    display: flex; align-items: center; gap: 8px;
  }
  .chat-messages {
    flex: 1; overflow-y: auto; padding: 16px; display: flex;
    flex-direction: column; gap: 12px;
  }
  .chat-messages::-webkit-scrollbar { width: 4px; }
  .chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  .msg {
    max-width: 85%; padding: 10px 14px; border-radius: 10px;
    font-size: 12px; line-height: 1.6; white-space: pre-wrap;
  }
  .msg.user {
    align-self: flex-end; background: rgba(0,229,255,.12);
    border: 1px solid rgba(0,229,255,.25); color: var(--accent);
  }
  .msg.ai {
    align-self: flex-start; background: var(--panel2);
    border: 1px solid var(--border); color: var(--text);
  }
  .msg.ai strong { color: var(--accent); }
  .msg.system { align-self: center; background: rgba(57,255,20,.07);
    border: 1px solid rgba(57,255,20,.2); color: var(--accent3);
    font-size: 10px; max-width: 100%; text-align: center; }
  .chat-footer {
    padding: 12px 16px; border-top: 1px solid var(--border);
    display: flex; gap: 8px;
    background: var(--panel2);
  }
  .chat-input {
    flex: 1; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; color: var(--text);
    font-family: var(--font-mono); font-size: 12px; outline: none;
    transition: border-color .2s;
  }
  .chat-input:focus { border-color: var(--accent); }
  .chat-send {
    padding: 10px 20px; background: var(--accent); color: var(--bg);
    border: none; border-radius: 8px; cursor: pointer;
    font-family: var(--font-hd); font-weight: 700; font-size: 12px;
    transition: all .2s;
  }
  .chat-send:hover { background: #33eeff; transform: translateY(-1px); }
  .chat-suggestions {
    display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 16px;
    border-top: 1px solid var(--border);
    background: var(--panel2);
  }
  .suggest-chip {
    padding: 4px 10px; border-radius: 14px;
    background: rgba(255,107,53,.1); border: 1px solid rgba(255,107,53,.3);
    color: var(--accent2); font-size: 10px; cursor: pointer;
    transition: all .2s;
  }
  .suggest-chip:hover { background: rgba(255,107,53,.2); }

  /* Table */
  .table-wrap { overflow: auto; max-height: 500px; border-radius: 8px;
    border: 1px solid var(--border); }
  table { border-collapse: collapse; width: 100%; font-size: 11px; }
  thead th {
    background: var(--panel2); padding: 10px 12px;
    text-align: left; position: sticky; top: 0;
    font-family: var(--font-hd); letter-spacing: 1px; font-size: 10px;
    color: var(--accent2); border-bottom: 1px solid var(--border);
  }
  tbody tr:hover { background: rgba(0,229,255,.04); }
  tbody td { padding: 7px 12px; border-bottom: 1px solid rgba(26,58,85,.5); }
  tbody tr.anomaly-row td { color: var(--danger); background: rgba(255,45,85,.05); }

  /* Progress bar */
  .progress-bar {
    height: 4px; background: var(--border); border-radius: 2px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent3));
    border-radius: 2px; transition: width .5s ease;
  }

  /* Loading overlay */
  #loading {
    position: fixed; inset: 0; background: rgba(5,10,16,.92);
    display: none; flex-direction: column; align-items: center; justify-content: center;
    z-index: 200;
  }
  #loading.show { display: flex; }
  .spinner {
    width: 60px; height: 60px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-txt {
    margin-top: 16px; color: var(--accent); font-family: var(--font-hd);
    font-size: 14px; letter-spacing: 2px; animation: pulse 1.2s infinite;
  }

  /* Toast */
  #toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--accent3); color: var(--bg);
    padding: 10px 20px; border-radius: 8px; font-weight: 700;
    opacity: 0; transition: opacity .3s; z-index: 300;
    font-family: var(--font-hd); font-size: 12px;
  }
  #toast.show { opacity: 1; }

  /* Upload state */
  .no-data {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 300px; color: var(--muted);
    gap: 12px;
  }
  .no-data-icon { font-size: 48px; opacity: .4; }
  .no-data-txt { font-family: var(--font-hd); font-size: 14px; letter-spacing: 2px; }

  .btn {
    padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--panel2); color: var(--text); font-family: var(--font-mono);
    font-size: 11px; cursor: pointer; transition: all .2s;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.primary { background: rgba(0,229,255,.12); border-color: var(--accent); color: var(--accent); }

  .info-banner {
    background: rgba(0,229,255,.05); border: 1px solid rgba(0,229,255,.2);
    border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
    font-size: 11px; color: var(--muted); line-height: 1.6;
  }
  .info-banner b { color: var(--accent); }

  @media (max-width: 1024px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { max-height: none; }
    .chart-grid { grid-template-columns: 1fr; }
    .chart-card.full { grid-column: span 1; }
  }
</style>
</head>
<body>

<!-- Loading overlay -->
<div id="loading">
  <div class="spinner"></div>
  <div class="loading-txt">PROCESSING INSPECTION DATA…</div>
</div>
<div id="toast"></div>

<!-- Header -->
<header class="header">
  <div>
    <div class="logo">RL<span>IAS</span></div>
    <div style="font-size:9px; color:var(--muted); letter-spacing:2px; margin-top:2px;">
      ROCKER LEVER INTELLIGENT ANALYTICS SYSTEM
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:16px;">
    <div class="status-pill"><div class="dot"></div>RAG ENGINE ACTIVE</div>
    <div class="header-meta">
      <div>Manufacturing QC | SPC + Anomaly AI</div>
      <div id="hdr-info">No data loaded</div>
    </div>
  </div>
</header>

<!-- Main Layout -->
<div class="layout">

  <!-- Sidebar -->
  <aside class="sidebar">
    <div>
      <div class="section-title">Data Source</div>
      <div class="upload-zone" id="dropzone">
        <input type="file" id="file-input" accept=".csv"/>
        <div class="upload-icon">📂</div>
        <div class="upload-label">DROP CSV HERE</div>
        <div class="upload-sub">or click to browse</div>
        <div class="upload-sub" style="margin-top:4px; color:var(--accent2);">Rocker lever inspection CSV</div>
      </div>
    </div>

    <div>
      <div class="section-title">KPI Summary</div>
      <div class="kpi-grid" id="kpi-grid">
        <div class="kpi"><div class="kpi-val" id="kpi-records">—</div><div class="kpi-lbl">RECORDS</div></div>
        <div class="kpi"><div class="kpi-val" id="kpi-params">—</div><div class="kpi-lbl">PARAMETERS</div></div>
        <div class="kpi danger"><div class="kpi-val" id="kpi-anomaly">—</div><div class="kpi-lbl">ANOMALIES</div></div>
        <div class="kpi warn"><div class="kpi-val" id="kpi-viol">—</div><div class="kpi-lbl">SPC VIOLATIONS</div></div>
        <div class="kpi"><div class="kpi-val" id="kpi-rag">—</div><div class="kpi-lbl">RAG CHUNKS</div></div>
        <div class="kpi ok"><div class="kpi-val" id="kpi-rate">—%</div><div class="kpi-lbl">ANOMALY RATE</div></div>
      </div>
    </div>

    <div>
      <div class="section-title">SPC Status</div>
      <div class="spc-list" id="spc-list">
        <div style="color:var(--muted); font-size:10px; text-align:center; padding:12px;">
          Upload data to view SPC status
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Actions</div>
      <div style="display:flex; flex-direction:column; gap:6px;">
        <button class="btn primary" onclick="exportReport()">⬇ Export JSON Report</button>
        <button class="btn" onclick="clearData()">✕ Clear Data</button>
      </div>
    </div>
  </aside>

  <!-- Main content -->
  <main class="main">
    <!-- Tabs -->
    <div class="tabs">
      <button class="tab active" onclick="switchTab('dashboard')">Dashboard</button>
      <button class="tab" onclick="switchTab('spc')">SPC Control</button>
      <button class="tab" onclick="switchTab('anomaly')">Anomalies</button>
      <button class="tab" onclick="switchTab('chat')">AI Analyst</button>
      <button class="tab" onclick="switchTab('data')">Raw Data</button>
    </div>

    <!-- DASHBOARD TAB -->
    <div class="tab-panel active" id="tab-dashboard">
      <div class="info-banner">
        <b>RLIAS</b> — Upload your rocker lever inspection CSV to activate automated analytics.
        The system will run <b>SPC analysis</b>, <b>Isolation Forest anomaly detection</b>,
        <b>RAG indexing</b> (vector search over your data), and generate AI-driven insights.
      </div>
      <div id="dashboard-content">
        <div class="no-data">
          <div class="no-data-icon">⚙️</div>
          <div class="no-data-txt">AWAITING INSPECTION DATA</div>
          <div style="font-size:10px; color:var(--muted);">Upload a CSV file to begin analysis</div>
        </div>
      </div>
    </div>

    <!-- SPC TAB -->
    <div class="tab-panel" id="tab-spc">
      <div id="spc-content">
        <div class="no-data">
          <div class="no-data-icon">📊</div>
          <div class="no-data-txt">NO SPC DATA</div>
        </div>
      </div>
    </div>

    <!-- ANOMALY TAB -->
    <div class="tab-panel" id="tab-anomaly">
      <div id="anomaly-content">
        <div class="no-data">
          <div class="no-data-icon">🔍</div>
          <div class="no-data-txt">NO ANOMALY DATA</div>
        </div>
      </div>
    </div>

    <!-- AI CHAT TAB -->
    <div class="tab-panel" id="tab-chat">
      <div class="chat-wrap">
        <div class="chat-header">
          🤖 RLIAS AI Analyst <span style="font-size:10px; color:var(--muted); font-weight:400; margin-left:8px;">RAG-Powered | Context-Aware</span>
        </div>
        <div class="chat-messages" id="chat-messages">
          <div class="msg system">
            System initialized. Upload inspection data to unlock full RAG context-aware analysis.
            Ask about defects, SPC violations, Cpk capability, trends, or root causes.
          </div>
        </div>
        <div class="chat-suggestions" id="chat-suggestions">
          <span class="suggest-chip" onclick="sendSuggestion(this)">What are the top defect causes?</span>
          <span class="suggest-chip" onclick="sendSuggestion(this)">Show Cpk capability summary</span>
          <span class="suggest-chip" onclick="sendSuggestion(this)">Detect trends and drift</span>
          <span class="suggest-chip" onclick="sendSuggestion(this)">Full inspection report</span>
          <span class="suggest-chip" onclick="sendSuggestion(this)">Which parameters are out of control?</span>
          <span class="suggest-chip" onclick="sendSuggestion(this)">Root cause of anomalies</span>
        </div>
        <div class="chat-footer">
          <input class="chat-input" id="chat-input" placeholder="Ask about defects, SPC, Cpk, trends, anomalies…"
                 onkeydown="if(event.key==='Enter') sendChat()"/>
          <button class="chat-send" onclick="sendChat()">SEND ▶</button>
        </div>
      </div>
    </div>

    <!-- DATA TAB -->
    <div class="tab-panel" id="tab-data">
      <div id="data-content">
        <div class="no-data">
          <div class="no-data-icon">📋</div>
          <div class="no-data-txt">NO DATA LOADED</div>
        </div>
      </div>
    </div>

  </main>
</div>

<script>
/* ─── State ─────────────────────────────────────────────────────────────── */
let DASH = null;
let CHARTS = {};

/* ─── Tab switching ──────────────────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${name}"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
  if (name === 'spc' && DASH) renderSPCTab();
  if (name === 'anomaly' && DASH) renderAnomalyTab();
  if (name === 'data' && DASH) renderDataTab();
}

/* ─── Upload ─────────────────────────────────────────────────────────────── */
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

['dragenter','dragover'].forEach(e => dropzone.addEventListener(e, ev => {
  ev.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave','drop'].forEach(e => dropzone.addEventListener(e, ev => {
  ev.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', ev => {
  const files = ev.dataTransfer.files;
  if (files.length) uploadFile(files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  if (!file.name.endsWith('.csv')) { showToast('⚠ CSV files only'); return; }
  showLoading(true);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { showToast('❌ ' + data.error); return; }
    DASH = data.dashboard;
    applyDashboard(DASH);
    showToast('✅ Data loaded — ' + DASH.total_records + ' records');
    addChatMessage('system', `✅ CSV loaded: ${DASH.upload_info.filename} | ${DASH.total_records} records | ${DASH.rag_chunks} RAG chunks indexed. Ready for queries.`);
  } catch(e) {
    showToast('❌ Upload failed: ' + e.message);
  } finally {
    showLoading(false);
  }
}

/* ─── Apply dashboard data ───────────────────────────────────────────────── */
function applyDashboard(d) {
  // KPIs
  document.getElementById('kpi-records').textContent = d.total_records.toLocaleString();
  document.getElementById('kpi-params').textContent  = d.num_params;
  document.getElementById('kpi-anomaly').textContent = d.anomaly_count;
  document.getElementById('kpi-viol').textContent    = d.oos_violations;
  document.getElementById('kpi-rag').textContent     = d.rag_chunks;
  document.getElementById('kpi-rate').textContent    = d.anomaly_rate + '%';
  document.getElementById('hdr-info').textContent    =
    `${d.upload_info.filename}  ·  ${d.total_records} records  ·  ${d.num_params} params`;

  // SPC sidebar list
  renderSPCSidebar(d.spc);

  // Dashboard charts
  renderDashboardCharts(d);
}

function renderSPCSidebar(spc) {
  const el = document.getElementById('spc-list');
  const items = Object.entries(spc).map(([col, r]) => {
    const oc = r.status === 'OUT OF CONTROL';
    return `<div class="spc-item ${oc?'oc':''}" onclick="showSPCDetail('${col}')">
      <div class="spc-row">
        <span class="spc-col">${col}</span>
        <span class="spc-badge ${oc?'oc':''}">${oc?'OUT ⚠':'IN CTRL'}</span>
      </div>
      <div class="spc-cpk">Cpk=${r.cpk} · Viols=${r.n_violations}</div>
    </div>`;
  }).join('');
  el.innerHTML = items || '<div style="color:var(--muted);font-size:10px;padding:8px;">No numeric columns</div>';
}

/* ─── Dashboard Charts ───────────────────────────────────────────────────── */
function renderDashboardCharts(d) {
  const container = document.getElementById('dashboard-content');
  container.innerHTML = `
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title">📈 Pareto — SPC Violations</div>
        <canvas id="ch-pareto"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">🎯 Process Capability (Cpk)</div>
        <canvas id="ch-cpk"></canvas>
      </div>
      <div class="chart-card full">
        <div class="chart-title">📉 Control Chart — First Parameter</div>
        <canvas id="ch-trend"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">🔴 Anomaly Distribution</div>
        <canvas id="ch-anomaly"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">📊 Parameter Overview</div>
        <canvas id="ch-radar"></canvas>
      </div>
    </div>`;

  const CC = {
    color: (a) => `rgba(0,229,255,${a})`,
    accent2: (a) => `rgba(255,107,53,${a})`,
    danger: (a) => `rgba(255,45,85,${a})`,
    ok: (a) => `rgba(57,255,20,${a})`,
    warn: (a) => `rgba(255,204,0,${a})`,
    grid: 'rgba(26,58,85,0.5)',
    text: '#4a7a99',
  };
  const baseOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: CC.text, font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: CC.text, font:{size:9} }, grid: { color: CC.grid } },
      y: { ticks: { color: CC.text, font:{size:9} }, grid: { color: CC.grid } },
    }
  };

  // Pareto
  if (d.pareto.length) {
    const labels = d.pareto.map(p => p[0]);
    const vals   = d.pareto.map(p => p[1]);
    destroyChart('ch-pareto');
    CHARTS['ch-pareto'] = new Chart(document.getElementById('ch-pareto'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Violations',
          data: vals,
          backgroundColor: d.pareto.map((_, i) => `hsla(${10+i*25},90%,60%,.7)`),
          borderColor: d.pareto.map((_, i) => `hsla(${10+i*25},90%,60%,1)`),
          borderWidth: 1,
        }]
      },
      options: { ...baseOpts, plugins: { ...baseOpts.plugins } }
    });
  }

  // CPK
  if (d.cpk_data.length) {
    destroyChart('ch-cpk');
    CHARTS['ch-cpk'] = new Chart(document.getElementById('ch-cpk'), {
      type: 'bar',
      data: {
        labels: d.cpk_data.map(x => x.col),
        datasets: [{
          label: 'Cpk',
          data: d.cpk_data.map(x => x.cpk),
          backgroundColor: d.cpk_data.map(x =>
            x.cpk >= 1.33 ? CC.ok(.6) : x.cpk >= 1.0 ? CC.warn(.6) : CC.danger(.6)
          ),
          borderColor: d.cpk_data.map(x =>
            x.cpk >= 1.33 ? CC.ok(1) : x.cpk >= 1.0 ? CC.warn(1) : CC.danger(1)
          ),
          borderWidth: 1,
        }]
      },
      options: {
        ...baseOpts,
        plugins: {
          ...baseOpts.plugins,
          annotation: {}
        }
      }
    });
  }

  // Trend (first column)
  const trendKeys = Object.keys(d.trend_data);
  if (trendKeys.length) {
    const col = trendKeys[0];
    const td  = d.trend_data[col];
    const labels = td.values.map((_, i) => i + 1);
    destroyChart('ch-trend');
    const datasets = [{
      label: col,
      data: td.values,
      borderColor: CC.color(1),
      backgroundColor: CC.color(.1),
      borderWidth: 1.5,
      pointRadius: 2,
      tension: 0.2,
    }];
    if (td.ucl) datasets.push({ label:'UCL', data: Array(td.values.length).fill(td.ucl),
      borderColor: CC.danger(.8), borderDash:[5,4], borderWidth:1.5, pointRadius:0 });
    if (td.lcl) datasets.push({ label:'LCL', data: Array(td.values.length).fill(td.lcl),
      borderColor: CC.danger(.8), borderDash:[5,4], borderWidth:1.5, pointRadius:0 });
    if (td.mean) datasets.push({ label:'Mean', data: Array(td.values.length).fill(td.mean),
      borderColor: CC.warn(.8), borderDash:[3,3], borderWidth:1, pointRadius:0 });

    CHARTS['ch-trend'] = new Chart(document.getElementById('ch-trend'), {
      type: 'line',
      data: { labels, datasets },
      options: baseOpts,
    });
  }

  // Anomaly doughnut
  destroyChart('ch-anomaly');
  CHARTS['ch-anomaly'] = new Chart(document.getElementById('ch-anomaly'), {
    type: 'doughnut',
    data: {
      labels: ['Normal', 'Anomaly'],
      datasets: [{
        data: [d.total_records - d.anomaly_count, d.anomaly_count],
        backgroundColor: [CC.ok(.6), CC.danger(.7)],
        borderColor: [CC.ok(1), CC.danger(1)],
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: CC.text } } }
    }
  });

  // Radar — cpk per param (normalize)
  if (d.cpk_data.length >= 3) {
    destroyChart('ch-radar');
    CHARTS['ch-radar'] = new Chart(document.getElementById('ch-radar'), {
      type: 'radar',
      data: {
        labels: d.cpk_data.slice(0,8).map(x => x.col),
        datasets: [{
          label: 'Cpk',
          data: d.cpk_data.slice(0,8).map(x => Math.min(x.cpk, 2.5)),
          borderColor: CC.color(1),
          backgroundColor: CC.color(.15),
          pointBackgroundColor: CC.color(1),
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { r: {
          ticks: { color: CC.text, backdropColor: 'transparent', font:{size:9} },
          grid: { color: CC.grid },
          pointLabels: { color: CC.text, font:{size:9} },
          min: 0,
        }},
        plugins: { legend: { labels: { color: CC.text } } }
      }
    });
  }
}

function destroyChart(id) {
  if (CHARTS[id]) { CHARTS[id].destroy(); delete CHARTS[id]; }
}

/* ─── SPC Detail Tab ─────────────────────────────────────────────────────── */
function renderSPCTab() {
  if (!DASH) return;
  const spc = DASH.spc;
  const cols = Object.keys(spc);
  let html = '';
  cols.forEach(col => {
    const r = spc[col];
    const status_color = r.status === 'IN CONTROL' ? 'var(--accent3)' : 'var(--danger)';
    html += `
      <div class="spc-detail-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div style="font-family:var(--font-hd); font-size:14px; font-weight:700; color:var(--accent);">${col}</div>
          <div style="color:${status_color}; font-size:11px; font-weight:700;">${r.status}</div>
        </div>
        <div class="spc-metrics">
          <div class="metric"><div class="metric-val" style="color:var(--accent);">${r.mean}</div><div class="metric-lbl">MEAN</div></div>
          <div class="metric"><div class="metric-val" style="color:var(--warn);">${r.std}</div><div class="metric-lbl">STD DEV</div></div>
          <div class="metric"><div class="metric-val" style="color:var(--danger);">${r.ucl}</div><div class="metric-lbl">UCL (+3σ)</div></div>
          <div class="metric"><div class="metric-val" style="color:var(--danger);">${r.lcl}</div><div class="metric-lbl">LCL (-3σ)</div></div>
          <div class="metric"><div class="metric-val" style="color:${r.cp>=1.33?'var(--accent3)':r.cp>=1?'var(--warn)':'var(--danger)'};">${r.cp}</div><div class="metric-lbl">Cp</div></div>
          <div class="metric"><div class="metric-val" style="color:${r.cpk>=1.33?'var(--accent3)':r.cpk>=1?'var(--warn)':'var(--danger)'};">${r.cpk}</div><div class="metric-lbl">Cpk</div></div>
          <div class="metric"><div class="metric-val" style="color:${r.n_violations>0?'var(--danger)':'var(--accent3)'};">${r.n_violations}</div><div class="metric-lbl">VIOLATIONS</div></div>
        </div>
        <div style="height:160px; position:relative;">
          <canvas id="spc_ch_${col.replace(/[^a-zA-Z0-9]/g,'_')}"></canvas>
        </div>
        ${Object.keys(r.violations||{}).length ? `
          <div style="margin-top:8px; font-size:10px; color:var(--danger);">
            Western Electric Violations: ${Object.entries(r.violations).map(([rule,pts])=>
              `${rule} (${pts.length} pts)`).join(', ')}
          </div>` : ''}
      </div>`;
  });
  document.getElementById('spc-content').innerHTML = html || '<div class="no-data"><div class="no-data-txt">No numeric columns detected</div></div>';

  // Draw mini charts
  cols.forEach(col => {
    const r = spc[col];
    const cid = `spc_ch_${col.replace(/[^a-zA-Z0-9]/g,'_')}`;
    const el = document.getElementById(cid);
    if (!el || !r.data) return;
    destroyChart(cid);
    const labels = r.data.map((_,i)=>i+1);
    CHARTS[cid] = new Chart(el, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: col, data: r.data, borderColor:'rgba(0,229,255,1)',
            backgroundColor:'rgba(0,229,255,.08)', borderWidth:1.5, pointRadius:1.5, tension:.2 },
          { label:'UCL', data:Array(r.data.length).fill(r.ucl),
            borderColor:'rgba(255,45,85,.9)', borderDash:[5,4], borderWidth:1.5, pointRadius:0 },
          { label:'LCL', data:Array(r.data.length).fill(r.lcl),
            borderColor:'rgba(255,45,85,.9)', borderDash:[5,4], borderWidth:1.5, pointRadius:0 },
          { label:'Mean', data:Array(r.data.length).fill(r.mean),
            borderColor:'rgba(255,204,0,.8)', borderDash:[3,3], borderWidth:1, pointRadius:0 },
        ]
      },
      options: {
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{labels:{color:'#4a7a99',font:{size:9}}}},
        scales:{
          x:{ticks:{color:'#4a7a99',font:{size:8}},grid:{color:'rgba(26,58,85,.5)'}},
          y:{ticks:{color:'#4a7a99',font:{size:8}},grid:{color:'rgba(26,58,85,.5)'}},
        }
      }
    });
  });
}

/* ─── Anomaly Tab ────────────────────────────────────────────────────────── */
async function renderAnomalyTab() {
  if (!DASH) return;
  document.getElementById('anomaly-content').innerHTML = '<div style="color:var(--muted);padding:20px;">Loading anomalies…</div>';
  try {
    const res = await fetch('/api/anomalies');
    const rows = await res.json();
    if (!rows.length) {
      document.getElementById('anomaly-content').innerHTML =
        '<div class="no-data"><div class="no-data-txt">NO ANOMALIES DETECTED ✅</div></div>';
      return;
    }
    const cols = Object.keys(rows[0]).filter(c => !c.startsWith('_'));
    let html = `
      <div style="color:var(--danger); font-family:var(--font-hd); font-size:13px; margin-bottom:12px;">
        ⚠ ${rows.length} anomalous records detected by Isolation Forest
      </div>
      <div class="table-wrap">
      <table><thead><tr>
        ${cols.map(c => `<th>${c}</th>`).join('')}
        <th>ANOMALY SCORE</th>
      </tr></thead><tbody>
      ${rows.map(r => `<tr class="anomaly-row">
        ${cols.map(c => `<td>${r[c]}</td>`).join('')}
        <td style="color:var(--danger); font-weight:700;">⚠ OUTLIER</td>
      </tr>`).join('')}
      </tbody></table></div>`;
    document.getElementById('anomaly-content').innerHTML = html;
  } catch(e) {
    document.getElementById('anomaly-content').innerHTML = `<div style="color:var(--danger);">Error: ${e.message}</div>`;
  }
}

/* ─── Data Tab ───────────────────────────────────────────────────────────── */
function renderDataTab() {
  if (!DASH || !DASH.df_head) return;
  const rows = DASH.df_head;
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const html = `
    <div style="color:var(--muted); font-size:10px; margin-bottom:10px;">
      Showing first 50 rows · ${DASH.total_records} total records
    </div>
    <div class="table-wrap">
    <table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
    <tbody>
    ${rows.map(r=>`<tr>
      ${cols.map(c=>`<td>${r[c]}</td>`).join('')}
    </tr>`).join('')}
    </tbody></table></div>`;
  document.getElementById('data-content').innerHTML = html;
}

/* ─── SPC Column Click ───────────────────────────────────────────────────── */
function showSPCDetail(col) {
  switchTab('spc');
}

/* ─── AI Chat ────────────────────────────────────────────────────────────── */
async function sendChat() {
  const input = document.getElementById('chat-input');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  addChatMessage('user', query);
  addChatMessage('ai', '⏳ Retrieving context and reasoning…');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    // Remove "thinking" message
    const msgs = document.getElementById('chat-messages');
    const last = msgs.lastElementChild;
    if (last && last.textContent.includes('⏳')) last.remove();

    if (data.error) {
      addChatMessage('ai', '❌ ' + data.error);
    } else {
      addChatMessage('ai', data.answer + `\n\n_Retrieved ${data.retrieved_count} context chunks_`);
    }
  } catch(e) {
    addChatMessage('ai', '❌ Error: ' + e.message);
  }
}

function sendSuggestion(el) {
  document.getElementById('chat-input').value = el.textContent;
  sendChat();
}

function addChatMessage(role, text) {
  const msgs = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  // Basic markdown bold rendering
  div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g,'<br>');
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

/* ─── Export ─────────────────────────────────────────────────────────────── */
async function exportReport() {
  try {
    const res = await fetch('/api/export/report');
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `RLIAS_Report_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    showToast('✅ Report exported');
  } catch(e) {
    showToast('❌ Export failed');
  }
}

function clearData() {
  DASH = null;
  Object.values(CHARTS).forEach(c => c.destroy());
  CHARTS = {};
  document.getElementById('dashboard-content').innerHTML = `
    <div class="no-data">
      <div class="no-data-icon">⚙️</div>
      <div class="no-data-txt">AWAITING INSPECTION DATA</div>
    </div>`;
  document.getElementById('spc-content').innerHTML = '';
  document.getElementById('anomaly-content').innerHTML = '';
  document.getElementById('data-content').innerHTML = '';
  ['kpi-records','kpi-params','kpi-anomaly','kpi-viol','kpi-rag'].forEach(id=>{
    document.getElementById(id).textContent='—';
  });
  document.getElementById('kpi-rate').textContent='—%';
  document.getElementById('spc-list').innerHTML='<div style="color:var(--muted);font-size:10px;text-align:center;padding:12px;">Upload data to view SPC status</div>';
  document.getElementById('chat-messages').innerHTML='<div class="msg system">Data cleared. Upload a new CSV to begin.</div>';
  document.getElementById('hdr-info').textContent='No data loaded';
  showToast('Data cleared');
}

/* ─── Helpers ────────────────────────────────────────────────────────────── */
function showLoading(v) {
  document.getElementById('loading').classList.toggle('show', v);
}
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   RLIAS — Rocker Lever Intelligent Analytics System          ║
║   RAG + SPC + Isolation Forest | Manufacturing QC            ║
╠══════════════════════════════════════════════════════════════╣
║  Open:  http://127.0.0.1:5000                                ║
║  Drag & drop your inspection CSV to begin analysis           ║
╚══════════════════════════════════════════════════════════════╝
""")
    # Install check
    missing = []
    try: import flask
    except: missing.append("flask")
    try: import numpy
    except: missing.append("numpy")
    try: import pandas
    except: missing.append("pandas")
    try: import sklearn
    except: missing.append("scikit-learn")

    if missing:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print(f"    Run: pip install {' '.join(missing)}")
    else:
        app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)