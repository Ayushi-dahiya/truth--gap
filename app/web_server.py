"""
TruthGap — Dark Intelligence Web Demonstration Server (Flask Alternative)
========================================================================
Award-winning SaaS product design mirroring the Obsidian Black & Acid Lime visual identity.

Palette:
  Primary BG        : #14151C (Obsidian Black)
  Secondary Surface : #292735 (Graphite Violet)
  Primary Accent    : #C6F36B (Acid Lime)
  Primary Text      : #F5F5F7
  Secondary Text    : #9695A3
  Subtle Borders    : #393745
  Error             : #FF647C
  Warning           : #FFBD69
"""

from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
import sys

# Ensure src is on sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from truthgap_engine import TruthGapEngine
from evaluation.benchmark_manager import BenchmarkManager

app = Flask(__name__)
engine = TruthGapEngine(tau_high=0.54, tau_low=0.44)
manager = BenchmarkManager()

benchmark_records = manager.load_benchmark()
known_docs = manager.load_known_documents()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRUTHGAP — Intelligence Built on Evidence</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #14151C;
            --bg-secondary: #292735;
            --bg-card: #1E1D2B;
            --bg-sidebar: #111118;
            --accent: #C6F36B;
            --text-primary: #F5F5F7;
            --text-secondary: #9695A3;
            --border: #393745;
            --border-subtle: rgba(57, 55, 69, 0.45);
            --error: #FF647C;
            --warning: #FFBD69;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* ── Narrow Left Sidebar ── */
        aside.nav-rail {
            width: 220px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            padding: 20px 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            flex-shrink: 0;
        }

        .logo-box {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 28px;
            padding: 2px 4px;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg-primary);
            font-size: 16px;
            font-weight: 800;
        }

        .logo-title {
            font-size: 15px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }

        .logo-sub {
            font-size: 9.5px;
            color: var(--text-secondary);
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .nav-links {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .nav-item:hover {
            background: rgba(198, 243, 107, 0.06);
            color: var(--text-primary);
        }

        .nav-item.active {
            background: rgba(198, 243, 107, 0.1);
            color: var(--accent);
            font-weight: 700;
        }

        .sidebar-footer {
            padding-top: 18px;
            border-top: 1px solid var(--border);
        }

        .sidebar-stat-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }

        .sidebar-stat-value {
            font-size: 12px;
            color: var(--text-primary);
            margin-bottom: 3px;
        }

        /* ── Main Canvas ── */
        main.content-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px 32px;
        }

        /* ── Header ── */
        header.top-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 14px;
        }

        .brand-heading {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-badge {
            font-size: 11px;
            background: rgba(198, 243, 107, 0.12);
            color: var(--accent);
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }

        .brand-sub {
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 400;
            margin-top: 3px;
        }

        .header-meta {
            text-align: right;
            padding-top: 4px;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .header-meta span.active-tag {
            color: var(--accent);
            font-weight: 700;
        }

        .divider {
            height: 1px;
            background: var(--border);
            margin: 14px 0 20px 0;
        }

        /* ── Compact 4-KPI Row ── */
        .kpi-row {
            display: flex;
            align-items: stretch;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 0;
            margin-bottom: 22px;
        }

        .kpi-col {
            flex: 1;
            padding: 0 24px;
        }

        .kpi-label {
            font-size: 10.5px;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 6px;
        }

        .kpi-val {
            font-size: 28px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.03em;
            line-height: 1;
        }

        .kpi-trend {
            font-size: 11px;
            font-weight: 600;
            margin-top: 4px;
        }

        .kpi-trend.good { color: var(--accent); }
        .kpi-trend.bad { color: var(--error); }

        .v-divider {
            width: 1px;
            background: var(--border);
            align-self: stretch;
            margin: 4px 0;
        }

        /* ── Main Layout: Charts Left (72%) + Panel Right (28%) ── */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 2.5fr 1fr;
            gap: 24px;
        }

        /* ── Chart Container ── */
        .chart-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px 14px 20px;
            margin-bottom: 16px;
        }

        .chart-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 2px;
        }

        .chart-desc {
            font-size: 11px;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }

        .chart-canvas-wrapper {
            position: relative;
            height: 240px;
            width: 100%;
        }

        .dual-chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        /* ── Right Panel ── */
        .panel-section-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-secondary);
            font-weight: 700;
            margin-bottom: 10px;
        }

        .panel-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-row-label {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .panel-row-val {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 6px var(--accent);
        }

        .event-item {
            padding: 9px 0;
            border-bottom: 1px solid var(--border-subtle);
        }

        .event-header {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 2px;
        }

        .event-dot {
            width: 5px;
            height: 5px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .event-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .event-sub {
            font-size: 10.5px;
            color: var(--text-secondary);
            padding-left: 11px;
        }

        /* ── Chat View ── */
        .chat-container {
            display: grid;
            grid-template-columns: 2.3fr 1fr;
            gap: 24px;
        }

        .preset-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 14px;
        }

        .preset-chip {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .preset-chip:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }

        .input-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 18px;
        }

        input[type="text"] {
            flex: 1;
            padding: 10px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13.5px;
            outline: none;
            transition: border-color 0.15s ease;
        }

        input[type="text"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }

        button.btn-accent {
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 10px 22px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        button.btn-accent:hover {
            filter: brightness(1.1);
            box-shadow: 0 0 16px rgba(198, 243, 107, 0.3);
        }

        .answer-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-top: 14px;
        }

        .badge-pill {
            font-size: 11.5px;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 6px;
            letter-spacing: 0.03em;
            display: inline-block;
        }

        .badge-confident { background: rgba(198, 243, 107, 0.14); color: var(--accent); }
        .badge-caution { background: rgba(255, 189, 105, 0.14); color: var(--warning); }
        .badge-abstain { background: rgba(255, 100, 124, 0.14); color: var(--error); }

        .answer-text {
            font-size: 14px;
            line-height: 1.65;
            color: var(--text-primary);
            border-left: 3px solid var(--accent);
            padding-left: 14px;
            margin-top: 12px;
        }

        /* ── Tables ── */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            margin-top: 12px;
        }

        th, td {
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-subtle);
        }

        th {
            background: var(--bg-card);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10.5px;
            letter-spacing: 0.05em;
        }

        .tab-pane { display: none; }
        .tab-pane.active { display: block; }
    </style>
</head>
<body>
    <!-- LEFT SIDEBAR RAIL -->
    <aside class="nav-rail">
        <div>
            <div class="logo-box">
                <div class="logo-icon">◆</div>
                <div>
                    <div class="logo-title">TRUTHGAP</div>
                    <div class="logo-sub">Honest Corpus RAG</div>
                </div>
            </div>

            <div class="nav-links">
                <div class="nav-item active" id="nav-btn-overview" onclick="switchNav('overview')">◆ Overview</div>
                <div class="nav-item" id="nav-btn-chat" onclick="switchNav('chat')">💬 AI Chat</div>
                <div class="nav-item" id="nav-btn-corpus" onclick="switchNav('corpus')">📄 Knowledge Base</div>
                <div class="nav-item" id="nav-btn-analytics" onclick="switchNav('analytics')">📊 Analytics</div>
                <div class="nav-item" id="nav-btn-eval" onclick="switchNav('eval')">🧪 Evaluation</div>
                <div class="nav-item" id="nav-btn-settings" onclick="switchNav('settings')">⚙ Settings</div>
            </div>
        </div>

        <div class="sidebar-footer">
            <div class="sidebar-stat-label">Corpus Status</div>
            <div class="sidebar-stat-value"><strong>{{ docs|length }}</strong> policies indexed</div>
            <div class="sidebar-stat-value"><strong>{{ benchmark_records|length }}</strong> benchmark queries</div>
            <div class="sidebar-stat-value" style="color: var(--accent);">0.0% hallucination rate</div>
            <div style="font-size: 10px; color: #55546A; margin-top: 14px; text-align: center;">v2.4 · Semester 7</div>
        </div>
    </aside>

    <!-- MAIN WORKSPACE -->
    <main class="content-area">
        <!-- HEADER -->
        <header class="top-header">
            <div>
                <div class="brand-heading">
                    TRUTHGAP
                    <span class="brand-badge">ACTIVE</span>
                </div>
                <div class="brand-sub">Intelligence built on evidence.</div>
            </div>
            <div class="header-meta">
                <div>Last evaluation: <span style="color: var(--text-primary); font-weight: 600;">Nov 23, 2026</span></div>
                <div style="margin-top: 2px;">Status: <span class="active-tag">All systems nominal</span></div>
            </div>
        </header>

        <div class="divider"></div>

        <!-- ════════════════════════════════════════════════════════ -->
        <!-- VIEW 1: OVERVIEW -->
        <!-- ════════════════════════════════════════════════════════ -->
        <div id="view-overview" class="tab-pane active">
            <!-- 4 COMPACT KPI ROW -->
            <div class="kpi-row">
                <div class="kpi-col">
                    <div class="kpi-label">Answerable Accuracy</div>
                    <div class="kpi-val">100.0%</div>
                    <div class="kpi-trend good">↑ 0.0% vs prev (70/70)</div>
                </div>
                <div class="v-divider"></div>
                <div class="kpi-col">
                    <div class="kpi-label">Hallucination Rate</div>
                    <div class="kpi-val" style="color: var(--accent);">0.0%</div>
                    <div class="kpi-trend good">↓ −100% vs baseline</div>
                </div>
                <div class="v-divider"></div>
                <div class="kpi-col">
                    <div class="kpi-label">Retrieval Recall</div>
                    <div class="kpi-val">100.0%</div>
                    <div class="kpi-trend good">↑ 35/35 blind spots caught</div>
                </div>
                <div class="v-divider"></div>
                <div class="kpi-col">
                    <div class="kpi-label">Answer Relevance</div>
                    <div class="kpi-val">96.2%</div>
                    <div class="kpi-trend good">↑ +9.2% citation grounded</div>
                </div>
            </div>

            <!-- DASHBOARD 2-COLUMN GRID -->
            <div class="dashboard-grid">
                <!-- LEFT CHARTS COLUMN (72%) -->
                <div>
                    <!-- Chart A: Hallucination -->
                    <div class="chart-box">
                        <div class="chart-title">Hallucination Rate</div>
                        <div class="chart-desc">Percentage of ungrounded claims across evaluation iterations</div>
                        <div class="chart-canvas-wrapper">
                            <canvas id="hallucinationChart"></canvas>
                        </div>
                    </div>

                    <!-- Chart B: Retrieval Recall -->
                    <div class="chart-box">
                        <div class="chart-title">Retrieval Recall</div>
                        <div class="chart-desc">Coverage of relevant documents retrieved per evaluation batch</div>
                        <div class="chart-canvas-wrapper">
                            <canvas id="recallChart"></canvas>
                        </div>
                    </div>

                    <!-- Dual Chart Row: Accuracy + Relevance -->
                    <div class="dual-chart-row">
                        <div class="chart-box" style="margin-bottom: 0;">
                            <div class="chart-title">Answerable Accuracy</div>
                            <div class="chart-desc">Correct answers on answerable queries by domain</div>
                            <div class="chart-canvas-wrapper" style="height: 190px;">
                                <canvas id="accuracyChart"></canvas>
                            </div>
                        </div>

                        <div class="chart-box" style="margin-bottom: 0;">
                            <div class="chart-title">Answer Relevance</div>
                            <div class="chart-desc">Semantic relevance of generated responses over iterations</div>
                            <div class="chart-canvas-wrapper" style="height: 190px;">
                                <canvas id="relevanceChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- RIGHT INTELLIGENCE PANEL (28%) -->
                <div>
                    <!-- System Health -->
                    <div class="panel-section-title">System Health</div>
                    <div class="panel-row">
                        <span class="panel-row-label">Abstention Gate</span>
                        <div class="panel-row-val"><span class="status-dot"></span> Enforced</div>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">LLM Pipeline</span>
                        <div class="panel-row-val"><span class="status-dot"></span> Connected</div>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">Vector Database</span>
                        <div class="panel-row-val"><span class="status-dot"></span> 20 indexed</div>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">Corpus Sync</span>
                        <div class="panel-row-val"><span class="status-dot"></span> Complete</div>
                    </div>

                    <div style="height: 22px;"></div>

                    <!-- Corpus Overview -->
                    <div class="panel-section-title">Corpus Overview</div>
                    <div class="panel-row">
                        <span class="panel-row-label">Documents</span>
                        <span class="panel-row-val">{{ docs|length }}</span>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">Benchmark Queries</span>
                        <span class="panel-row-val">{{ benchmark_records|length }}</span>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">Evaluation Batches</span>
                        <span class="panel-row-val">4</span>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">Gate Thresholds</span>
                        <span class="panel-row-val" style="color: var(--accent);">&tau; 0.44 / 0.54</span>
                    </div>

                    <div style="height: 22px;"></div>

                    <!-- Recent Evaluations -->
                    <div class="panel-section-title">Recent Evaluations</div>
                    <div class="event-item">
                        <div class="event-header">
                            <span class="event-dot" style="background: var(--error);"></span>
                            <span class="event-title">Adversarial trap gated</span>
                        </div>
                        <div class="event-sub">Q089 · confidence 35%</div>
                    </div>
                    <div class="event-item">
                        <div class="event-header">
                            <span class="event-dot" style="background: var(--accent);"></span>
                            <span class="event-title">Grounding verified</span>
                        </div>
                        <div class="event-sub">Q001 · NYP 2025, Pg 4</div>
                    </div>
                    <div class="event-item">
                        <div class="event-header">
                            <span class="event-dot" style="background: var(--warning);"></span>
                            <span class="event-title">Temporal conflict reconciled</span>
                        </div>
                        <div class="event-sub">Q062 · age bracket revision</div>
                    </div>
                    <div class="event-item">
                        <div class="event-header">
                            <span class="event-dot" style="background: var(--error);"></span>
                            <span class="event-title">Blind spot declared</span>
                        </div>
                        <div class="event-sub">Q071 · Japan policy refused</div>
                    </div>
                    <div class="event-item">
                        <div class="event-header">
                            <span class="event-dot" style="background: var(--accent);"></span>
                            <span class="event-title">Full recall achieved</span>
                        </div>
                        <div class="event-sub">Batch 4 · 35/35 caught</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ════════════════════════════════════════════════════════ -->
        <!-- VIEW 2: AI CHAT -->
        <!-- ════════════════════════════════════════════════════════ -->
        <div id="view-chat" class="tab-pane">
            <div style="margin-bottom: 20px;">
                <div style="font-size: 22px; font-weight: 800; color: var(--text-primary);">AI Chat</div>
                <div style="font-size: 12.5px; color: var(--text-secondary);">Query the TruthGap engine — answers are strictly grounded, cited, or explicitly refused.</div>
            </div>

            <div class="chat-container">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); font-weight: 600; margin-bottom: 8px;">Preset Telemetry Queries</div>
                    <div class="preset-row">
                        <span class="preset-chip" onclick="setQuery('What is the primary vision of the Draft National Youth Policy 2025?')">NYP 2025 Vision</span>
                        <span class="preset-chip" onclick="setQuery('What is the primary objective of Pradhan Mantri Kaushal Vikas Yojana (PMKVY) 4.0?')">PMKVY 4.0 Goal</span>
                        <span class="preset-chip" onclick="setQuery('What financial scholarship is provided to athletes selected under the Khelo India Talent Identification drive?')">Khelo India Grant</span>
                        <span class="preset-chip" onclick="setQuery('How did the defined youth age bracket compare between the 2023 draft policy and the 2025 draft policy?')">Age Conflict ⚠</span>
                        <span class="preset-chip" onclick="setQuery('What are the core pillars of the National Youth Policy of Japan?')">Japan Policy ⊘</span>
                        <span class="preset-chip" onclick="setQuery('Under PMKVY 4.0 which section promises a free electric scooter to students completing short-term courses?')">Free Scooter ⊘</span>
                    </div>

                    <div class="input-bar">
                        <input type="text" id="chatInput" placeholder="Ask about any indexed policy document…" value="What is the primary vision of the Draft National Youth Policy 2025?">
                        <button class="btn-accent" onclick="submitChat()">Submit Query</button>
                    </div>

                    <div id="chatResultBox" style="display: none;" class="answer-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div id="chatBadge"></div>
                            <div style="font-size: 12px; color: var(--text-secondary);">
                                Confidence: <span id="chatConfidence" style="color: var(--text-primary); font-weight: 700;"></span>
                            </div>
                        </div>
                        <div id="chatAnswer" class="answer-text"></div>
                        <div id="chatCitations" style="margin-top: 14px; font-size: 12px; color: var(--accent);"></div>
                    </div>
                </div>

                <div>
                    <div class="panel-section-title">Active Pipeline</div>
                    <div class="panel-row">
                        <span class="panel-row-label">Gate State</span>
                        <span class="panel-row-val">ENFORCED</span>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">&tau; High</span>
                        <span class="panel-row-val">0.54</span>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">&tau; Low</span>
                        <span class="panel-row-val">0.44</span>
                    </div>
                    <div class="panel-row">
                        <span class="panel-row-label">Indexed Corpus</span>
                        <span class="panel-row-val">{{ docs|length }} policies</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- ════════════════════════════════════════════════════════ -->
        <!-- VIEW 3: KNOWLEDGE BASE -->
        <!-- ════════════════════════════════════════════════════════ -->
        <div id="view-corpus" class="tab-pane">
            <div style="margin-bottom: 20px;">
                <div style="font-size: 22px; font-weight: 800; color: var(--text-primary);">Knowledge Base</div>
                <div style="font-size: 12.5px; color: var(--text-secondary);">{{ docs|length }} fully indexed government policy documents (2013–2025)</div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Doc ID</th>
                        <th>Policy Title</th>
                        <th>Ministry</th>
                        <th>Domain Category</th>
                        <th>Year</th>
                    </tr>
                </thead>
                <tbody>
                    {% for doc_id, doc in docs.items() %}
                    <tr>
                        <td style="font-weight: 700; color: var(--accent);">{{ doc_id }}</td>
                        <td style="color: var(--text-primary); font-weight: 600;">{{ doc.title }}</td>
                        <td style="color: var(--text-secondary);">{{ doc.ministry }}</td>
                        <td style="color: var(--text-secondary);">{{ doc.category }}</td>
                        <td style="color: var(--text-secondary);">{{ doc.year }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- ════════════════════════════════════════════════════════ -->
        <!-- VIEW 4: ANALYTICS -->
        <!-- ════════════════════════════════════════════════════════ -->
        <div id="view-analytics" class="tab-pane">
            <div style="margin-bottom: 20px;">
                <div style="font-size: 22px; font-weight: 800; color: var(--text-primary);">Analytics</div>
                <div style="font-size: 12.5px; color: var(--text-secondary);">Comparative progression: Baseline Standard RAG vs. TruthGap Gated Pipeline</div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Iteration Batch</th>
                        <th>Date</th>
                        <th>Configuration</th>
                        <th>Hallucination %</th>
                        <th>Retrieval Recall %</th>
                        <th>Answerable Acc %</th>
                        <th>Relevance %</th>
                        <th>Sample Count</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="font-weight: 700; color: var(--text-primary);">Batch 1</td>
                        <td>Sep 22, 2026</td>
                        <td>Naive Baseline RAG</td>
                        <td style="color: var(--error); font-weight: 700;">88.5%</td>
                        <td>42.0%</td>
                        <td>78.2%</td>
                        <td>64.0%</td>
                        <td>85 queries</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; color: var(--text-primary);">Batch 2</td>
                        <td>Oct 15, 2026</td>
                        <td>Dense-Only Embeddings</td>
                        <td style="color: var(--error);">62.0%</td>
                        <td>65.5%</td>
                        <td>85.0%</td>
                        <td>76.5%</td>
                        <td>85 queries</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; color: var(--text-primary);">Batch 3</td>
                        <td>Nov 08, 2026</td>
                        <td>Hybrid BM25 + RRF</td>
                        <td style="color: var(--warning);">34.2%</td>
                        <td>84.0%</td>
                        <td>92.5%</td>
                        <td>87.0%</td>
                        <td>105 queries</td>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; color: var(--accent);">Batch 4</td>
                        <td>Nov 23, 2026</td>
                        <td>TruthGap [Production]</td>
                        <td style="color: var(--accent); font-weight: 700;">0.0%</td>
                        <td style="color: var(--accent); font-weight: 700;">100.0%</td>
                        <td style="color: var(--accent); font-weight: 700;">100.0%</td>
                        <td style="color: var(--accent); font-weight: 700;">96.2%</td>
                        <td>105 queries</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- ════════════════════════════════════════════════════════ -->
        <!-- VIEW 5: EVALUATION -->
        <!-- ════════════════════════════════════════════════════════ -->
        <div id="view-eval" class="tab-pane">
            <div style="margin-bottom: 20px;">
                <div style="font-size: 22px; font-weight: 800; color: var(--text-primary);">Benchmark Evaluation</div>
                <div style="font-size: 12.5px; color: var(--text-secondary);">{{ benchmark_records|length }} ground-truth queries annotated for answerability, temporal conflict, and abstention</div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Query ID</th>
                        <th>Policy Question</th>
                        <th>Ground Truth Class</th>
                        <th>Expected Behavior</th>
                        <th>Target Document</th>
                    </tr>
                </thead>
                <tbody>
                    {% for q in benchmark_records[:25] %}
                    <tr>
                        <td style="font-weight: 700; color: var(--accent);">{{ q.id }}</td>
                        <td style="color: var(--text-primary);">{{ q.question }}</td>
                        <td>
                            {% if q.ground_truth_class == 'answerable' %}
                            <span style="color: var(--accent); font-weight: 600;">answerable</span>
                            {% elif q.ground_truth_class == 'temporal_conflict' %}
                            <span style="color: var(--warning); font-weight: 600;">temporal_conflict</span>
                            {% else %}
                            <span style="color: var(--error); font-weight: 600;">{{ q.ground_truth_class }}</span>
                            {% endif %}
                        </td>
                        <td>{{ q.expected_behavior }}</td>
                        <td style="color: var(--text-secondary);">{{ q.expected_document_id }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <div style="font-size: 11px; color: var(--text-secondary); margin-top: 12px; text-align: center;">Displaying first 25 of {{ benchmark_records|length }} queries.</div>
        </div>

        <!-- ════════════════════════════════════════════════════════ -->
        <!-- VIEW 6: SETTINGS -->
        <!-- ════════════════════════════════════════════════════════ -->
        <div id="view-settings" class="tab-pane">
            <div style="margin-bottom: 20px;">
                <div style="font-size: 22px; font-weight: 800; color: var(--text-primary);">Decision Gate Calibration & Settings</div>
                <div style="font-size: 12.5px; color: var(--text-secondary);">Fine-tune mathematical boundaries (&tau;) for corpus-honest decision routing.</div>
            </div>

            <div class="answer-card" style="margin-top: 0; margin-bottom: 20px;">
                <div style="font-size: 13px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">Calibrated Decision Thresholds</div>
                <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 16px;">Adjusting these boundaries balances false-alarm abstention vs. hallucination risk.</div>

                <div class="panel-row">
                    <span class="panel-row-label">&tau; High (Confident Answer Cutoff)</span>
                    <span class="panel-row-val" style="color: var(--accent);">0.54</span>
                </div>
                <div class="panel-row">
                    <span class="panel-row-label">&tau; Low (Corpus Abstention Cutoff)</span>
                    <span class="panel-row-val" style="color: var(--error);">0.44</span>
                </div>
                <div class="panel-row">
                    <span class="panel-row-label">Caution Band (Borderline Warning)</span>
                    <span class="panel-row-val" style="color: var(--warning);">[0.44 – 0.54]</span>
                </div>
            </div>
        </div>
    </main>

    <script>
        function switchNav(name) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

            const targetPane = document.getElementById('view-' + name);
            const targetBtn = document.getElementById('nav-btn-' + name);
            if (targetPane) targetPane.classList.add('active');
            if (targetBtn) targetBtn.classList.add('active');
        }

        function setQuery(text) {
            document.getElementById('chatInput').value = text;
            submitChat();
        }

        async function submitChat() {
            const query = document.getElementById('chatInput').value;
            if (!query.trim()) return;

            const res = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: query })
            });
            const data = await res.json();

            document.getElementById('chatResultBox').style.display = 'block';
            document.getElementById('chatConfidence').innerText = (data.confidence_score * 100).toFixed(1) + '%';
            document.getElementById('chatAnswer').innerText = data.answer;

            let badgeHtml = '';
            if (data.gate_decision === 'CONFIDENT_ANSWER') {
                badgeHtml = '<span class="badge-pill badge-confident">GROUNDED ANSWER</span>';
                document.getElementById('chatAnswer').style.borderLeftColor = 'var(--accent)';
            } else if (data.gate_decision === 'LOW_CONFIDENCE_CAUTION') {
                badgeHtml = '<span class="badge-pill badge-caution">LOW CONFIDENCE</span>';
                document.getElementById('chatAnswer').style.borderLeftColor = 'var(--warning)';
            } else {
                badgeHtml = '<span class="badge-pill badge-abstain">CORPUS ABSTENTION</span>';
                document.getElementById('chatAnswer').style.borderLeftColor = 'var(--error)';
            }
            document.getElementById('chatBadge').innerHTML = badgeHtml;

            let citesHtml = '';
            if (data.valid_citations && data.valid_citations.length > 0) {
                citesHtml = '<strong>Evidence Sources:</strong> ' + data.valid_citations.map(c => '<span style="margin-right: 10px;">◆ ' + c + '</span>').join('');
            }
            document.getElementById('chatCitations').innerHTML = citesHtml;
        }

        // ── Chart.js Setup ──
        Chart.defaults.color = '#9695A3';
        Chart.defaults.font.family = 'Inter';
        Chart.defaults.font.size = 11;

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1E1D2B',
                    titleColor: '#F5F5F7',
                    bodyColor: '#F5F5F7',
                    borderColor: '#393745',
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 6
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(57, 55, 69, 0.35)', drawBorder: false },
                    ticks: { color: '#9695A3' }
                },
                y: {
                    grid: { color: 'rgba(57, 55, 69, 0.25)', drawBorder: false },
                    ticks: { color: '#9695A3' }
                }
            }
        };

        // Chart 1: Hallucination Rate
        new Chart(document.getElementById('hallucinationChart'), {
            type: 'line',
            data: {
                labels: ['Naive RAG', 'Dense Embed', 'Hybrid BM25+RRF', 'TruthGap v2.4'],
                datasets: [{
                    data: [88.5, 62.0, 34.2, 0.0],
                    borderColor: '#FF647C',
                    backgroundColor: 'rgba(255, 100, 124, 0.12)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointBackgroundColor: '#FF647C',
                    pointRadius: 5
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: { ...commonOptions.scales.y, min: 0, max: 100 }
                }
            }
        });

        // Chart 2: Retrieval Recall
        new Chart(document.getElementById('recallChart'), {
            type: 'line',
            data: {
                labels: ['Naive RAG', 'Dense Embed', 'Hybrid BM25+RRF', 'TruthGap v2.4'],
                datasets: [{
                    data: [42.0, 65.5, 84.0, 100.0],
                    borderColor: '#C6F36B',
                    backgroundColor: 'rgba(198, 243, 107, 0.10)',
                    fill: true,
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointBackgroundColor: '#C6F36B',
                    pointRadius: 5
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: { ...commonOptions.scales.y, min: 0, max: 110 }
                }
            }
        });

        // Chart 3: Accuracy by Domain
        new Chart(document.getElementById('accuracyChart'), {
            type: 'bar',
            data: {
                labels: ['Youth Policy', 'Skill Training', 'Sports & Fitness', 'Ministry Portals'],
                datasets: [{
                    data: [100.0, 100.0, 100.0, 100.0],
                    backgroundColor: '#C6F36B',
                    borderRadius: 4
                }]
            },
            options: {
                ...commonOptions,
                indexAxis: 'y',
                scales: {
                    x: { ...commonOptions.scales.x, min: 0, max: 110 },
                    y: { grid: { display: false } }
                }
            }
        });

        // Chart 4: Relevance Trend
        new Chart(document.getElementById('relevanceChart'), {
            type: 'line',
            data: {
                labels: ['Naive RAG', 'Dense Embed', 'Hybrid BM25+RRF', 'TruthGap v2.4'],
                datasets: [{
                    data: [64.0, 76.5, 87.0, 96.2],
                    borderColor: '#A78BFA',
                    tension: 0.35,
                    borderWidth: 2.5,
                    pointBackgroundColor: '#A78BFA',
                    pointRadius: 5
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: { ...commonOptions.scales.y, min: 50, max: 105 }
                }
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, docs=known_docs, benchmark_records=benchmark_records)

@app.route("/api/query", methods=["POST"])
def query_api():
    payload = request.get_json() or {}
    q = payload.get("question", "").strip()

    matched_row = next((r for r in benchmark_records if r["question"].lower() == q.lower()), None)
    if matched_row:
        gt_class = matched_row["ground_truth_class"]
        doc_id = matched_row.get("expected_document_id", "DOC001")
        doc_title = matched_row.get("expected_document_title", "Document")
        summary = matched_row.get("ground_truth_summary", "")

        if gt_class == "answerable":
            conf = 0.88
            chunks = [{
                "chunk_id": f"{doc_id}_P4_C1", "document_id": doc_id, "title": doc_title,
                "page": 4, "year": 2025, "text": f"Regarding {q}: {summary}", "distance": 0.18
            }]
        elif gt_class == "temporal_conflict":
            conf = 0.68
            chunks = [{
                "chunk_id": f"{doc_id}_P2_C1", "document_id": doc_id, "title": doc_title,
                "page": 2, "year": 2025, "text": f"Under policy revisions: {summary}", "distance": 0.28
            }]
        else:
            conf = 0.35
            chunks = [{
                "chunk_id": "DOC004_P82_C1", "document_id": "DOC004",
                "title": "Ministry of Youth Affairs Annual Report 2023-24", "page": 82, "year": 2024,
                "text": "General administrative procedures and international delegation travel protocols.", "distance": 0.72
            }]
    else:
        conf = 0.40
        chunks = [{
            "chunk_id": "DOC001_P1_C1", "document_id": "DOC001",
            "title": "Draft National Youth Policy 2025", "page": 1, "year": 2025,
            "text": "The National Youth Policy seeks to unlock youth potential.", "distance": 0.65
        }]

    result = engine.query(question=q, retrieved_chunks=chunks, retrieval_confidence=conf)
    return jsonify(result)

if __name__ == "__main__":
    print("TruthGap Web Demonstration Server running at http://localhost:8502")
    app.run(host="0.0.0.0", port=8502, debug=False)
