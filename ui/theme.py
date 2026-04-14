from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #eef3f6;
            --surface: rgba(255, 255, 255, 0.94);
            --surface-strong: #ffffff;
            --surface-muted: #f6f9fb;
            --border: rgba(25, 50, 74, 0.12);
            --text-main: #122033;
            --text-muted: #5e7187;
            --accent: #1d5f7a;
            --accent-soft: #dbeff5;
            --critical: #a7343f;
            --critical-soft: #fae9eb;
            --warning: #9a6615;
            --warning-soft: #fff2dc;
            --safe: #206f52;
            --safe-soft: #e6f5ed;
            --shadow: 0 18px 44px rgba(21, 39, 60, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 95, 122, 0.10), transparent 22%),
                linear-gradient(180deg, #f8fbfc 0%, var(--bg) 100%);
            color: var(--text-main);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #0f2234 0%, #173246 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f3f7fb;
        }

        .shell-card,
        .metric-card,
        .banner-card,
        .triage-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        .page-header {
            padding: 1.7rem 1.8rem;
            background:
                linear-gradient(135deg, rgba(15, 34, 52, 0.98), rgba(29, 95, 122, 0.92));
            color: #f8fbff;
            border-radius: 24px;
            box-shadow: 0 24px 54px rgba(14, 28, 46, 0.18);
        }

        .page-header-kicker {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            opacity: 0.72;
            margin-bottom: 0.7rem;
        }

        .page-header-title {
            font-size: 2.1rem;
            line-height: 1.08;
            margin: 0;
            font-weight: 700;
        }

        .page-header-copy {
            margin-top: 0.85rem;
            max-width: 50rem;
            line-height: 1.65;
            color: rgba(248, 251, 255, 0.86);
            font-size: 0.98rem;
        }

        .page-header-meta {
            margin-top: 1rem;
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .meta-pill {
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            font-size: 0.82rem;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.16);
        }

        .shell-card {
            padding: 1.25rem 1.3rem;
            margin-top: 0.95rem;
        }

        .section-title {
            margin: 0;
            color: var(--text-main);
            font-size: 1.03rem;
            font-weight: 700;
        }

        .section-copy {
            margin: 0.42rem 0 0 0;
            color: var(--text-muted);
            line-height: 1.58;
            font-size: 0.94rem;
        }

        .metric-card {
            padding: 1rem 1.05rem;
            min-height: 132px;
        }

        .metric-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
        }

        .metric-value {
            margin-top: 0.52rem;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text-main);
            word-break: break-word;
        }

        .metric-subtext {
            margin-top: 0.48rem;
            color: var(--text-muted);
            line-height: 1.45;
            font-size: 0.9rem;
        }

        .banner-card {
            margin-top: 1rem;
            padding: 1rem 1.1rem;
            border-left: 6px solid transparent;
        }

        .banner-card h3 {
            margin: 0;
            font-size: 1rem;
        }

        .banner-card p {
            margin: 0.45rem 0 0 0;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .banner-critical {
            border-left-color: var(--critical);
            background: linear-gradient(0deg, var(--critical-soft), var(--critical-soft)), var(--surface);
        }

        .banner-warning {
            border-left-color: var(--warning);
            background: linear-gradient(0deg, var(--warning-soft), var(--warning-soft)), var(--surface);
        }

        .banner-safe {
            border-left-color: var(--safe);
            background: linear-gradient(0deg, var(--safe-soft), var(--safe-soft)), var(--surface);
        }

        .triage-card {
            padding: 1.12rem 1.18rem;
            min-height: 164px;
            border-top: 5px solid transparent;
        }

        .triage-critical {
            border-top-color: var(--critical);
        }

        .triage-warning {
            border-top-color: var(--warning);
        }

        .triage-safe {
            border-top-color: var(--safe);
        }

        .triage-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
        }

        .triage-value {
            margin-top: 0.55rem;
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .triage-copy {
            margin-top: 0.55rem;
            color: var(--text-muted);
            line-height: 1.5;
            font-size: 0.92rem;
        }

        .intake-note,
        .empty-state {
            border-radius: 18px;
            border: 1px dashed rgba(18, 32, 51, 0.18);
            padding: 1.1rem 1.15rem;
            background: rgba(255, 255, 255, 0.7);
            color: var(--text-muted);
        }

        .sidebar-heading {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
        }

        .sidebar-copy {
            margin: 0 0 1rem 0;
            color: rgba(243, 247, 251, 0.82);
            line-height: 1.55;
        }

        @media (max-width: 768px) {
            .page-header {
                padding: 1.3rem;
            }

            .page-header-title {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

