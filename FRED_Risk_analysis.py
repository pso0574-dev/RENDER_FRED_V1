from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc


# ============================================================
# App / server
# ============================================================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = "FRED Macro Risk Dashboard"


# ============================================================
# Config
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

REQUEST_TIMEOUT = 20
MAX_RETRIES = 4
BACKOFF_SEC = 1.2

DEFAULT_LOOKBACK_YEARS = 5

SERIES_META = {
    "WALCL": {
        "name": "Fed Balance Sheet",
        "category": "Liquidity",
        "unit": "Million USD",
        "desc": "Federal Reserve total assets. Rapid expansion can indicate stress support.",
    },
    "RRPONTSYD": {
        "name": "Reverse Repo",
        "category": "Liquidity",
        "unit": "Billion USD",
        "desc": "Overnight Reverse Repo usage. Reflects liquidity parking / drainage.",
    },
    "WTREGEN": {
        "name": "Treasury General Account",
        "category": "Liquidity",
        "unit": "Million USD",
        "desc": "US Treasury cash balance. Falling TGA can inject liquidity.",
    },
    "DGS10": {
        "name": "10Y Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "desc": "Long-term Treasury yield.",
    },
    "DGS2": {
        "name": "2Y Treasury Yield",
        "category": "Rates",
        "unit": "%",
        "desc": "Short-term Treasury yield.",
    },
    "T10Y2Y": {
        "name": "10Y-2Y Yield Curve",
        "category": "Rates",
        "unit": "%",
        "desc": "Yield curve spread. Prolonged inversion often precedes recession.",
    },
    "BAMLH0A0HYM2": {
        "name": "High Yield Spread",
        "category": "Credit",
        "unit": "%",
        "desc": "High-yield corporate bond spread. Rising spread signals credit stress.",
    },
    "BAMLC0A0CM": {
        "name": "Investment Grade Spread",
        "category": "Credit",
        "unit": "%",
        "desc": "Investment-grade corporate bond spread.",
    },
    "STLFSI4": {
        "name": "Financial Stress Index",
        "category": "Stress",
        "unit": "Index",
        "desc": "Composite financial stress index.",
    },
    "INDPRO": {
        "name": "Industrial Production",
        "category": "Macro",
        "unit": "Index",
        "desc": "Industrial activity proxy.",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "category": "Macro",
        "unit": "%",
        "desc": "US unemployment rate.",
    },
    "DFF": {
        "name": "Effective Fed Funds Rate",
        "category": "Policy",
        "unit": "%",
        "desc": "Effective federal funds rate.",
    },
    "SOFR": {
        "name": "SOFR",
        "category": "Policy",
        "unit": "%",
        "desc": "Secured Overnight Financing Rate.",
    },
    "CPIAUCSL": {
        "name": "CPI",
        "category": "Inflation",
        "unit": "Index",
        "desc": "Consumer Price Index.",
    },
    "DCOILWTICO": {
        "name": "WTI Oil",
        "category": "Inflation",
        "unit": "USD/bbl",
        "desc": "WTI crude oil spot price.",
    },
    "CPFF": {
        "name": "Commercial Paper Funding Facility",
        "category": "Stress",
        "unit": "Index",
        "desc": "Funding stress related series.",
    },
    "TEDRATE": {
        "name": "TED Spread",
        "category": "Stress",
        "unit": "%",
        "desc": "Bank funding stress proxy.",
    },
    "DRBLACBS": {
        "name": "Bank Lending Tightening",
        "category": "Credit",
        "unit": "%",
        "desc": "Banks tightening standards for C&I loans.",
    },
}

DEFAULT_SERIES = [
    "WALCL",
    "RRPONTSYD",
    "WTREGEN",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "STLFSI4",
    "INDPRO",
    "UNRATE",
    "DFF",
    "SOFR",
    "CPIAUCSL",
    "DCOILWTICO",
    "CPFF",
    "TEDRATE",
    "DRBLACBS",
]

KEY_SIGNALS = [
    "BAMLH0A0HYM2",
    "T10Y2Y",
    "STLFSI4",
    "TEDRATE",
    "DRBLACBS",
    "UNRATE",
    "DCOILWTICO",
]


# ============================================================
# Helpers
# ============================================================
def safe_float(x) -> float:
    try:
        if x in [".", None, ""]:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fred_request(params: Dict) -> Dict:
    last_error = None
    for i in range(MAX_RETRIES):
        try:
            res = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            last_error = e
            time.sleep(BACKOFF_SEC * (i + 1))
    raise RuntimeError(f"FRED request failed: {last_error}")


def fetch_fred_series(series_id: str, start_date: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": start_date,
        "sort_order": "asc",
    }
    if FRED_API_KEY:
        params["api_key"] = FRED_API_KEY

    data = fred_request(params)
    obs = data.get("observations", [])

    if not obs:
        return pd.DataFrame(columns=["date", "value"]).set_index("date")

    df = pd.DataFrame(obs)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"].apply(safe_float)
    df = df.set_index("date").sort_index()
    return df


def latest_valid_value(series: pd.Series) -> Tuple[Optional[pd.Timestamp], float]:
    s = series.dropna()
    if s.empty:
        return None, np.nan
    return s.index[-1], float(s.iloc[-1])


def prev_valid_value(series: pd.Series, n: int = 1) -> float:
    s = series.dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1 - n])


def yoy_change(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 13:
        return np.nan
    current = s.iloc[-1]
    past = s.iloc[-13]
    if pd.isna(current) or pd.isna(past) or past == 0:
        return np.nan
    return (current / past - 1.0) * 100.0


def pct_change_recent(series: pd.Series, periods: int = 20) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    prev = s.iloc[-1 - periods]
    curr = s.iloc[-1]
    if pd.isna(prev) or pd.isna(curr) or prev == 0:
        return np.nan
    return (curr / prev - 1.0) * 100.0


def diff_recent(series: pd.Series, periods: int = 20) -> float:
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] - s.iloc[-1 - periods])


def normalize_series(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return series * np.nan
    min_v = s.min()
    max_v = s.max()
    if max_v == min_v:
        return series * 0.0
    return (series - min_v) / (max_v - min_v)


def annualized_inflation_from_cpi(cpi: pd.Series, months: int = 3) -> float:
    s = cpi.dropna()
    if len(s) <= months:
        return np.nan
    recent = s.iloc[-1]
    past = s.iloc[-1 - months]
    if recent <= 0 or past <= 0:
        return np.nan
    return ((recent / past) ** (12 / months) - 1) * 100.0


def classify_signal(series_id: str, value: float) -> Tuple[str, str]:
    if pd.isna(value):
        return "N/A", "No recent data"

    if series_id == "BAMLH0A0HYM2":
        if value >= 6:
            return "High Risk", "Credit stress elevated"
        elif value >= 4:
            return "Watch", "Credit conditions weakening"
        return "Low Risk", "Credit spread contained"

    if series_id == "BAMLC0A0CM":
        if value >= 2.5:
            return "High Risk", "IG spread elevated"
        elif value >= 1.7:
            return "Watch", "IG spread rising"
        return "Low Risk", "IG spread stable"

    if series_id == "T10Y2Y":
        if value < -0.5:
            return "High Risk", "Deep inversion / recession signal"
        elif value < 0:
            return "Watch", "Yield curve inverted"
        return "Low Risk", "Curve normal or steepening"

    if series_id == "STLFSI4":
        if value >= 1.0:
            return "High Risk", "Financial stress elevated"
        elif value >= 0:
            return "Watch", "Stress above normal"
        return "Low Risk", "Stress below average"

    if series_id == "UNRATE":
        if value >= 5.0:
            return "High Risk", "Labor market weakening"
        elif value >= 4.3:
            return "Watch", "Unemployment drifting higher"
        return "Low Risk", "Labor market relatively firm"

    if series_id == "DCOILWTICO":
        if value >= 100:
            return "High Risk", "Oil shock risk"
        elif value >= 80:
            return "Watch", "Inflation pressure from oil"
        return "Low Risk", "Oil not yet shock-level"

    if series_id == "TEDRATE":
        if value >= 1.0:
            return "High Risk", "Funding stress elevated"
        elif value >= 0.5:
            return "Watch", "Funding stress rising"
        return "Low Risk", "Funding stress contained"

    if series_id == "DRBLACBS":
        if value >= 30:
            return "High Risk", "Banks tightening aggressively"
        elif value >= 10:
            return "Watch", "Lending standards tightening"
        return "Low Risk", "Credit standards not severely tight"

    return "Watch", "Check chart and trend"


def risk_color(status: str) -> str:
    if status == "High Risk":
        return "#dc3545"
    if status == "Watch":
        return "#f0ad4e"
    if status == "Low Risk":
        return "#198754"
    return "#6c757d"


def compute_dashboard_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in data.columns:
        series = data[sid]
        dt, latest = latest_valid_value(series)
        prev = prev_valid_value(series, 1)
        chg = latest - prev if pd.notna(latest) and pd.notna(prev) else np.nan
        yoy = yoy_change(series)
        status, note = classify_signal(sid, latest)

        row = {
            "Category": SERIES_META[sid]["category"],
            "Series ID": sid,
            "Indicator": SERIES_META[sid]["name"],
            "Latest Date": dt.date().isoformat() if dt is not None else "",
            "Latest": latest,
            "1-step Change": chg,
            "YoY %": yoy,
            "Status": status,
            "Interpretation": note,
            "Description": SERIES_META[sid]["desc"],
            "3M Annualized %": annualized_inflation_from_cpi(series, 3) if sid == "CPIAUCSL" else np.nan,
            "20-period % Change": pct_change_recent(series, 20) if sid in ["WALCL", "RRPONTSYD", "WTREGEN", "INDPRO", "DCOILWTICO"] else np.nan,
            "20-period Diff": diff_recent(series, 20) if sid in ["BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "TEDRATE", "T10Y2Y", "UNRATE"] else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    cat_order = ["Liquidity", "Rates", "Credit", "Stress", "Macro", "Policy", "Inflation"]
    out["CategoryOrder"] = out["Category"].map({c: i for i, c in enumerate(cat_order)})
    out = out.sort_values(["CategoryOrder", "Indicator"]).drop(columns=["CategoryOrder"])
    return out


def infer_overall_risk(table: pd.DataFrame) -> Tuple[str, str]:
    score = 0
    for sid, weight in [
        ("BAMLH0A0HYM2", 3),
        ("T10Y2Y", 2),
        ("STLFSI4", 3),
        ("TEDRATE", 2),
        ("DRBLACBS", 2),
        ("UNRATE", 2),
        ("DCOILWTICO", 1),
    ]:
        row = table[table["Series ID"] == sid]
        if row.empty:
            continue
        status = row["Status"].iloc[0]
        if status == "High Risk":
            score += 2 * weight
        elif status == "Watch":
            score += 1 * weight

    if score >= 18:
        return "High Risk", "Broad multi-asset stress is building."
    elif score >= 9:
        return "Watch", "Several warning signals are active."
    return "Low Risk", "Risk signals are mixed or contained."


def build_line_chart(df: pd.DataFrame, sid: str) -> go.Figure:
    meta = SERIES_META[sid]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[sid],
            mode="lines",
            name=f"{meta['name']} ({sid})",
        )
    )
    fig.update_layout(
        title=f"{meta['name']} ({sid})",
        height=420,
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis_title="Date",
        yaxis_title=meta["unit"],
        template="plotly_white",
    )
    return fig


def build_normalized_chart(df: pd.DataFrame, sids: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for sid in sids:
        if sid not in df.columns:
            continue
        s = normalize_series(df[sid])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=s,
                mode="lines",
                name=f"{SERIES_META[sid]['name']} ({sid})",
            )
        )
    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis_title="Date",
        yaxis_title="Normalized (0-1)",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


def status_badge(status: str):
    color = risk_color(status)
    return html.Span(
        status,
        style={
            "backgroundColor": color,
            "color": "white",
            "padding": "4px 8px",
            "borderRadius": "8px",
            "fontWeight": "600",
            "fontSize": "12px",
        },
    )


def build_signal_card(name: str, value: float, unit: str, status: str, interpretation: str):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(name, style={"fontSize": "14px", "fontWeight": "700"}),
                html.Div(
                    "N/A" if pd.isna(value) else f"{value:.2f} {unit}",
                    style={"fontSize": "24px", "fontWeight": "800", "marginTop": "8px"},
                ),
                html.Div(status_badge(status), style={"marginTop": "8px"}),
                html.Div(
                    interpretation,
                    style={"fontSize": "12px", "marginTop": "8px", "color": "#666"},
                ),
            ]
        ),
        style={"borderRadius": "16px", "height": "100%"},
        className="shadow-sm",
    )


def load_all_data(lookback_years: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str, str]:
    start_date = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    data_dict = {}
    failed_series = []

    for sid in DEFAULT_SERIES:
        try:
            df_sid = fetch_fred_series(sid, start_date)
            if not df_sid.empty:
                data_dict[sid] = df_sid["value"].rename(sid)
            else:
                failed_series.append(sid)
        except Exception:
            failed_series.append(sid)

    if not data_dict:
        empty = pd.DataFrame()
        return empty, empty, failed_series, "N/A", "No FRED data could be loaded."

    data = pd.concat(data_dict.values(), axis=1).sort_index()

    if "T10Y2Y" not in data.columns and {"DGS10", "DGS2"}.issubset(data.columns):
        data["T10Y2Y"] = data["DGS10"] - data["DGS2"]

    table = compute_dashboard_table(data)
    overall_status, overall_msg = infer_overall_risk(table)
    return data, table, failed_series, overall_status, overall_msg


# ============================================================
# Layout
# ============================================================
app.layout = dbc.Container(
    [
        dcc.Store(id="stored-data"),
        dcc.Store(id="stored-table"),
        dcc.Store(id="stored-failed"),
        dcc.Interval(id="auto-refresh", interval=30 * 60 * 1000, n_intervals=0),

        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("FRED Macro Risk Dashboard", className="fw-bold"),
                        html.Div(
                            "Liquidity / Rates / Credit / Stress / Macro / Inflation",
                            className="text-muted",
                        ),
                    ],
                    xs=12,
                    lg=8,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Controls", className="fw-bold"),
                                html.Br(),
                                html.Label("Lookback Years"),
                                dcc.Slider(
                                    id="lookback-slider",
                                    min=1,
                                    max=20,
                                    step=1,
                                    value=DEFAULT_LOOKBACK_YEARS,
                                    marks={i: str(i) for i in [1, 3, 5, 10, 15, 20]},
                                ),
                                html.Br(),
                                dbc.Button("Refresh Data", id="refresh-btn", color="primary", className="w-100"),
                            ]
                        ),
                        style={"borderRadius": "16px"},
                        className="shadow-sm",
                    ),
                    xs=12,
                    lg=4,
                ),
            ],
            className="g-3",
        ),

        html.Br(),

        dbc.Row(
            [
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("Overall Risk"), html.H3(id="overall-risk")])), xs=6, md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("Loaded Series"), html.H3(id="loaded-series")])), xs=6, md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("Lookback"), html.H3(id="lookback-value")])), xs=6, md=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.Div("Last Data Date"), html.H3(id="last-date")])), xs=6, md=3),
            ],
            className="g-3",
        ),

        html.Br(),
        dbc.Alert(id="overall-message", color="info"),
        html.Div(id="warning-message"),

        html.H4("Key Crisis Signals", className="mt-4 mb-3"),
        dbc.Row(id="signal-cards-row", className="g-3"),

        html.Br(),

        dcc.Tabs(
            id="main-tabs",
            value="snapshot",
            children=[
                dcc.Tab(label="Snapshot Table", value="snapshot"),
                dcc.Tab(label="Risk Dashboard", value="risk"),
                dcc.Tab(label="Normalized View", value="normalized"),
                dcc.Tab(label="Raw Charts", value="charts"),
                dcc.Tab(label="Indicator Guide", value="guide"),
            ],
        ),
        html.Br(),
        html.Div(id="tab-content"),

        html.H4("Quick Interpretation", className="mt-4"),
        html.Ul(id="quick-interpretation"),

        html.Hr(),
        html.Div(
            "Tip: Focus on High Yield Spread, 10Y-2Y curve, Financial Stress Index, TED Spread, "
            "Bank Lending Tightening, and Unemployment trend first.",
            className="text-muted mb-4",
        ),
    ],
    fluid=True,
    style={"maxWidth": "1500px"},
)


# ============================================================
# Data loading callback
# ============================================================
@app.callback(
    Output("stored-data", "data"),
    Output("stored-table", "data"),
    Output("stored-failed", "data"),
    Output("overall-risk", "children"),
    Output("loaded-series", "children"),
    Output("lookback-value", "children"),
    Output("last-date", "children"),
    Output("overall-message", "children"),
    Input("refresh-btn", "n_clicks"),
    Input("auto-refresh", "n_intervals"),
    Input("lookback-slider", "value"),
)
def update_data(_, __, lookback_years):
    data, table, failed_series, overall_status, overall_msg = load_all_data(lookback_years)

    if data.empty:
        return (
            None,
            None,
            failed_series,
            "N/A",
            "0",
            f"{lookback_years}Y",
            "N/A",
            "No FRED data could be loaded.",
        )

    last_date = data.dropna(how="all").index.max().date().isoformat()
    return (
        data.to_json(date_format="iso", orient="split"),
        table.to_json(date_format="iso", orient="split"),
        failed_series,
        overall_status,
        str(data.shape[1]),
        f"{lookback_years}Y",
        last_date,
        overall_msg,
    )


# ============================================================
# Warning message
# ============================================================
@app.callback(
    Output("warning-message", "children"),
    Input("stored-failed", "data"),
)
def update_warning_message(failed_series):
    if failed_series:
        return dbc.Alert(
            "Some series could not be loaded: " + ", ".join(failed_series),
            color="warning",
        )
    return html.Div()


# ============================================================
# Key signal cards
# ============================================================
@app.callback(
    Output("signal-cards-row", "children"),
    Input("stored-table", "data"),
)
def update_signal_cards(table_json):
    if not table_json:
        return []

    table = pd.read_json(table_json, orient="split")
    cards = []

    for sid in KEY_SIGNALS:
        row = table[table["Series ID"] == sid]
        if row.empty:
            cards.append(dbc.Col(build_signal_card(sid, np.nan, "", "N/A", "No data"), xs=12, sm=6, lg=3))
            continue

        latest = row["Latest"].iloc[0]
        status = row["Status"].iloc[0]
        interpretation = row["Interpretation"].iloc[0]
        name = row["Indicator"].iloc[0]
        unit = SERIES_META[sid]["unit"]

        cards.append(
            dbc.Col(
                build_signal_card(name, latest, unit, status, interpretation),
                xs=12, sm=6, lg=3,
            )
        )

    return cards


# ============================================================
# Tab content
# ============================================================
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("stored-data", "data"),
    Input("stored-table", "data"),
)
def render_tab(tab, data_json, table_json):
    if not data_json or not table_json:
        return dbc.Alert("No data loaded yet.", color="secondary")

    data = pd.read_json(data_json, orient="split")
    table = pd.read_json(table_json, orient="split")

    if tab == "snapshot":
        display_cols = [
            "Category", "Series ID", "Indicator", "Latest Date", "Latest",
            "1-step Change", "YoY %", "3M Annualized %", "20-period % Change",
            "20-period Diff", "Status", "Interpretation"
        ]
        df = table[display_cols].copy()
        for col in ["Latest", "1-step Change", "YoY %", "3M Annualized %", "20-period % Change", "20-period Diff"]:
            df[col] = df[col].round(2)

        style_data_conditional = []
        for status, color in [("High Risk", "#dc3545"), ("Watch", "#f0ad4e"), ("Low Risk", "#198754"), ("N/A", "#6c757d")]:
            style_data_conditional.append({
                "if": {"filter_query": f'{{Status}} = "{status}"', "column_id": "Status"},
                "backgroundColor": color,
                "color": "white",
                "fontWeight": "bold",
            })

        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            page_size=20,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px", "whiteSpace": "normal"},
            style_header={"fontWeight": "bold"},
            style_data_conditional=style_data_conditional,
        )

    if tab == "risk":
        children = []
        for category in ["Liquidity", "Rates", "Credit", "Stress", "Macro", "Policy", "Inflation"]:
            sub = table[table["Category"] == category].copy()
            if sub.empty:
                continue

            df = sub[["Indicator", "Series ID", "Latest", "Status", "Interpretation", "Description"]].copy()
            df["Latest"] = df["Latest"].round(2)

            style_data_conditional = []
            for status, color in [("High Risk", "#dc3545"), ("Watch", "#f0ad4e"), ("Low Risk", "#198754"), ("N/A", "#6c757d")]:
                style_data_conditional.append({
                    "if": {"filter_query": f'{{Status}} = "{status}"', "column_id": "Status"},
                    "backgroundColor": color,
                    "color": "white",
                    "fontWeight": "bold",
                })

            children.extend([
                html.H5(category, className="mt-3"),
                dash_table.DataTable(
                    data=df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in df.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "300px", "whiteSpace": "normal"},
                    style_header={"fontWeight": "bold"},
                    style_data_conditional=style_data_conditional,
                ),
            ])
        return html.Div(children)

    if tab == "normalized":
        figures = []
        norm_groups = {
            "Liquidity Comparison": ["WALCL", "RRPONTSYD", "WTREGEN"],
            "Rates Comparison": ["DGS10", "DGS2", "T10Y2Y", "DFF", "SOFR"],
            "Credit and Stress Comparison": ["BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "TEDRATE", "DRBLACBS"],
            "Macro and Inflation Comparison": ["INDPRO", "UNRATE", "CPIAUCSL", "DCOILWTICO"],
        }
        for title, sids in norm_groups.items():
            existing = [sid for sid in sids if sid in data.columns]
            if not existing:
                continue
            fig = build_normalized_chart(data, existing, title)
            figures.append(dcc.Graph(figure=fig))
        return html.Div(figures)

    if tab == "charts":
        options = [{"label": f"{SERIES_META[sid]['name']} ({sid})", "value": sid} for sid in data.columns]
        default_values = [sid for sid in ["BAMLH0A0HYM2", "T10Y2Y", "STLFSI4"] if sid in data.columns]

        return html.Div([
            html.Label("Select indicators"),
            dcc.Dropdown(
                id="chart-series-dropdown",
                options=options,
                value=default_values,
                multi=True,
            ),
            html.Br(),
            html.Div(id="raw-charts-container"),
        ])

    if tab == "guide":
        guide_rows = []
        for sid, meta in SERIES_META.items():
            guide_rows.append({
                "Category": meta["category"],
                "Series ID": sid,
                "Indicator": meta["name"],
                "Description": meta["desc"],
                "What to Check": {
                    "WALCL": "Rapid expansion can signal emergency liquidity support.",
                    "RRPONTSYD": "Large shifts can show liquidity absorption/release changes.",
                    "WTREGEN": "Falling TGA can add liquidity; rising TGA can drain it.",
                    "DGS10": "Sharp declines may reflect recession fears.",
                    "DGS2": "Sensitive to Fed expectations.",
                    "T10Y2Y": "Negative values suggest inversion.",
                    "BAMLH0A0HYM2": "Above 4 watch; above 6 high risk.",
                    "BAMLC0A0CM": "Persistent widening suggests worsening corporate credit.",
                    "STLFSI4": "Above 0 watch; above 1 elevated stress.",
                    "INDPRO": "Look for rolling weakness and negative YoY trend.",
                    "UNRATE": "Rising unemployment often confirms recession pressure.",
                    "DFF": "High policy rate for longer increases refinancing pressure.",
                    "SOFR": "Unexpected jumps may reflect short-term funding pressure.",
                    "CPIAUCSL": "Check YoY and 3M annualized trend.",
                    "DCOILWTICO": "Oil spikes can reignite inflation.",
                    "CPFF": "Useful as short-term funding stress context.",
                    "TEDRATE": "Funding stress tends to rise when bank confidence weakens.",
                    "DRBLACBS": "Rising tightening means credit availability is worsening.",
                }.get(sid, ""),
                "Rule of Thumb": {
                    "BAMLH0A0HYM2": ">= 6 high risk",
                    "T10Y2Y": "< 0 inversion, < -0.5 deep inversion",
                    "STLFSI4": ">= 1 high stress",
                    "UNRATE": ">= 5 labor weakness",
                    "DCOILWTICO": ">= 100 oil shock",
                    "TEDRATE": ">= 1 funding stress",
                    "DRBLACBS": ">= 30 aggressive tightening",
                }.get(sid, "Interpret together with trend"),
            })

        guide_df = pd.DataFrame(guide_rows).sort_values(["Category", "Indicator"])
        return dash_table.DataTable(
            data=guide_df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in guide_df.columns],
            page_size=15,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "140px", "maxWidth": "320px", "whiteSpace": "normal"},
            style_header={"fontWeight": "bold"},
        )

    return html.Div("Select a tab.")


# ============================================================
# Raw charts callback
# ============================================================
@app.callback(
    Output("raw-charts-container", "children"),
    Input("chart-series-dropdown", "value"),
    Input("stored-data", "data"),
    prevent_initial_call=True,
)
def update_raw_charts(selected_sids, data_json):
    if not data_json or not selected_sids:
        return dbc.Alert("Choose at least one indicator.", color="secondary")

    data = pd.read_json(data_json, orient="split")
    if isinstance(selected_sids, str):
        selected_sids = [selected_sids]

    charts = []
    for sid in selected_sids:
        if sid not in data.columns:
            continue
        charts.append(dcc.Graph(figure=build_line_chart(data, sid)))
    return charts


# ============================================================
# Quick interpretation
# ============================================================
@app.callback(
    Output("quick-interpretation", "children"),
    Input("stored-table", "data"),
)
def update_quick_interpretation(table_json):
    if not table_json:
        return [html.Li("No data loaded.")]

    table = pd.read_json(table_json, orient="split")

    def get_val(sid: str) -> float:
        row = table[table["Series ID"] == sid]
        if row.empty:
            return np.nan
        return float(row["Latest"].iloc[0])

    hy = get_val("BAMLH0A0HYM2")
    yc = get_val("T10Y2Y")
    fsi = get_val("STLFSI4")
    ted = get_val("TEDRATE")
    oil = get_val("DCOILWTICO")
    unr = get_val("UNRATE")

    messages = []

    if pd.notna(hy):
        if hy >= 6:
            messages.append("High-yield spreads are in a stressed zone.")
        elif hy >= 4:
            messages.append("High-yield spreads are elevated but not yet full-crisis level.")
        else:
            messages.append("High-yield spreads remain relatively contained.")

    if pd.notna(yc):
        if yc < -0.5:
            messages.append("The yield curve is deeply inverted, historically aligned with recession risk.")
        elif yc < 0:
            messages.append("The yield curve is inverted, still a cautionary signal.")
        else:
            messages.append("The yield curve is no longer inverted or is normalizing.")

    if pd.notna(fsi):
        if fsi >= 1:
            messages.append("Financial stress is elevated.")
        elif fsi >= 0:
            messages.append("Financial stress is above normal but not extreme.")
        else:
            messages.append("Financial stress remains below historical average.")

    if pd.notna(ted):
        if ted >= 1:
            messages.append("Bank funding stress is elevated.")
        elif ted >= 0.5:
            messages.append("Funding stress is rising.")
        else:
            messages.append("Funding stress appears contained.")

    if pd.notna(oil):
        if oil >= 100:
            messages.append("Oil is at a shock-like level that can pressure inflation and growth.")
        elif oil >= 80:
            messages.append("Oil is high enough to keep inflation pressure alive.")
        else:
            messages.append("Oil is not currently in a shock zone.")

    if pd.notna(unr):
        if unr >= 5:
            messages.append("Unemployment is at a level consistent with a weakening economy.")
        elif unr >= 4.3:
            messages.append("Unemployment is drifting higher and deserves monitoring.")
        else:
            messages.append("Labor market still looks relatively firm.")

    return [html.Li(m) for m in messages] if messages else [html.Li("No interpretation available.")]


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
