import base64
import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate


# ============================================================
# 1. PATHS AND MODEL
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from utils.model_utils import (
    load_model,
    load_metadata,
    validate_input,
    predict,
    risk_category
)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

META = load_metadata()
MODEL = load_model()

# Top 3 models (ranked by CV roc_auc) for multi-observation prediction
REGISTRY = json.loads(
    (ROOT / "models" / "top_models_registry.json").read_text(encoding="utf-8")
)

MODEL_SHORT = {
    "logreg": "LogReg",
    "logreg_ros": "LogReg + ROS",
    "gradient_boosting": "Gradient Boosting"
}

MODELS = {
    m["key"]: {
        "name": m["name"],
        "short": MODEL_SHORT.get(m["key"], m["name"]),
        "auc": m["cv_roc_auc"]["mean"],
        "model": load_model(ROOT / "models" / m["file"])
    }
    for m in sorted(REGISTRY["models"], key=lambda m: m["rank"])
}

raw = pd.read_csv(
    ROOT / "data" / "customer_churn.csv",
    parse_dates=["signup_date"]
)

raw["total_charges"] = raw["total_charges"].fillna(
    raw["monthly_charges"] * raw["tenure_months"]
)

# Batch prediction for all customers
DF = predict(MODEL, raw, META)

DF["status"] = DF["churn"].map({
    0: "Bertahan",
    1: "Churn"
})

DF["periode"] = DF["signup_date"].dt.to_period("Q").astype(str)

RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]

RISK_COLOR = {
    "Rendah": "#55A868",
    "Sedang": "#DD8452",
    "Tinggi": "#C44E52"
}

CATS = META["categories"]


# ============================================================
# 3. STYLE AND HELPER FUNCTIONS
# ============================================================

C = {
    "bg": "#F5F7FB",
    "card": "white",
    "ink": "#1F2A44",
    "muted": "#6B7280",
    "accent": "#2F5DA8"
}

CARD = {
    "background": C["card"],
    "borderRadius": "12px",
    "padding": "16px",
    "boxShadow": "0 1px 3px rgba(0,0,0,.08)"
}


def label(t):
    return html.Label(
        t,
        style={
            "fontWeight": 600,
            "fontSize": "13px"
        }
    )


def kpi(judul, nilai, sub=""):
    return html.Div(
        style={**CARD, "flex": 1, "minWidth": 0},
        children=[
            html.Div(
                judul,
                style={
                    "color": C["muted"],
                    "fontSize": "13px"
                }
            ),
            html.Div(
                nilai,
                style={
                    "fontSize": "28px",
                    "fontWeight": 700,
                    "color": C["ink"]
                }
            ),
            html.Div(
                sub,
                style={
                    "color": C["muted"],
                    "fontSize": "12px"
                }
            )
        ]
    )


def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")


def num_input(id_, lab, val, mn, mx, step=1):
    return html.Div([
        label(lab),
        dcc.Input(
            id=id_,
            type="number",
            value=val,
            min=mn,
            max=mx,
            step=step,
            style={"width": "100%"}
        )
    ])


def dd_input(id_, lab, opts):
    return html.Div([
        label(lab),
        dcc.Dropdown(
            id=id_,
            options=opts,
            value=opts[0],
            clearable=False
        )
    ])


# ============================================================
# 4. SIDEBAR FILTER
# ============================================================

SB = {
    "bg": "#6B6258",
    "text": "#F3EFEA",
    "muted": "#C9C2B8",
    "active": "rgba(255,255,255,.15)"
}

# (id, label, emoji, target tab or None if page not built yet)
NAV_ITEMS = [
    ("nav-overview", "Data Overview", "📊", "overview"),
    ("nav-eda", "Exploratory Data Analysis", "🔍", None),
    ("nav-single", "Prediksi Single Observasi", "👤", "predict"),
    ("nav-multi", "Prediksi Multi Observasi", "🗂️", "multi"),
    ("nav-team", "Tim Developer", "👥", None)
]

NAV_BASE = {
    "display": "block",
    "width": "100%",
    "textAlign": "left",
    "background": "transparent",
    "color": SB["text"],
    "border": 0,
    "borderRadius": "8px",
    "padding": "10px 12px",
    "fontSize": "14px",
    "cursor": "pointer"
}


def sb_label(t):
    return html.Div(
        t,
        style={
            "fontWeight": 600,
            "fontSize": "13px",
            "color": SB["text"],
            "margin": "14px 0 6px"
        }
    )


sidebar = html.Div(
    id="sidebar",
    style={
        "background": SB["bg"],
        "color": SB["text"],
        "borderRadius": "12px",
        "padding": "16px",
        "width": "260px",
        "flexShrink": 0,
        "boxShadow": "0 1px 3px rgba(0,0,0,.2)"
    },
    children=[
        # Header: title + hamburger
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "flex-start",
                "marginBottom": "16px"
            },
            children=[
                html.Div([
                    html.Div(
                        "Customer Churn",
                        style={"fontSize": "18px", "fontWeight": 700}
                    ),
                    html.Div(
                        "Intelligence Dashboard",
                        style={"fontSize": "11px", "color": SB["muted"]}
                    )
                ]),
                html.Button(
                    "«",
                    id="sb-close",
                    n_clicks=0,
                    title="Tutup sidebar",
                    style={
                        "background": "transparent",
                        "color": SB["text"],
                        "border": 0,
                        "fontSize": "20px",
                        "lineHeight": 1,
                        "cursor": "pointer"
                    }
                )
            ]
        ),

        # Collapsible body
        html.Div(id="sb-body", children=[
            # Navigation
            html.Div(
                id="sb-nav",
                children=[
                    html.Button(
                        f"{icon}  {text}",
                        id=id_,
                        n_clicks=0,
                        style=NAV_BASE
                    )
                    for id_, text, icon, _ in NAV_ITEMS
                ]
            ),

            html.Hr(style={"borderColor": "rgba(255,255,255,.2)", "margin": "16px 0 8px"}),

            html.Div(
                "FILTER DATA",
                style={
                    "fontSize": "11px",
                    "letterSpacing": ".08em",
                    "color": SB["muted"]
                }
            ),

            sb_label("Region"),
            dcc.Dropdown(
                id="f-region",
                options=CATS["region"],
                multi=True,
                placeholder="Semua",
                style={"color": C["ink"]}
            ),

            sb_label("Segmen"),
            dcc.Dropdown(
                id="f-segment",
                options=CATS["segment"],
                multi=True,
                placeholder="Semua",
                style={"color": C["ink"]}
            ),

            sb_label("Jenis kontrak"),
            dcc.Checklist(
                id="f-kontrak",
                options=CATS["contract_type"],
                value=CATS["contract_type"]
            ),

            sb_label("Kategori risiko (prediksi)"),
            dcc.Checklist(
                id="f-risk",
                options=RISK_ORDER,
                value=RISK_ORDER,
                inline=True,
                labelStyle={"marginRight": "10px"}
            ),

            sb_label("Tenure (bulan)"),
            dcc.RangeSlider(
            id="f-tenure",
            min=1,
            max=72,
            value=[1, 72],
            step=1,
            marks={
                1: "1",
                24: "24",
                48: "48",
                72: "72"
            }
        ),

            html.Hr(style={"borderColor": "rgba(255,255,255,.2)"}),

            html.Small(
                f"Model: {META['model_name']} v{META['model_version']}",
                style={"color": SB["muted"]}
            )
        ])
    ]
)


# ============================================================
# 5. OVERVIEW TAB
# ============================================================

tab_overview = html.Div([
    html.Div(
        id="kpi-row",
        style={
            "display": "flex",
            "gap": "12px",
            "marginBottom": "12px"
        }
    ),

    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "12px"
        },
        children=[
            html.Div(dcc.Graph(id="g-donut"), style=CARD),
            html.Div(dcc.Graph(id="g-region-risk"), style=CARD),
            html.Div(dcc.Graph(id="g-trend"), style=CARD),
            html.Div(dcc.Graph(id="g-contract"), style=CARD),
            html.Div(dcc.Graph(id="g-risk"), style=CARD),
            html.Div(dcc.Graph(id="g-actual-pred"), style=CARD)
        ]
    ),

    html.Div(
        style={
            **CARD,
            "marginTop": "12px"
        },
        children=[
            html.H4(
                id="tbl-title",
                style={"marginTop": 0}
            ),

            dash_table.DataTable(
                id="tbl-risk",
                page_size=10,
                sort_action="native",
                filter_action="native",

                columns=[
                    {
                        "name": name,
                        "id": column_id,
                        **(
                            {
                                "type": "numeric",
                                "format": {"specifier": ".1%"}
                            }
                            if column_id == "churn_probability"
                            else {}
                        )
                    }
                    for name, column_id in [
                        ("ID", "customer_id"),
                        ("Region", "region"),
                        ("Segmen", "segment"),
                        ("Kontrak", "contract_type"),
                        ("Tenure", "tenure_months"),
                        ("Keluhan", "num_complaints"),
                        ("Prob. churn", "churn_probability"),
                        ("Risiko", "risk_category")
                    ]
                ],

                style_cell={
                    "padding": "6px",
                    "fontSize": "13px",
                    "textAlign": "left"
                },

                style_header={
                    "fontWeight": "bold",
                    "background": "#E9EDF5"
                },

                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": "{risk_category} = 'Tinggi'",
                            "column_id": "risk_category"
                        },
                        "color": "white",
                        "backgroundColor": RISK_COLOR["Tinggi"]
                    },
                    {
                        "if": {
                            "filter_query": "{risk_category} = 'Sedang'",
                            "column_id": "risk_category"
                        },
                        "backgroundColor": "#FBE3D3"
                    }
                ]
            )
        ]
    )
])


# ============================================================
# 6. PREDICTION TAB
# ============================================================

tab_predict = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gap": "12px"
    },
    children=[
        html.Div(
            style=CARD,
            children=[
                html.H4(
                    "Simulasi prediksi pelanggan",
                    style={"marginTop": 0}
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "10px"
                    },
                    children=[
                        dd_input("p-region", "Region", CATS["region"]),
                        dd_input("p-segment", "Segmen", CATS["segment"]),
                        dd_input("p-contract", "Kontrak", CATS["contract_type"]),
                        dd_input("p-payment", "Pembayaran", CATS["payment_method"]),
                        dd_input("p-service", "Layanan", CATS["internet_service"]),

                        num_input("p-tenure", "Tenure (bulan)", 6, 1, 72),
                        num_input("p-monthly", "Tagihan bulanan (Rp)", 450000, 100000, 2000000, 10000),
                        num_input("p-complaints", "Jumlah keluhan", 2, 0, 10),
                        num_input("p-calls", "Panggilan CS", 3, 0, 15)
                    ]
                ),

                html.Br(),

                html.Button(
                    "Jalankan prediksi",
                    id="btn-predict",
                    n_clicks=0,
                    style={
                        "background": C["accent"],
                        "color": "white",
                        "border": 0,
                        "padding": "10px 18px",
                        "borderRadius": "8px",
                        "cursor": "pointer"
                    }
                )
            ]
        ),

        html.Div(
            style=CARD,
            children=[
                html.H4(
                    "Hasil prediksi",
                    style={"marginTop": 0}
                ),

                html.Div(
                    id="pred-output",
                    children=html.P("Isi form lalu klik tombol.")
                ),

                dcc.Graph(
                    id="g-gauge",
                    style={"height": "260px"}
                )
            ]
        )
    ]
)


# ============================================================
# 6b. MULTI-OBSERVATION PREDICTION TAB
# ============================================================

FEATURES = META["feature_order"]
THRESHOLD = META["decision_threshold"]

FEATURE_LABELS = {
    "customer_id": "ID",
    "region": "Region",
    "segment": "Segmen",
    "contract_type": "Kontrak",
    "payment_method": "Pembayaran",
    "internet_service": "Layanan",
    "tenure_months": "Tenure",
    "monthly_charges": "Tagihan bulanan",
    "total_charges": "Total tagihan",
    "num_complaints": "Keluhan",
    "support_calls": "Panggilan CS"
}

BTN = {
    "background": C["accent"],
    "color": "white",
    "border": 0,
    "padding": "8px 14px",
    "borderRadius": "8px",
    "cursor": "pointer",
    "marginRight": "8px"
}

BTN_OUTLINE = {
    **BTN,
    "background": "white",
    "color": C["accent"],
    "border": f"1px solid {C['accent']}"
}

tab_multi = html.Div([
    dcc.Store(id="m-data"),
    dcc.Store(id="m-result"),
    dcc.Download(id="dl-template"),
    dcc.Download(id="dl-result"),

    html.Div(
        style={**CARD, "marginBottom": "12px"},
        children=[
            html.H4("Prediksi multi observasi", style={"marginTop": 0}),

            html.P(
                "Unggah file CSV atau Excel berisi data pelanggan. Kolom wajib: "
                + ", ".join(c for c in FEATURES if c != "total_charges")
                + ". Kolom customer_id dan total_charges opsional "
                "(total_charges dihitung dari tagihan bulanan × tenure bila kosong).",
                style={"color": C["muted"], "fontSize": "13px"}
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "3fr 2fr",
                    "gap": "16px"
                },
                children=[
                    dcc.Upload(
                        id="m-upload",
                        accept=".csv,.xlsx,.xls",
                        multiple=False,
                        children=html.Div([
                            "📁 Tarik & lepas file di sini atau ",
                            html.A("pilih file", style={"color": C["accent"], "fontWeight": 600}),
                            html.Div("(.csv / .xlsx)", style={"color": C["muted"], "fontSize": "12px"})
                        ]),
                        style={
                            "border": f"2px dashed {C['accent']}",
                            "borderRadius": "10px",
                            "padding": "28px 12px",
                            "textAlign": "center",
                            "cursor": "pointer",
                            "background": "#F8FAFF"
                        }
                    ),

                    html.Div([
                        label("Model yang digunakan"),
                        dcc.Checklist(
                            id="m-models",
                            options=[
                                {
                                    "label": f" {v['name']} (CV AUC {v['auc']:.3f})",
                                    "value": k
                                }
                                for k, v in MODELS.items()
                            ],
                            value=list(MODELS),
                            labelStyle={"display": "block", "fontSize": "13px", "margin": "4px 0"}
                        ),
                        html.Div(
                            "Prediksi akhir = rata-rata probabilitas model terpilih "
                            f"(threshold {THRESHOLD:.0%}).",
                            style={"color": C["muted"], "fontSize": "12px", "marginTop": "6px"}
                        )
                    ])
                ]
            ),

            html.Div(
                style={"marginTop": "12px"},
                children=[
                    html.Button("Unduh contoh file", id="btn-template", n_clicks=0, style=BTN_OUTLINE)
                ]
            ),

            dcc.Loading(html.Div(id="m-status", style={"marginTop": "10px", "fontSize": "13px"}))
        ]
    ),

    html.Div(
        id="m-results",
        style={"display": "none"},
        children=[
            html.Div(
                id="m-kpi",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "12px",
                    "marginBottom": "12px"
                }
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "12px"
                },
                children=[
                    html.Div(
                        dcc.Graph(
                            id=gid,
                            config={"responsive": True, "displaylogo": False},
                            style={"height": "300px", "width": "100%"}
                        ),
                        style={**CARD, "minWidth": 0, "overflow": "hidden"}
                    )
                    for gid in ["g-m-pie", "g-m-risk-model", "g-m-contract", "g-m-region"]
                ]
            ),

            html.Div(
                style={**CARD, "marginTop": "12px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "8px"
                        },
                        children=[
                            html.H4("Tabel hasil prediksi", style={"margin": 0}),
                            html.Div([
                                html.Button("⬇ Export Excel", id="btn-dl-xlsx", n_clicks=0, style=BTN),
                                html.Button("⬇ Export CSV", id="btn-dl-csv", n_clicks=0, style=BTN_OUTLINE)
                            ])
                        ]
                    ),

                    dash_table.DataTable(
                        id="tbl-multi",
                        page_size=15,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "padding": "6px",
                            "fontSize": "13px",
                            "textAlign": "left",
                            "whiteSpace": "nowrap"
                        },
                        style_header={
                            "fontWeight": "bold",
                            "background": "#E9EDF5"
                        }
                    )
                ]
            )
        ]
    )
])


# ============================================================
# 7. MAIN DASHBOARD LAYOUT
# ============================================================

PAGES = {
    "overview": tab_overview,
    "predict": tab_predict,
    "multi": tab_multi
}

app = Dash(
    __name__,
    title="Final Dashboard — Customer Churn"
)

server = app.server

app.layout = html.Div(
    style={
        "background": C["bg"],
        "minHeight": "100vh",
        "padding": "16px",
        "fontFamily": "Inter, Arial, sans-serif",
        "color": C["ink"]
    },
    children=[
        dcc.Store(id="sb-open", data=True),

        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "marginBottom": "12px"
            },
            children=[
                html.Button(
                    "☰",
                    id="sb-show",
                    n_clicks=0,
                    title="Buka sidebar",
                    style={"display": "none"}
                ),

                html.Div([
                    html.H2(
                        "Customer Churn Intelligence Dashboard",
                        style={"margin": 0}
                    ),

                    html.Div(
                        "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus dilakukan? "
                        "· Data latihan 2023–2025 · Prediksi dari final trained model",
                        style={"color": C["muted"]}
                    )
                ])
            ]
        ),

        html.Div(
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "flex-start"
            },
            children=[
                sidebar,

                html.Div(
                    style={
                        "flex": 1,
                        "minWidth": 0
                    },
                    children=[
                        dcc.Store(id="page", data="overview"),
                        *[
                            html.Div(id=f"page-{key}", children=[content])
                            for key, content in PAGES.items()
                        ]
                    ]
                )
            ]
        )
    ]
)


# ============================================================
# 7b. SIDEBAR CALLBACKS (navigation, active highlight, collapse)
# ============================================================

NAV_TARGET = {id_: tab for id_, _, _, tab in NAV_ITEMS}


@callback(
    Output("page", "data"),
    [Input(id_, "n_clicks") for id_, *_ in NAV_ITEMS],
    prevent_initial_call=True
)
def nav_to_page(*_):
    page = NAV_TARGET.get(ctx.triggered_id)
    if page is None:
        raise PreventUpdate  # page belum tersedia
    return page


@callback(
    [Output(id_, "style") for id_, *_ in NAV_ITEMS],
    [Output(f"page-{key}", "style") for key in PAGES],
    Input("page", "data")
)
def show_page(active):
    nav = [
        {**NAV_BASE, "background": SB["active"], "fontWeight": 600}
        if page == active
        else NAV_BASE
        for _, _, _, page in NAV_ITEMS
    ]
    pages = [{"display": "block" if key == active else "none"} for key in PAGES]
    return nav + pages


SB_SHOW_BTN = {
    "background": SB["bg"],
    "color": SB["text"],
    "border": 0,
    "borderRadius": "8px",
    "width": "40px",
    "height": "40px",
    "fontSize": "20px",
    "cursor": "pointer",
    "flexShrink": 0,
    "boxShadow": "0 1px 3px rgba(0,0,0,.2)"
}


@callback(
    Output("sb-open", "data"),
    Input("sb-close", "n_clicks"),
    Input("sb-show", "n_clicks"),
    prevent_initial_call=True
)
def toggle_sidebar(*_):
    return ctx.triggered_id == "sb-show"


@callback(
    Output("sidebar", "style"),
    Output("sb-show", "style"),
    Input("sb-open", "data"),
    State("sidebar", "style")
)
def render_sidebar(is_open, style):
    style = {**style, "display": "block" if is_open else "none"}
    return style, {**SB_SHOW_BTN, "display": "none" if is_open else "block"}


# ============================================================
# 8. CALLBACK FILTER FUNCTION
# ============================================================

def apply_filters(region, segment, kontrak, risk, tenure):

    d = DF

    if region:
        d = d[d["region"].isin(region)]

    if segment:
        d = d[d["segment"].isin(segment)]

    d = d[
        d["contract_type"].isin(kontrak or [])
        & d["risk_category"].isin(risk or [])
    ]

    return d[d["tenure_months"].between(*tenure)]


FILTERS = [
    Input("f-region", "value"),
    Input("f-segment", "value"),
    Input("f-kontrak", "value"),
    Input("f-risk", "value"),
    Input("f-tenure", "value")
]


# ============================================================
# 9. CALLBACK 1: UPDATE OVERVIEW DASHBOARD
# ============================================================

@callback(
    Output("kpi-row", "children"),
    Output("g-donut", "figure"),
    Output("g-region-risk", "figure"),
    Output("g-trend", "figure"),
    Output("g-contract", "figure"),
    Output("g-risk", "figure"),
    Output("g-actual-pred", "figure"),
    Output("tbl-risk", "data"),
    Output("tbl-title", "children"),
    *FILTERS
)

def update_overview(region, segment, kontrak, risk, tenure):

    d = apply_filters(region, segment, kontrak, risk, tenure)

    if d.empty:

        f = px.scatter(
            title="Tidak ada data — longgarkan filter",
            template="plotly_white"
        )

        return (
            [kpi("Pelanggan", "0")],
            f, f, f, f, f, f,
            [],
            "Tidak ada data"
        )

    # KPI
    tinggi = (d["risk_category"] == "Tinggi").sum()

    kpis = [
        kpi("Pelanggan", f"{len(d):,}", "sesuai filter"),

        kpi(
            "Churn rate aktual",
            f"{d['churn'].mean():.1%}",
            f"total: {DF['churn'].mean():.1%}"
        ),

        kpi(
            "Rata-rata prob. churn",
            f"{d['churn_probability'].mean():.1%}",
            "prediksi model"
        ),

        kpi(
            "Risiko tinggi",
            f"{tinggi:,}",
            rupiah(
                d.loc[
                    d["risk_category"] == "Tinggi",
                    "monthly_charges"
                ].sum()
            ) + "/bln berisiko"
        )
    ]

    # Chart 1: Risk composition
    g0 = px.pie(
        d,
        names="risk_category",
        hole=0.55,
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_category": RISK_ORDER},
        title="Komposisi kategori risiko"
    )

    g0.update_traces(
        textinfo="percent+label",
        sort=False
    )

    g0.add_annotation(
        text=f"<b>{len(d):,}</b><br>pelanggan",
        showarrow=False,
        font_size=14
    )

    g0.update_layout(showlegend=False)

    # Chart 2: High-risk customers by region
    rr = (
        d.assign(
            tinggi=d["risk_category"].eq("Tinggi")
        )
        .groupby("region", as_index=False)["tinggi"]
        .mean()
        .sort_values("tinggi")
    )

    g5 = px.bar(
        rr,
        x="tinggi",
        y="region",
        orientation="h",
        text_auto=".1%",
        template="plotly_white",
        color_discrete_sequence=[RISK_COLOR["Tinggi"]],
        title="Porsi pelanggan risiko tinggi per region",
        labels={
            "tinggi": "Porsi risiko tinggi",
            "region": ""
        }
    )

    g5.update_xaxes(tickformat=".0%")

    # Chart 3: Churn trend
    tr = (
        d.groupby("periode")
        .agg(
            aktual=("churn", "mean"),
            prediksi=("churn_probability", "mean")
        )
        .reset_index()
    )

    g1 = px.line(
        tr.melt(
            "periode",
            var_name="seri",
            value_name="rate"
        ),
        x="periode",
        y="rate",
        color="seri",
        markers=True,
        template="plotly_white",
        title="Churn aktual vs rata-rata prediksi per kuartal pendaftaran",
        labels={
            "periode": "Kuartal pendaftaran",
            "rate": "Rate",
            "seri": ""
        }
    )

    g1.update_yaxes(tickformat=".0%")

    # Chart 4: Contract type and risk
    ck = (
        d.groupby(["contract_type", "risk_category"])
        .size()
        .reset_index(name="n")
    )

    g2 = px.bar(
        ck,
        x="contract_type",
        y="n",
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={
            "risk_category": RISK_ORDER,
            "contract_type": [
                "Bulanan",
                "Tahunan",
                "Dua Tahun"
            ]
        },
        template="plotly_white",
        title="Komposisi risiko per jenis kontrak",
        labels={
            "contract_type": "",
            "n": "Pelanggan",
            "risk_category": "Risiko"
        }
    )

    # Chart 5: Churn probability distribution
    g3 = px.histogram(
        d,
        x="churn_probability",
        nbins=30,
        color="risk_category",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_category": RISK_ORDER},
        template="plotly_white",
        title="Distribusi probabilitas churn (prediksi)",
        labels={
            "churn_probability": "Probabilitas churn",
            "risk_category": "Risiko"
        }
    )

    g3.add_vline(
        x=META["decision_threshold"],
        line_dash="dash",
        annotation_text="threshold"
    )

    # Chart 6: Actual vs predicted churn by region
    rg = (
        d.groupby("region")
        .agg(
            aktual=("churn", "mean"),
            prediksi=("churn_probability", "mean"),
            n=("customer_id", "count")
        )
        .reset_index()
    )

    g4 = px.scatter(
        rg,
        x="aktual",
        y="prediksi",
        size="n",
        text="region",
        template="plotly_white",
        title="Region: churn aktual vs prediksi (ukuran = jumlah pelanggan)",
        labels={
            "aktual": "Churn rate aktual",
            "prediksi": "Rata-rata prediksi"
        }
    )

    g4.update_traces(
        textposition="top center",
        cliponaxis=False
    )

    pad = 0.03

    g4.update_xaxes(
        tickformat=".0%",
        range=[
            rg["aktual"].min() - pad,
            rg["aktual"].max() + pad
        ]
    )

    g4.update_yaxes(
        tickformat=".0%",
        range=[
            rg["prediksi"].min() - pad,
            rg["prediksi"].max() + pad
        ]
    )

    # Risk priority table
    top = (
        d.sort_values(
            "churn_probability",
            ascending=False
        )
        .head(200)
    )

    return (
        kpis,
        g0,
        g5,
        g1,
        g2,
        g3,
        g4,
        top.to_dict("records"),
        f"Prioritas retensi — {min(len(d), 200)} pelanggan dengan probabilitas tertinggi"
    )


# ============================================================
# 10. CALLBACK 2: CUSTOMER CHURN PREDICTION
# ============================================================

@callback(
    Output("pred-output", "children"),
    Output("g-gauge", "figure"),

    Input("btn-predict", "n_clicks"),

    State("p-region", "value"),
    State("p-segment", "value"),
    State("p-contract", "value"),
    State("p-payment", "value"),
    State("p-service", "value"),
    State("p-tenure", "value"),
    State("p-monthly", "value"),
    State("p-complaints", "value"),
    State("p-calls", "value"),

    prevent_initial_call=True
)

def run_prediction(
    n,
    region,
    segment,
    contract,
    payment,
    service,
    tenure,
    monthly,
    complaints,
    calls
):

    if None in (tenure, monthly, complaints, calls):
        raise PreventUpdate

    # Create customer input DataFrame
    row = pd.DataFrame([{
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": monthly * tenure,
        "num_complaints": complaints,
        "support_calls": calls,
        "region": region,
        "segment": segment,
        "contract_type": contract,
        "payment_method": payment,
        "internet_service": service
    }])

    # Validate input
    errs = validate_input(row, META)

    if errs:
        return (
            html.Div([
                html.P(e, style={"color": "crimson"})
                for e in errs
            ]),
            px.scatter()
        )

    # Run prediction
    r = predict(MODEL, row, META).iloc[0]

    # Recommendation based on risk category
    saran = {
        "Tinggi": (
            "Prioritaskan retensi: hubungi dalam 48 jam, "
            "tawarkan kontrak tahunan."
        ),

        "Sedang": (
            "Pantau: kirim survei kepuasan "
            "dan penawaran loyalitas."
        ),

        "Rendah": "Pertahankan layanan standar."
    }[r.risk_category]

    # Prediction output
    out = html.Div([
        html.Div(
            f"{r.churn_probability:.1%}",
            style={
                "fontSize": "40px",
                "fontWeight": 700,
                "color": RISK_COLOR[r.risk_category]
            }
        ),

        html.Div(
            f"Prediksi: "
            f"{'Churn' if r.predicted_churn else 'Bertahan'} "
            f"· Risiko {r.risk_category}"
        ),

        html.P(
            saran,
            style={"color": C["muted"]}
        )
    ])

    # Gauge chart
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=r.churn_probability * 100,

            number={"suffix": "%"},

            gauge={
                "axis": {"range": [0, 100]},

                "bar": {
                    "color": RISK_COLOR[r.risk_category]
                },

                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#E6F4EA"
                    },
                    {
                        "range": [30, 60],
                        "color": "#FDF0E6"
                    },
                    {
                        "range": [60, 100],
                        "color": "#FBE3E3"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        margin=dict(
            t=20,
            b=10,
            l=20,
            r=20
        ),
        height=240
    )

    return out, gauge


# ============================================================
# 11. CALLBACK 3: MULTI-OBSERVATION PREDICTION
# ============================================================

def read_upload(contents, filename):
    _, b64 = contents.split(",", 1)
    data = base64.b64decode(b64)
    name = (filename or "").lower()

    if name.endswith(".csv"):
        text = data.decode("utf-8-sig")
        header = text.splitlines()[0] if text else ""
        sep = ";" if header.count(";") > header.count(",") else ","
        return pd.read_csv(io.StringIO(text), sep=sep)

    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))

    raise ValueError("Format file tidak didukung. Gunakan .csv atau .xlsx.")


def clean_upload(df):
    """Mengembalikan (data valid, catatan, error)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "total_charges" not in df.columns:
        df["total_charges"] = pd.NA

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return None, [], f"Kolom wajib tidak ditemukan: {', '.join(missing)}"

    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", [f"ROW-{i + 1:04d}" for i in range(len(df))])
    df["customer_id"] = df["customer_id"].astype(str)

    for c in META["numeric_features"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])

    for c in META["categorical_features"]:
        df[c] = df[c].astype("string").str.strip()

    bad = df[FEATURES].isna().any(axis=1)
    for c in META["categorical_features"]:
        bad |= ~df[c].isin(META["categories"][c])

    notes = []
    if bad.any():
        ids = ", ".join(df.loc[bad, "customer_id"].head(10))
        more = " …" if bad.sum() > 10 else ""
        notes.append(
            f"{bad.sum()} baris dilewati karena nilai kosong atau kategori tidak dikenal "
            f"(ID: {ids}{more})."
        )

    df = df.loc[~bad, ["customer_id", *FEATURES]].reset_index(drop=True)
    for c in META["categorical_features"]:
        df[c] = df[c].astype(str)

    errs = validate_input(df, META)
    return df, notes, "; ".join(errs) if errs else None


def predict_multi(df, keys):
    out = df.copy()
    for k in keys:
        r = predict(MODELS[k]["model"], df, META)
        out[f"prob_{k}"] = r["churn_probability"]
        out[f"pred_{k}"] = r["predicted_churn"].map({1: "Churn", 0: "Bertahan"})
        out[f"risk_{k}"] = r["risk_category"]

    probs = out[[f"prob_{k}" for k in keys]]
    out["prob_avg"] = probs.mean(axis=1).round(4)
    out["votes"] = (probs >= THRESHOLD).sum(axis=1)
    out["pred_final"] = (out["prob_avg"] >= THRESHOLD).map({True: "Churn", False: "Bertahan"})
    out["risk_final"] = [risk_category(p, META) for p in out["prob_avg"]]
    return out


def result_columns(keys):
    """(id, nama kolom, format persen?) untuk tabel dan export."""
    cols = [(c, FEATURE_LABELS[c], False) for c in ["customer_id", *FEATURES]]
    for k in keys:
        s = MODELS[k]["short"]
        cols += [
            (f"prob_{k}", f"Prob. {s}", True),
            (f"pred_{k}", f"Prediksi {s}", False),
            (f"risk_{k}", f"Risiko {s}", False)
        ]
    cols += [
        ("prob_avg", "Prob. rata-rata", True),
        ("votes", f"Model prediksi churn (dari {len(keys)})", False),
        ("pred_final", "Prediksi akhir", False),
        ("risk_final", "Risiko akhir", False)
    ]
    return cols


def model_long(res, keys):
    """Hasil per model dalam format panjang untuk visualisasi."""
    return pd.concat([
        pd.DataFrame({
            "model": MODELS[k]["short"],
            "customer_id": res["customer_id"],
            "contract_type": res["contract_type"],
            "prob": res[f"prob_{k}"],
            "pred": res[f"pred_{k}"],
            "risk": res[f"risk_{k}"]
        })
        for k in keys
    ], ignore_index=True)


def risk_styles(keys):
    styles = []
    for col in [f"risk_{k}" for k in keys] + ["risk_final"]:
        styles += [
            {
                "if": {"filter_query": f"{{{col}}} = 'Tinggi'", "column_id": col},
                "color": "white",
                "backgroundColor": RISK_COLOR["Tinggi"]
            },
            {
                "if": {"filter_query": f"{{{col}}} = 'Sedang'", "column_id": col},
                "backgroundColor": "#FBE3D3"
            }
        ]
    for col in [f"pred_{k}" for k in keys] + ["pred_final"]:
        styles.append({
            "if": {"filter_query": f"{{{col}}} = 'Churn'", "column_id": col},
            "color": "crimson",
            "fontWeight": 600
        })
    return styles


def msg(text, color):
    return html.Div(text, style={"color": color})


@callback(
    Output("m-status", "children"),
    Output("m-results", "style"),
    Output("m-result", "data"),
    Output("m-kpi", "children"),
    Output("g-m-pie", "figure"),
    Output("g-m-risk-model", "figure"),
    Output("g-m-contract", "figure"),
    Output("g-m-region", "figure"),
    Output("tbl-multi", "data"),
    Output("tbl-multi", "columns"),
    Output("tbl-multi", "style_data_conditional"),
    Input("m-upload", "contents"),
    Input("m-models", "value"),
    State("m-upload", "filename")
)
def run_multi_prediction(contents, keys, filename):
    hidden = {"display": "none"}
    skip = [no_update] * 8

    if not contents:
        return msg("Belum ada file yang diunggah.", C["muted"]), hidden, None, *skip

    keys = [k for k in MODELS if k in (keys or [])]
    if not keys:
        return msg("Pilih minimal satu model.", "crimson"), hidden, None, *skip

    try:
        df, notes, err = clean_upload(read_upload(contents, filename))
    except Exception as e:
        return msg(f"Gagal membaca file '{filename}': {e}", "crimson"), hidden, None, *skip

    if err:
        return msg(err, "crimson"), hidden, None, *skip
    if df.empty:
        return msg("Tidak ada baris valid untuk diprediksi. " + " ".join(notes), "crimson"), \
            hidden, None, *skip

    res = predict_multi(df, keys)
    long = model_long(res, keys)
    n = len(res)

    status = html.Div([
        msg(f"✅ {filename}: {n:,} pelanggan diprediksi dengan {len(keys)} model.", "#2E7D32"),
        *[msg(f"⚠️ {t}", "#B26A00") for t in notes]
    ])

    # KPI
    n_churn = (res["pred_final"] == "Churn").sum()
    high = res["risk_final"] == "Tinggi"
    agree = (res["votes"].isin([0, len(keys)])).mean()
    kpis = [
        kpi("Pelanggan diprediksi", f"{n:,}", filename),
        kpi("Prediksi churn", f"{n_churn:,}", f"{n_churn / n:.1%} dari total"),
        kpi("Risiko tinggi", f"{high.sum():,}",
            rupiah(res.loc[high, "monthly_charges"].sum()) + "/bln berisiko"),
        kpi("Kesepakatan model", f"{agree:.1%}", "semua model memberi prediksi sama")
    ]

    # Chart 1: final risk composition
    g_pie = px.pie(
        res,
        names="risk_final",
        hole=0.55,
        color="risk_final",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_final": RISK_ORDER},
        title="Komposisi risiko akhir"
    )
    g_pie.update_traces(textinfo="percent+label", sort=False)
    g_pie.add_annotation(text=f"<b>{n:,}</b><br>pelanggan", showarrow=False, font_size=13)
    g_pie.update_layout(showlegend=False)

    # Chart 2: risk category count per model
    rk = long.groupby(["model", "risk"], sort=False).size().reset_index(name="jumlah")
    g_risk = px.bar(
        rk,
        x="model",
        y="jumlah",
        color="risk",
        barmode="group",
        text_auto=True,
        color_discrete_map=RISK_COLOR,
        category_orders={"risk": RISK_ORDER},
        template="plotly_white",
        title="Kategori risiko per model",
        labels={"model": "", "jumlah": "Pelanggan", "risk": ""}
    )

    # Chart 3: average churn probability per contract type, per model
    ct = long.groupby(["contract_type", "model"], sort=False)["prob"].mean().reset_index()
    g_contract = px.bar(
        ct,
        x="contract_type",
        y="prob",
        color="model",
        barmode="group",
        category_orders={"contract_type": ["Bulanan", "Tahunan", "Dua Tahun"]},
        template="plotly_white",
        title="Rata-rata prob. churn per kontrak",
        labels={"contract_type": "", "prob": "Prob. churn", "model": ""}
    )
    g_contract.update_yaxes(tickformat=".0%")

    # Chart 4: final risk per region
    rg = res.groupby(["region", "risk_final"]).size().reset_index(name="jumlah")
    g_region = px.bar(
        rg,
        x="jumlah",
        y="region",
        color="risk_final",
        orientation="h",
        color_discrete_map=RISK_COLOR,
        category_orders={"risk_final": RISK_ORDER},
        template="plotly_white",
        title="Risiko akhir per region",
        labels={"jumlah": "Pelanggan", "region": "", "risk_final": ""}
    )

    for fig in (g_pie, g_risk, g_contract, g_region):
        fig.update_layout(
            autosize=True,
            height=None,
            margin=dict(t=60, b=30, l=10, r=10),
            title=dict(font_size=14, x=0.02, y=0.97),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font_size=11),
            font_size=11
        )

    columns = [
        {"name": name, "id": cid, "type": "numeric", "format": {"specifier": ".1%"}}
        if pct else {"name": name, "id": cid}
        for cid, name, pct in result_columns(keys)
    ]
    records = res.to_dict("records")

    return (
        status,
        {"display": "block"},
        {"keys": keys, "records": records, "filename": filename},
        kpis,
        g_pie, g_risk, g_contract, g_region,
        records,
        columns,
        risk_styles(keys)
    )


@callback(
    Output("dl-result", "data"),
    Input("btn-dl-xlsx", "n_clicks"),
    Input("btn-dl-csv", "n_clicks"),
    State("m-result", "data"),
    prevent_initial_call=True
)
def export_result(_xlsx, _csv, result):
    if not result:
        raise PreventUpdate

    keys = result["keys"]
    cols = result_columns(keys)
    res = pd.DataFrame(result["records"])[[c for c, *_ in cols]]
    table = res.rename(columns={c: name for c, name, _ in cols})
    base = Path(result["filename"] or "data").stem + "_hasil_prediksi"

    if ctx.triggered_id == "btn-dl-csv":
        return dcc.send_data_frame(table.to_csv, f"{base}.csv", index=False)

    summary = pd.DataFrame([
        {
            "Model": MODELS[k]["name"],
            "CV ROC AUC": MODELS[k]["auc"],
            "Prediksi churn": int((res[f"pred_{k}"] == "Churn").sum()),
            "% churn": round((res[f"pred_{k}"] == "Churn").mean(), 4),
            "Rata-rata prob.": round(res[f"prob_{k}"].mean(), 4),
            **{f"Risiko {r}": int((res[f"risk_{k}"] == r).sum()) for r in RISK_ORDER}
        }
        for k in keys
    ])

    def write(buf):
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            table.to_excel(xw, sheet_name="Hasil Prediksi", index=False)
            summary.to_excel(xw, sheet_name="Ringkasan Model", index=False)

    return dcc.send_bytes(write, f"{base}.xlsx")


@callback(
    Output("dl-template", "data"),
    Input("btn-template", "n_clicks"),
    prevent_initial_call=True
)
def download_template(_):
    return dcc.send_file(str(ROOT / "data" / "new_customers.csv"))


# ============================================================
# 12. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=8054
    )