import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate


# ============================================================
# 1. PATHS AND MODEL
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.model_utils import (
    load_model,
    load_metadata,
    validate_input,
    predict
)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

META = load_metadata()
MODEL = load_model()

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
        style={**CARD, "flex": 1},
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
    ("nav-multi", "Prediksi Multi Observasi", "🗂️", None),
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
                    "☰",
                    id="sb-toggle",
                    n_clicks=0,
                    style={
                        "background": "transparent",
                        "color": SB["text"],
                        "border": 0,
                        "fontSize": "18px",
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
# 7. MAIN DASHBOARD LAYOUT
# ============================================================

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
        html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.H2(
                    "Customer Churn Intelligence Dashboard",
                    style={"margin": 0}
                ),

                html.Div(
                    "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus dilakukan? "
                    "· Data latihan 2023–2025 · Prediksi dari final trained model",
                    style={"color": C["muted"]}
                )
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
                        dcc.Tabs(
                            id="tabs",
                            value="overview",
                            children=[
                                dcc.Tab(
                                    label="Overview & Risiko",
                                    value="overview",
                                    children=[tab_overview]
                                ),

                                dcc.Tab(
                                    label="Simulasi Prediksi",
                                    value="predict",
                                    children=[tab_predict]
                                )
                            ]
                        )
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
    Output("tabs", "value"),
    [Input(id_, "n_clicks") for id_, *_ in NAV_ITEMS],
    prevent_initial_call=True
)
def nav_to_tab(*_):
    tab = NAV_TARGET.get(ctx.triggered_id)
    if tab is None:
        raise PreventUpdate  # page belum tersedia
    return tab


@callback(
    [Output(id_, "style") for id_, *_ in NAV_ITEMS],
    Input("tabs", "value")
)
def highlight_nav(active_tab):
    return [
        {**NAV_BASE, "background": SB["active"], "fontWeight": 600}
        if tab == active_tab
        else NAV_BASE
        for _, _, _, tab in NAV_ITEMS
    ]


@callback(
    Output("sb-body", "style"),
    Output("sidebar", "style"),
    Input("sb-toggle", "n_clicks"),
    State("sidebar", "style")
)
def toggle_sidebar(n, style):
    style = dict(style)
    collapsed = bool(n and n % 2)
    style["width"] = "70px" if collapsed else "260px"
    return {"display": "none" if collapsed else "block"}, style


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
# 11. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=8053
    )