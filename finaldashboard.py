import json
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

ROOT = Path(__file__).resolve().parents[0]
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

# Top 3 models (ranked by CV ROC-AUC) for the single-observation page
_REGISTRY = json.loads(
    (ROOT / "models" / "top_models_registry.json").read_text(encoding="utf-8")
)

TOP_MODELS = [
    {
        "rank": m["rank"],
        "key": m["key"],
        "name": m["name"],
        "cv_auc": m["cv_roc_auc"]["mean"],
        "model": load_model(ROOT / "models" / m["file"])
    }
    for m in _REGISTRY["models"]
]


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
            "fontSize": "12px",
            "display": "block",
            "margin": "0 0 4px"
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


STEPPERS = [
    ("p-tenure", 1, 72, 1),
    ("p-monthly", 100000, 2000000, 10000),
    ("p-complaints", 0, 10, 1),
    ("p-calls", 0, 15, 1)
]


def num_input(id_, lab, val, mn, mx, step=1):
    return html.Div([
        label(lab),
        html.Div(
            className="stepper",
            children=[
                html.Button("−", id=f"{id_}-minus", n_clicks=0),
                dcc.Input(
                    id=id_,
                    type="number",
                    value=val,
                    min=mn,
                    max=mx,
                    step=step
                ),
                html.Button("+", id=f"{id_}-plus", n_clicks=0)
            ]
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

# (id, label, emoji, target page or None if page not built yet)
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

        html.Div(id="sb-body", children=[
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
                style={"fontSize": "11px", "letterSpacing": ".08em", "color": SB["muted"]}
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
                options=["Bulanan", "Dua Tahun", "Tahunan"],
                value=["Bulanan", "Dua Tahun", "Tahunan"],
                labelStyle={"display": "block", "fontSize": "14px", "margin": "4px 0"}
            ),

            sb_label("Kategori risiko (prediksi)"),
            dcc.Checklist(
                id="f-risk",
                options=RISK_ORDER,
                value=RISK_ORDER,
                inline=True,
                labelStyle={"marginRight": "12px", "fontSize": "14px"}
            ),

            sb_label("Tenure (bulan)"),
            dcc.RangeSlider(
                id="f-tenure",
                min=1,
                max=72,
                value=[1, 72],
                step=1,
                marks={1: "1", 24: "24", 48: "48", 72: "72"}
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

# ------------------------------------------------------------
# Insight: how each factor affects churn, per model
# ------------------------------------------------------------

# (column, label, explanation)
FEATURES = [
    ("tenure_months", "Tenure (bulan)",
     "Lama pelanggan berlangganan. Pelanggan baru belum terikat dan paling mudah "
     "pindah; makin lama berlangganan, hubungan makin kuat dan risiko churn biasanya turun."),
    ("monthly_charges", "Tagihan bulanan (Rp)",
     "Biaya yang dibayar tiap bulan. Tagihan tinggi membuat pelanggan lebih sensitif "
     "harga dan lebih tergoda mencari penyedia yang lebih murah."),
    ("total_charges", "Total tagihan (Rp)",
     "Akumulasi tagihan sejak berlangganan (≈ tagihan bulanan × tenure). Mencerminkan "
     "nilai dan lamanya hubungan; sangat berkaitan dengan tenure."),
    ("num_complaints", "Jumlah keluhan",
     "Keluhan yang pernah diajukan. Sinyal langsung ketidakpuasan; keluhan berulang "
     "sering muncul sebelum pelanggan berhenti."),
    ("support_calls", "Panggilan CS",
     "Seberapa sering pelanggan menghubungi customer service. Panggilan yang banyak "
     "menandakan masalah yang belum tuntas atau layanan yang membingungkan."),
    ("contract_type", "Kontrak",
     "Jenis komitmen kontrak. Kontrak bulanan tidak mengikat sehingga mudah dihentikan; "
     "kontrak tahunan/dua tahun memberi penghalang berhenti."),
    ("payment_method", "Pembayaran",
     "Cara membayar. Metode yang manual atau kurang praktis dapat menambah gesekan "
     "dan peluang pembayaran terlewat, sedangkan yang otomatis membuat pelanggan lebih betah."),
    ("internet_service", "Layanan",
     "Jenis layanan internet. Kualitas dan harga tiap layanan berbeda sehingga kepuasan "
     "dan kecenderungan pindah ikut berbeda."),
    ("segment", "Segmen",
     "Tipe pelanggan (Korporat, Retail, UMKM). Kebutuhan, anggaran dan kesetiaan tiap "
     "segmen berbeda."),
    ("region", "Region",
     "Wilayah pelanggan. Perbedaan infrastruktur, persaingan dan kualitas layanan antar "
     "wilayah memengaruhi risiko churn."),
]

NUMERIC = META["numeric_features"]
FEAT_COLS = META["feature_order"]
FEAT_LABEL = {k: lab for k, lab, _ in FEATURES}
_SAMPLE = DF[FEAT_COLS].sample(min(600, len(DF)), random_state=0).reset_index(drop=True)
BASELINE = {
    **{c: DF[c].median() for c in NUMERIC},
    **{c: DF[c].mode()[0] for c in META["categorical_features"]}
}


def _proba(model, X):
    return model.predict_proba(X[FEAT_COLS])[:, 1]


def _with(X, col, val):
    """Copy of X with `col` set to `val` (total_charges follows tenure/tagihan)."""
    X = X.copy()
    X[col] = val
    if col in ("tenure_months", "monthly_charges"):
        X["total_charges"] = X["monthly_charges"] * X["tenure_months"]
    return X


def feature_effects(model):
    """Average effect on churn probability, measured on a sample of customers."""
    base = _proba(model, _SAMPLE).mean()
    out = {}
    for col, _, _ in FEATURES:
        if col in NUMERIC:
            lo, hi = DF[col].quantile([0.1, 0.9])
            d = (
                _proba(model, _with(_SAMPLE, col, hi)).mean()
                - _proba(model, _with(_SAMPLE, col, lo)).mean()
            )
            out[col] = {"kind": "num", "lo": lo, "hi": hi, "delta": d}
        else:
            means = {
                c: _proba(model, _with(_SAMPLE, col, c)).mean() - base
                for c in CATS[col]
            }
            out[col] = {
                "kind": "cat",
                "means": means,
                "delta": max(means.values()) - min(means.values())
            }
    return out


EFFECTS = {m["key"]: feature_effects(m["model"]) for m in TOP_MODELS}


def _pp(x):
    return f"{x * 100:+.1f} pp"


def _tone(d):
    if d > 0.005:
        return RISK_COLOR["Tinggi"], "▲"
    if d < -0.005:
        return RISK_COLOR["Rendah"], "▼"
    return C["muted"], "●"


def _num(x, col):
    return rupiah(x) if "charges" in col else f"{x:.0f}"


def effect_cell(col, e):
    if e["kind"] == "num":
        color, arrow = _tone(e["delta"])
        return html.Td([
            html.Div(
                f"{arrow} {_pp(e['delta'])}",
                style={"color": color, "fontWeight": 600}
            ),
            html.Div(
                f"jika naik dari {_num(e['lo'], col)} ke {_num(e['hi'], col)}",
                style={"color": C["muted"], "fontSize": "11px"}
            )
        ], style=TD)

    hi_c = max(e["means"], key=e["means"].get)
    lo_c = min(e["means"], key=e["means"].get)
    return html.Td([
        html.Div(
            f"▲ {hi_c} {_pp(e['means'][hi_c])}",
            style={"color": RISK_COLOR["Tinggi"], "fontWeight": 600}
        ),
        html.Div(
            f"▼ {lo_c} {_pp(e['means'][lo_c])}",
            style={"color": RISK_COLOR["Rendah"], "fontWeight": 600}
        ),
        html.Div(
            "dibanding rata-rata pelanggan",
            style={"color": C["muted"], "fontSize": "11px"}
        )
    ], style=TD)


def top_drivers(key, n=3):
    eff = EFFECTS[key]
    order = sorted(eff, key=lambda c: abs(eff[c]["delta"]), reverse=True)[:n]
    return [FEAT_LABEL[c] for c in order]


TD = {
    "padding": "8px 10px",
    "borderTop": "1px solid #E5E7EB",
    "verticalAlign": "top",
    "fontSize": "13px"
}
TH = {
    "padding": "8px 10px",
    "textAlign": "left",
    "fontSize": "12px",
    "background": "#E9EDF5"
}

insight_card = html.Div(
    style=CARD,
    children=[
        html.H4(
            "Insight: bagaimana tiap faktor memengaruhi risiko churn",
            style={"marginTop": 0, "fontSize": "15px"}
        ),
        html.P(
            "Angka dihitung dari masing-masing model terlatih pada contoh pelanggan nyata: "
            "satu faktor diubah (numerik: dari persentil ke-10 ke ke-90; kategori: dibanding "
            "rata-rata) sementara faktor lain tetap. “pp” = poin persentase probabilitas churn. "
            "Merah (▲) = menaikkan risiko, hijau (▼) = menurunkan. Ini hubungan statistik, "
            "bukan bukti sebab-akibat.",
            style={"color": C["muted"], "fontSize": "13px", "marginTop": 0}
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                "gap": "10px",
                "marginBottom": "12px"
            },
            children=[
                html.Div(
                    style={"background": "#F5F7FB", "borderRadius": "8px", "padding": "10px 12px"},
                    children=[
                        html.Div(
                            f"#{m['rank']} {m['name']}",
                            style={"fontWeight": 600, "fontSize": "13px"}
                        ),
                        html.Div(
                            "Pendorong terbesar: " + ", ".join(top_drivers(m["key"])),
                            style={"fontSize": "13px", "color": C["ink"]}
                        )
                    ]
                )
                for m in TOP_MODELS
            ]
        ),

        html.Div(
            style={"overflowX": "auto"},
            children=html.Table(
                style={"width": "100%", "borderCollapse": "collapse"},
                children=[
                    html.Thead(html.Tr([
                        html.Th("Faktor", style=TH),
                        html.Th("Arti & mengapa berpengaruh", style=TH),
                        *[html.Th(f"#{m['rank']} {m['name']}", style=TH) for m in TOP_MODELS]
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(label_, style={**TD, "fontWeight": 600, "whiteSpace": "nowrap"}),
                            html.Td(desc, style={**TD, "color": C["muted"], "minWidth": "240px"}),
                            *[effect_cell(col, EFFECTS[m["key"]][col]) for m in TOP_MODELS]
                        ])
                        for col, label_, desc in FEATURES
                    ])
                ]
            )
        )
    ]
)


def contrib_fig(model, row):
    """Bar chart: how each of this customer's values shifts churn vs a typical customer."""
    X = row[FEAT_COLS].copy()
    p = _proba(model, X)[0]
    items = []
    for col, label_, _ in FEATURES:
        d = p - _proba(model, _with(X, col, BASELINE[col]))[0]
        val = X[col].iloc[0]
        shown = _num(val, col) if col in NUMERIC else val
        items.append((f"{label_}<br><b>{shown}</b>", d))
    items.sort(key=lambda t: abs(t[1]))

    vals = [d * 100 for _, d in items]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = max(hi - lo, 1) * 0.35  # room so value labels never touch the axis labels

    fig = go.Figure(go.Bar(
        x=vals,
        y=[n for n, _ in items],
        orientation="h",
        marker_color=[
            RISK_COLOR["Tinggi"] if d > 0 else RISK_COLOR["Rendah"] for _, d in items
        ],
        text=[f"{d * 100:+.1f}" for _, d in items],
        textposition="outside",
        cliponaxis=False
    ))
    fig.update_layout(
        title={"text": "Pendorong churn pelanggan ini (vs tipikal)",
               "font": {"size": 12}, "x": 0.01},
        template="plotly_white",
        height=420,
        margin=dict(t=40, b=40, l=10, r=10),
        xaxis={
            "title": "Perubahan prob. churn (pp)",
            "zeroline": True,
            "range": [lo - pad, hi + pad]
        },
        yaxis={"automargin": True, "tickfont": {"size": 10}},
        font={"family": "Inter, Arial, sans-serif", "size": 11},
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def section_title(text, sub=None):
    return html.Div(
        style={"margin": "4px 2px 0"},
        children=[
            html.Div(text, style={"fontSize": "15px", "fontWeight": 700}),
            html.Div(sub, style={"fontSize": "12px", "color": C["muted"]}) if sub else None
        ]
    )


def result_card(m):
    return html.Div(
        style={**CARD, "minWidth": 0},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "gap": "8px",
                    "paddingBottom": "8px",
                    "borderBottom": "1px solid #E5E7EB"
                },
                children=[
                    html.Div([
                        html.Div(
                            f"#{m['rank']}",
                            style={"fontSize": "11px", "color": C["muted"], "fontWeight": 600}
                        ),
                        html.Div(
                            m["name"],
                            style={"fontSize": "15px", "fontWeight": 700, "lineHeight": 1.2}
                        )
                    ]),
                    html.Div(
                        f"CV ROC-AUC {m['cv_auc']:.4f}",
                        style={
                            "fontSize": "11px",
                            "color": C["muted"],
                            "background": "#F0F2F6",
                            "borderRadius": "999px",
                            "padding": "3px 8px",
                            "whiteSpace": "nowrap"
                        }
                    )
                ]
            ),

            html.Div(
                id=f"pred-output-{m['key']}",
                style={"padding": "10px 0 0", "minHeight": "118px"},
                children=html.P("Memuat prediksi…")
            ),

            dcc.Graph(
                id=f"g-gauge-{m['key']}",
                config={"displayModeBar": False},
                style={"height": "210px"}
            ),

            html.Div(
                style={"borderTop": "1px solid #E5E7EB", "marginTop": "6px"}
            ),

            dcc.Graph(
                id=f"g-contrib-{m['key']}",
                config={"displayModeBar": False},
                style={"height": "420px"}
            )
        ]
    )


tab_predict = html.Div(
    style={"display": "flex", "flexDirection": "column", "gap": "12px"},
    children=[
        # --- Input form ---------------------------------------------------
        html.Div(
            style=CARD,
            children=[
                html.H4(
                    "Simulasi prediksi pelanggan",
                    style={"marginTop": 0, "marginBottom": "12px", "fontSize": "15px"}
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                        "gap": "12px 16px",
                        "alignItems": "end"
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
                        num_input("p-calls", "Panggilan CS", 3, 0, 15),

                        html.Button(
                            "Jalankan prediksi",
                            id="btn-predict",
                            n_clicks=0,
                            className="primary-btn",
                            style={
                                "background": C["accent"],
                                "color": "white",
                                "border": 0,
                                "padding": "10px 16px",
                                "fontSize": "13px",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "height": "38px"
                            }
                        )
                    ]
                )
            ]
        ),

        # --- Results: one card per model, side by side ---------------------
        section_title(
            "Hasil prediksi dari 3 model",
            "Diurutkan berdasarkan CV ROC-AUC. Semua model memakai input yang sama."
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                "gap": "12px",
                "alignItems": "start"
            },
            children=[result_card(m) for m in TOP_MODELS]
        ),

        # --- Insight table ---------------------------------------------------
        insight_card
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
        "padding": "20px 28px",
        "fontFamily": "Inter, Arial, sans-serif",
        "color": C["ink"]
    },
    children=[
        html.Div(
            style={"marginBottom": "14px"},
            children=[
                html.H2(
                    "Customer Churn Intelligence Dashboard",
                    style={"margin": "0 0 4px", "fontSize": "26px"}
                ),

                html.Div(
                    "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus dilakukan? "
                    "· Data latihan 2023–2025 · Prediksi dari final trained model",
                    style={"color": C["muted"], "fontSize": "14px"}
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
                    style={"flex": 1, "minWidth": 0},
                    children=[
                        dcc.Store(id="page", data="overview"),
                        html.Div(id="page-overview", children=[tab_overview]),
                        html.Div(
                            id="page-predict",
                            style={"display": "none"},
                            children=[tab_predict]
                        )
                    ]
                )
            ]
        )
    ]
)


# ============================================================
# 7b. STEPPER (+/-) CALLBACKS
# ============================================================

def _register_stepper(id_, mn, mx, step):
    @callback(
        Output(id_, "value"),
        Input(f"{id_}-minus", "n_clicks"),
        Input(f"{id_}-plus", "n_clicks"),
        State(id_, "value"),
        prevent_initial_call=True
    )
    def _step(_m, _p, val):
        val = mn if val is None else val
        val = val - step if ctx.triggered_id == f"{id_}-minus" else val + step
        return max(mn, min(mx, val))


for _id, _mn, _mx, _st in STEPPERS:
    _register_stepper(_id, _mn, _mx, _st)



# ============================================================
# 7c. SIDEBAR NAVIGATION CALLBACKS
# ============================================================

NAV_TARGET = {id_: page for id_, _, _, page in NAV_ITEMS}


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
    Output("page-overview", "style"),
    Output("page-predict", "style"),
    *[Output(id_, "style") for id_, *_ in NAV_ITEMS],
    Input("page", "data")
)
def show_page(page):
    return (
        {"display": "block" if page == "overview" else "none"},
        {"display": "block" if page == "predict" else "none"},
        *[
            {**NAV_BASE, "background": SB["active"], "fontWeight": 600}
            if target == page else NAV_BASE
            for _, _, _, target in NAV_ITEMS
        ]
    )


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
    *[
        Output(f"{kind}-{m['key']}", prop)
        for m in TOP_MODELS
        for kind, prop in (
            ("pred-output", "children"),
            ("g-gauge", "figure"),
            ("g-contrib", "figure")
        )
    ],

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
        err = html.Div([
            html.P(e, style={"color": "crimson"})
            for e in errs
        ])
        return tuple(
            x for _ in TOP_MODELS for x in (err, px.scatter(), px.scatter())
        )

    # Run prediction with each of the top models
    outputs = []
    for m in TOP_MODELS:
        outputs.extend(
            render_result(predict(m["model"], row, META).iloc[0])
        )
        outputs.append(contrib_fig(m["model"], row))

    return tuple(outputs)


def render_result(r):

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
            style={"color": C["muted"], "fontSize": "14px"}
        )
    ])

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=r.churn_probability * 100,
            number={
                "suffix": "%",
                "valueformat": ".1f",
                "font": {"size": 38, "color": C["ink"]}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "tickfont": {"size": 10, "color": C["muted"]}
                },
                "bar": {
                    "color": RISK_COLOR[r.risk_category],
                    "thickness": 0.75
                },
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#E6F4EA"},
                    {"range": [30, 60], "color": "#FDF0E6"},
                    {"range": [60, 100], "color": "#FBE3E3"}
                ]
            }
        )
    )

    gauge.update_layout(
        margin=dict(t=20, b=5, l=25, r=25),
        height=210,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial, sans-serif"}
    )

    return out, gauge


# ============================================================
# 11. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=8054
    )