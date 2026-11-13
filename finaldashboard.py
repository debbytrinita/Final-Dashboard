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

# Raw copy (before imputation) for the column profile
RAW_TRAIN = raw.copy()

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


def kpi_plain(judul, nilai, sub=""):
    """KPI without card, used on the Data Overview page."""
    return html.Div([
        html.Div(judul, style={"color": C["muted"], "fontSize": "12px"}),
        html.Div(
            nilai,
            style={
                "fontSize": "26px",
                "fontWeight": 700,
                "color": C["ink"],
                "lineHeight": 1.25
            }
        ),
        html.Div(sub, style={"color": C["muted"], "fontSize": "11px"})
    ])


def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")


MONO = "Consolas, 'Courier New', monospace"

GRID2 = {
    "display": "grid",
    # equal-width columns; stacks on narrow screens without overflowing
    "gridTemplateColumns": "repeat(auto-fit, minmax(min(360px, 100%), 1fr))",
    "gap": "24px"
}

STATUS_COLOR = {"Bertahan": RISK_COLOR["Rendah"], "Churn": RISK_COLOR["Tinggi"]}

RAW_COLS = list(RAW_TRAIN.columns)

NUM_LABEL = {
    "tenure_months": "Tenure (bulan)",
    "monthly_charges": "Tagihan bulanan",
    "total_charges": "Total tagihan",
    "num_complaints": "Jumlah keluhan",
    "support_calls": "Panggilan CS"
}

MONO_TABLE = dict(
    style_table={"overflowX": "auto"},
    style_cell={
        "fontFamily": MONO,
        "fontSize": "12px",
        "padding": "6px 10px",
        "textAlign": "left",
        "border": "1px solid #EEF1F6",
        "whiteSpace": "nowrap"
    },
    style_header={
        "fontWeight": 700,
        "background": "#E9EDF5",
        "color": C["ink"]
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFD"}
    ]
)


def block_title(t, sub=None):
    return html.Div(
        style={"marginBottom": "8px"},
        children=[
            html.Div(
                t,
                style={
                    "fontSize": "15px",
                    "fontWeight": 700,
                    "color": C["ink"],
                    "borderLeft": f"4px solid {C['accent']}",
                    "paddingLeft": "8px"
                }
            ),
            html.Div(
                sub,
                style={
                    "fontSize": "12px",
                    "color": C["muted"],
                    "margin": "2px 0 0 12px"
                }
            ) if sub else None
        ]
    )


def info_row(k, v):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "minmax(150px, 40%) 1fr",
            "gap": "12px",
            "padding": "7px 0",
            "fontSize": "13px",
            "borderBottom": "1px solid #EEF1F6"
        },
        children=[
            html.Div(k, style={"color": C["muted"]}),
            html.Div(v, style={"fontWeight": 600, "color": C["ink"]})
        ]
    )


def form_label(t):
    return html.Label(
        t,
        style={
            "fontSize": "12px",
            "display": "block",
            "margin": "0 0 4px"
        }
    )



STEPPERS = [
    ("p-tenure", 1, 72, 1),
    ("p-monthly", 100000, 2000000, 10000),
    ("p-complaints", 0, 10, 1),
    ("p-calls", 0, 15, 1)
]


def num_input(id_, lab, val, mn, mx, step=1):
    return html.Div([
        form_label(lab),
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
        form_label(lab),
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
    ("nav-team", "Tim Developer", "👥", "team")
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
# 5. DATA OVERVIEW PAGE
# ============================================================

tab_overview = html.Div(
    style={**CARD, "padding": "20px 24px"},
    children=[
        html.H2("Data Overview", style={"margin": 0, "fontSize": "22px"}),
        html.Div(
            "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus dilakukan? "
            "· Data latihan 2023–2025 · Prediksi dari final trained model",
            style={"color": C["muted"], "fontSize": "13px", "marginTop": "2px"}
        ),

        # KPI row
        html.Div(
            id="kpi-row",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "16px",
                "margin": "20px 0 12px"
            }
        ),

        # Donut + dataset information
        html.Div(
            style=GRID2,
            children=[
                html.Div([
                    block_title("Komposisi status churn aktual"),
                    dcc.Graph(
                        id="g-status",
                        config={"displayModeBar": False},
                        style={"height": "300px"}
                    )
                ]),
                html.Div([
                    block_title("Informasi dataset"),
                    html.Div(id="ds-info")
                ])
            ]
        ),

        # Column profile + descriptive statistics
        html.Div(
            style={**GRID2, "marginTop": "24px"},
            children=[
                html.Div([
                    block_title(
                        "Profil kolom",
                        "Missing dihitung dari data mentah sebelum imputasi total_charges."
                    ),
                    dash_table.DataTable(
                        id="tbl-profile",
                        page_size=len(RAW_COLS),
                        columns=[
                            {"name": n, "id": i}
                            for n, i in [
                                ("Kolom", "kolom"),
                                ("Tipe", "tipe"),
                                ("Missing (mentah)", "missing"),
                                ("Nilai unik", "unik"),
                                ("Contoh", "contoh")
                            ]
                        ],
                        **MONO_TABLE,
                        style_cell_conditional=[
                            {"if": {"column_id": c}, "textAlign": "right"}
                            for c in ["missing", "unik"]
                        ]
                    )
                ]),
                html.Div([
                    block_title("Statistik deskriptif variabel numerik"),
                    dash_table.DataTable(
                        id="tbl-stats",
                        columns=[
                            {"name": n, "id": i}
                            for n, i in [
                                ("variabel", "variabel"),
                                ("count", "count"),
                                ("mean", "mean"),
                                ("std", "std"),
                                ("min", "min"),
                                ("25%", "q25"),
                                ("50%", "q50"),
                                ("75%", "q75"),
                                ("max", "max")
                            ]
                        ],
                        **{
                            **MONO_TABLE,
                            "style_cell": {
                                **MONO_TABLE["style_cell"],
                                "padding": "6px 8px"
                            }
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": c}, "textAlign": "right"}
                            for c in ["count", "mean", "std", "min",
                                      "q25", "q50", "q75", "max"]
                        ]
                    )
                ])
            ]
        )
    ]
)


# ============================================================
# 6. PREDICTION TAB
# ============================================================

# Top 3 models (ranked by CV ROC-AUC) for the single-observation page
TOP_MODELS = [
    {
        "rank": i,
        "key": key,
        "name": m["name"],
        "cv_auc": m["auc"],
        "model": m["model"]
    }
    for i, (key, m) in enumerate(MODELS.items(), start=1)
]


# ------------------------------------------------------------
# Insight: how each factor affects churn, per model
# ------------------------------------------------------------

# (column, label, explanation)
INSIGHT_FEATURES = [
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
FEAT_LABEL = {k: lab for k, lab, _ in INSIGHT_FEATURES}
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
    for col, _, _ in INSIGHT_FEATURES:
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
                        for col, label_, desc in INSIGHT_FEATURES
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
    for col, label_, _ in INSIGHT_FEATURES:
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
# 6c. TEAM PAGE
# ============================================================

TEAM = [
    "Debby Trinita",
    "Jessica Saputra",
    "Nitriko Puji Raharjo",
    "Darren Davarel"
]


def member_card(no, name):
    initials = "".join(w[0] for w in name.split()[:2]).upper()
    return html.Div(
        style={
            **CARD,
            "border": "1px solid #EEF1F6",
            "boxShadow": "none",
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "textAlign": "center",
            "padding": "24px 16px"
        },
        children=[
            html.Div(
                initials,
                style={
                    "width": "72px",
                    "height": "72px",
                    "borderRadius": "50%",
                    "background": C["accent"],
                    "color": "white",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "24px",
                    "fontWeight": 700,
                    "marginBottom": "12px"
                }
            ),
            html.Div(
                name,
                style={"fontSize": "15px", "fontWeight": 700, "color": C["ink"]}
            ),
            html.Div(
                f"Anggota {no}",
                style={"fontSize": "12px", "color": C["muted"], "marginTop": "2px"}
            )
        ]
    )


tab_team = html.Div(
    style={**CARD, "padding": "20px 24px"},
    children=[
        html.H2("Tim Developer", style={"margin": 0, "fontSize": "22px"}),
        html.Div(
            "Tim pengembang Customer Churn Intelligence Dashboard.",
            style={"color": C["muted"], "fontSize": "13px", "marginTop": "2px"}
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(min(200px, 100%), 1fr))",
                "gap": "16px",
                "marginTop": "20px"
            },
            children=[
                member_card(i, name)
                for i, name in enumerate(TEAM, start=1)
            ]
        )
    ]
)


# ============================================================
# 7. MAIN DASHBOARD LAYOUT
# ============================================================

PAGES = {
    "overview": tab_overview,
    "predict": tab_predict,
    "multi": tab_multi,
    "team": tab_team
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
            style={"marginBottom": "0"},
            children=[
                html.Button(
                    "☰",
                    id="sb-show",
                    n_clicks=0,
                    title="Buka sidebar",
                    style={"display": "none"}
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
    "boxShadow": "0 1px 3px rgba(0,0,0,.2)",
    "marginBottom": "12px"
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
# 9. CALLBACK 1: UPDATE DATA OVERVIEW
# ============================================================

@callback(
    Output("kpi-row", "children"),
    Output("g-status", "figure"),
    Output("ds-info", "children"),
    Output("tbl-profile", "data"),
    Output("tbl-stats", "data"),
    *FILTERS
)
def update_overview(region, segment, kontrak, risk, tenure):

    d = apply_filters(region, segment, kontrak, risk, tenure)
    raw_f = RAW_TRAIN.loc[d.index]  # same rows, before imputation
    n = len(d)
    total = f"{len(DF):,}".replace(",", ".")

    if n == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Tidak ada data — longgarkan filter",
            showarrow=False
        )
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return (
            [kpi_plain("Pelanggan", "0", f"dari {total} total")],
            fig,
            [info_row("Jumlah baris (sesuai filter)", "0")],
            [],
            []
        )

    # KPI
    n_churn = int(d["churn"].sum())
    pendapatan = d["monthly_charges"].sum()
    kpis = [
        kpi_plain("Pelanggan", f"{n:,}", f"dari {total} total"),
        kpi_plain("Churn aktual", f"{n_churn:,}", f"churn rate {d['churn'].mean():.1%}"),
        kpi_plain(
            "Rata-rata tenure",
            f"{d['tenure_months'].mean():.1f} bln",
            f"median {d['tenure_months'].median():.0f} bln"
        ),
        kpi_plain(
            "Pendapatan bulanan",
            rupiah(pendapatan),
            f"rata-rata {rupiah(pendapatan / n)}/pelanggan"
        )
    ]

    # Donut: actual churn status
    counts = d["status"].value_counts().reindex(["Bertahan", "Churn"], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=list(counts.index),
        values=list(counts.values),
        hole=0.55,
        sort=False,
        direction="clockwise",
        marker=dict(
            colors=[STATUS_COLOR[s] for s in counts.index],
            line=dict(color="white", width=2)
        ),
        textinfo="label+percent",
        textposition="inside",
        insidetextorientation="radial",
        textfont=dict(color="white", size=12)
    ))
    fig.add_annotation(
        text=f"<b>{n:,}</b><br>pelanggan",
        showarrow=False,
        font=dict(size=15, color=C["ink"])
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif")
    )

    # Dataset information
    num = META["numeric_features"]
    cat = META["categorical_features"]
    info = [
        info_row("Jumlah baris (sesuai filter)", f"{n:,}"),
        info_row("Jumlah kolom", f"{len(RAW_COLS)}"),
        info_row("Variabel numerik", ", ".join(num)),
        info_row("Variabel kategorik", ", ".join(cat)),
        info_row("Variabel target", "churn (0 = Bertahan, 1 = Churn)"),
        info_row(
            "Periode pendaftaran",
            f"{d['signup_date'].min():%d %b %Y} – {d['signup_date'].max():%d %b %Y}"
        ),
        info_row("Total missing value (mentah)", f"{int(raw_f.isna().sum().sum())} sel"),
        info_row("Baris duplikat", f"{int(raw_f.duplicated().sum())}")
    ]

    # Column profile (raw data, before imputation)
    profile = []
    for c in RAW_COLS:
        s = raw_f[c]
        contoh = s.dropna().iloc[0] if s.notna().any() else "-"
        if isinstance(contoh, pd.Timestamp):
            contoh = contoh.strftime("%Y-%m-%d")
        profile.append({
            "kolom": c,
            "tipe": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unik": int(s.nunique()),
            "contoh": str(contoh)
        })

    # Descriptive statistics of numeric variables
    desc = d[num].describe().T.round(2)
    stats = [
        {
            "variabel": NUM_LABEL.get(c, c),
            "count": int(r["count"]),
            **{
                # large values without decimals so the table fits its column
                key: (
                    int(round(r[col])) if abs(r[col]) >= 1000
                    else round(float(r[col]), 2)
                )
                for key, col in [
                    ("mean", "mean"), ("std", "std"), ("min", "min"),
                    ("q25", "25%"), ("q50", "50%"), ("q75", "75%"),
                    ("max", "max")
                ]
            }
        }
        for c, r in desc.iterrows()
    ]

    return kpis, fig, info, profile, stats


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