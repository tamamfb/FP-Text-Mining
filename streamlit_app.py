import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Deteksi Inkonsistensi Review",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigasi ──────────────────────────────────────────────────
st.sidebar.title("Inkonsistensi Review\nAplikasi Mobile")
st.sidebar.markdown("---")
PAGE = st.sidebar.radio("Navigasi", [
    "Dashboard Dataset",
    "Eksplorasi Data",
    "Skenario 1: Encoding Rating",
    "Skenario 2: Perbandingan Model",
    "Skenario 3: Panjang Teks",
    "Demo Prediksi",
])
st.sidebar.markdown("---")
st.sidebar.caption("FP Text Mining — ITS 2025/2026")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists("clean_reviews.csv"):
        return None
    df = pd.read_csv("clean_reviews.csv")
    df["segment"] = pd.cut(
        df["word_count"],
        bins=[0, 19, 50, 9999],
        labels=["Pendek", "Sedang", "Panjang"],
    )
    return df

def no_data():
    st.error("File `clean_reviews.csv` tidak ditemukan. Jalankan `preprocess.ipynb` terlebih dahulu.")
    st.stop()

# ── Hardcoded Results ─────────────────────────────────────────
S1 = {
    "A": {
        "name": "Variant A — Fitur Numerik",
        "desc": "teks → [CLS] ⊕ (rating/5.0)",
        "acc": 0.7775, "f1": 0.7710,
        "report": {"Konsisten": {"P": 0.83, "R": 0.88, "F1": 0.86},
                   "Inkonsisten": {"P": 0.56, "R": 0.48, "F1": 0.52}},
        "history": {
            "train_loss": [0.4581, 0.3915, 0.3151, 0.2246, 0.1586],
            "val_loss":   [0.4233, 0.4242, 0.5058, 0.5312, 0.7642],
            "train_acc":  [0.7783, 0.8157, 0.8615, 0.9074, 0.9406],
            "val_acc":    [0.8001, 0.7988, 0.7674, 0.7877, 0.7836],
        },
    },
    "B": {
        "name": "Variant B — Rating Embedding",
        "desc": "teks → [CLS] ⊕ Embedding(rating, dim=32)",
        "acc": 0.9578, "f1": 0.9581,
        "report": {"Konsisten": {"P": 0.98, "R": 0.97, "F1": 0.97},
                   "Inkonsisten": {"P": 0.90, "R": 0.93, "F1": 0.92}},
        "history": {
            "train_loss": [0.2732, 0.1224, 0.0855, 0.0593, 0.0411],
            "val_loss":   [0.1481, 0.1302, 0.1383, 0.1521, 0.1751],
            "train_acc":  [0.9063, 0.9665, 0.9769, 0.9848, 0.9887],
            "val_acc":    [0.9588, 0.9581, 0.9579, 0.9575, 0.9583],
        },
    },
    "C": {
        "name": "Variant C — Token di Teks",
        "desc": "`[RATING_X]` teks → encoder bersama",
        "acc": 0.9677, "f1": 0.9678,
        "report": {"Konsisten": {"P": 0.98, "R": 0.98, "F1": 0.98},
                   "Inkonsisten": {"P": 0.93, "R": 0.94, "F1": 0.94}},
        "history": {
            "train_loss": [0.1463, 0.0825, 0.0522, 0.0305, 0.0234],
            "val_loss":   [0.1181, 0.1084, 0.1663, 0.1966, 0.2235],
            "train_acc":  [0.9467, 0.9735, 0.9843, 0.9917, 0.9939],
            "val_acc":    [0.9595, 0.9634, 0.9570, 0.9607, 0.9572],
        },
    },
}

S2 = {
    "LR":  {"name": "TF-IDF + Logistic Regression", "cat": "Baseline Klasik",
             "acc": 0.8989, "f1_w": 0.9026, "f1_m": 0.8760,
             "report": {"Konsisten": {"P": 0.98, "R": 0.89, "F1": 0.93},
                        "Inkonsisten": {"P": 0.73, "R": 0.94, "F1": 0.82}}},
    "SVC": {"name": "TF-IDF + Linear SVC",          "cat": "Baseline Klasik",
             "acc": 0.9000, "f1_w": 0.9023, "f1_m": 0.8730,
             "report": {"Konsisten": {"P": 0.96, "R": 0.91, "F1": 0.93},
                        "Inkonsisten": {"P": 0.76, "R": 0.88, "F1": 0.81}}},
    "NB":  {"name": "TF-IDF + Multinomial NB",      "cat": "Baseline Klasik",
             "acc": 0.7659, "f1_w": 0.7036, "f1_m": 0.5450,
             "report": {"Konsisten": {"P": 0.77, "R": 0.98, "F1": 0.86},
                        "Inkonsisten": {"P": 0.65, "R": 0.14, "F1": 0.23}}},
    "CNN": {"name": "TextCNN",                       "cat": "DL non-pretrained",
             "acc": 0.8954, "f1_w": 0.8958, "f1_m": 0.8616,
             "report": {"Konsisten": {"P": 0.93, "R": 0.93, "F1": 0.93},
                        "Inkonsisten": {"P": 0.78, "R": 0.80, "F1": 0.79}},
             "history": {
                 "train_loss": [0.5037, 0.4201, 0.3337, 0.2572, 0.1964, 0.1615, 0.1288, 0.1121, 0.0973, 0.0812],
                 "val_loss":   [0.4900, 0.3632, 0.3182, 0.2985, 0.2879, 0.3044, 0.3402, 0.4135, 0.4760, 0.4102],
                 "train_acc":  [0.7658, 0.8044, 0.8552, 0.8950, 0.9235, 0.9375, 0.9516, 0.9593, 0.9633, 0.9697],
                 "val_acc":    [0.7815, 0.8340, 0.8606, 0.8860, 0.8905, 0.8780, 0.8819, 0.8730, 0.8685, 0.8655],
             }},
    "BERT":{"name": "IndoBERT Variant C",            "cat": "DL pretrained",
             "acc": 0.9619, "f1_w": 0.9621, "f1_m": 0.9497,
             "report": {"Konsisten": {"P": 0.98, "R": 0.97, "F1": 0.97},
                        "Inkonsisten": {"P": 0.91, "R": 0.94, "F1": 0.92}},
             "history": {
                 "train_loss": [0.1467, 0.0829, 0.0534, 0.0356, 0.0245],
                 "val_loss":   [0.1238, 0.1032, 0.1700, 0.2178, 0.2013],
                 "train_acc":  [0.9475, 0.9727, 0.9839, 0.9902, 0.9939],
                 "val_acc":    [0.9581, 0.9619, 0.9600, 0.9583, 0.9615],
             }},
}

S3 = {
    "Pendek":  {"n": 5238, "n_incon": 1296, "acc": 0.9624, "f1_w": 0.9626, "f1_m": 0.9500, "f1_i": 0.9251, "fn": 79,  "fp": 118,
                "report": {"Konsisten": {"P": 0.98, "R": 0.97, "F1": 0.97}, "Inkonsisten": {"P": 0.91, "R": 0.94, "F1": 0.93}}},
    "Sedang":  {"n": 2168, "n_incon": 558,  "acc": 0.9613, "f1_w": 0.9617, "f1_m": 0.9505, "f1_i": 0.9273, "fn": 22,  "fp": 62,
                "report": {"Konsisten": {"P": 0.99, "R": 0.96, "F1": 0.97}, "Inkonsisten": {"P": 0.90, "R": 0.96, "F1": 0.93}}},
    "Panjang": {"n": 437,  "n_incon": 106,  "acc": 0.9703, "f1_w": 0.9705, "f1_m": 0.9602, "f1_i": 0.9401, "fn": 4,   "fp": 9,
                "report": {"Konsisten": {"P": 0.99, "R": 0.97, "F1": 0.98}, "Inkonsisten": {"P": 0.92, "R": 0.96, "F1": 0.94}}},
}
S3_HIST = {
    "train_loss": [0.1420, 0.0841, 0.0509, 0.0354, 0.0238],
    "val_loss":   [0.1215, 0.1303, 0.1677, 0.1662, 0.2437],
    "train_acc":  [0.9478, 0.9721, 0.9845, 0.9900, 0.9937],
    "val_acc":    [0.9595, 0.9569, 0.9596, 0.9598, 0.9588],
    "val_f1":     [0.9596, 0.9576, 0.9598, 0.9601, 0.9590],
}

FN_EXAMPLES = {
    "Pendek": [
        {"Rating": 1, "Teks Review": "aplikasi lawak. ambil driver jauh2 mulu aneh, lawak lawak asli", "Kata": 10},
        {"Rating": 4, "Teks Review": "tolong pebaiki tampilan supaya nyaman di instal di tablet", "Kata": 9},
        {"Rating": 4, "Teks Review": "limit dana cicil ada pi tidak tersedia tu kenapa bos..", "Kata": 10},
    ],
    "Sedang": [
        {"Rating": 1, "Teks Review": "shoope mantap. tiap kali baca manhwa padahal diem doang ga scroll ga mencet apa apa. tapi tiba tiba teleportasi ke apk lain", "Kata": 36},
        {"Rating": 5, "Teks Review": "yang mati karena naik sepeda motor lebih banyak dari yang mati karena vape. kenapa tokopedia permudah beli sesuatu berbahaya", "Kata": 28},
        {"Rating": 4, "Teks Review": "mohon disegerakan mode gelap di go-jek. menunya sudah lama ada tapi kok nggak buru-buru direalisasikan untuk semua pengguna", "Kata": 29},
    ],
    "Panjang": [
        {"Rating": 1, "Teks Review": "karena gaada penilaian ekspedisi jadi sekalian aja kunilai bareng aplikasi (platform) nya. sistem kurir rekomendasi shopee express...", "Kata": 62},
        {"Rating": 4, "Teks Review": "saya dari dulu senang memakai bca karena mudah, praktis, ga sering kendala kaya bank lain, tapi saya mau mengeluhkan fitur terbarunya...", "Kata": 69},
        {"Rating": 1, "Teks Review": "makin lama makin mantap nih tokped pengirimannya. bisa sampe seminggu lebih juga loh ternyata. hebat ya, bisa lama gitu...", "Kata": 67},
    ],
}

# ── Model Architecture ────────────────────────────────────────
MODEL_NAME = "indobenchmark/indobert-base-p1"

class ModelVariantC(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        h = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes) + 1, 128),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
    def forward(self, input_ids, rating):
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), conv(x).size(2)).squeeze(2) for conv in self.convs]
        x = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(torch.cat([x, rating.unsqueeze(1)], dim=1))

# ── Model Loading ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat IndoBERT S3...")
def load_indobert_s3():
    path = "models/model_s3_indobert.pt"
    if not os.path.exists(path):
        return None, None
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ModelVariantC().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer

@st.cache_resource(show_spinner="Memuat TF-IDF + LR...")
def load_tfidf_lr():
    p1, p2 = "models/tfidf_s2.pkl", "models/lr_s2.pkl"
    if not os.path.exists(p1) or not os.path.exists(p2):
        return None, None
    import joblib
    return joblib.load(p1), joblib.load(p2)

@st.cache_resource(show_spinner="Memuat TextCNN...")
def load_textcnn():
    path = "models/model_s2_textcnn.pt"
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = TextCNNModel(
        ckpt["vocab_size"], ckpt["embed_dim"],
        ckpt["num_filters"], ckpt["kernel_sizes"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

# ── Prediction Helpers ────────────────────────────────────────
def predict_bert(text, rating, model, tokenizer, max_len=128):
    inp = f"[RATING_{rating}] {text}"
    enc = tokenizer(inp, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs

def predict_lr(text, rating, tfidf, lr):
    from scipy.sparse import hstack, csr_matrix
    X = hstack([tfidf.transform([text]), csr_matrix([[rating / 5.0]])])
    pred  = lr.predict(X)[0]
    probs = lr.predict_proba(X)[0]
    return int(pred), probs

def predict_cnn(text, rating, model, ckpt):
    vocab   = ckpt["vocab"]
    max_len = ckpt["max_len"]
    tokens  = text.lower().split()[:max_len]
    ids     = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
    ids    += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    inp = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    rat = torch.tensor([rating / 5.0], dtype=torch.float).to(DEVICE)
    with torch.no_grad():
        out   = model(inp, rat)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs

# ── Reusable chart helpers ────────────────────────────────────
def loss_acc_chart(history, epochs, title=""):
    ep = list(range(1, epochs + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Accuracy"])
    fig.add_trace(go.Scatter(x=ep, y=history["train_loss"], name="Train Loss",
                             mode="lines+markers", line=dict(color="#3498db")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=history["val_loss"],   name="Val Loss",
                             mode="lines+markers", line=dict(color="#3498db", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=history["train_acc"],  name="Train Acc",
                             mode="lines+markers", line=dict(color="#e74c3c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=history["val_acc"],    name="Val Acc",
                             mode="lines+markers", line=dict(color="#e74c3c", dash="dash")), row=1, col=2)
    fig.update_layout(height=350, title_text=title)
    return fig

def report_df(report_dict):
    return pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "Kelas"})

# ── Logic Flow Visualizer ─────────────────────────────────────
def render_logic_flow(text, rating, pred, probs, model_name="Model"):
    confidence = probs[pred] * 100
    text_preview = (text[:24] + "…") if len(text) > 24 else text
    stars = "★" * rating + "☆" * (5 - rating)

    out_label  = "1 — Inkonsisten" if pred == 1 else "0 — Konsisten"
    out_color  = "#e05c5c" if pred == 1 else "#3dba7e"
    out_bg     = "#1c0e0e" if pred == 1 else "#0e1c14"
    result_tag = "Inkonsisten" if pred == 1 else "Konsisten"

    anim_key = model_name.replace(" ", "").replace("+", "")
    html = f"""
    <style>
      @keyframes pulse_{anim_key} {{
        0%,100% {{ border-color: {out_color}99; }}
        50%      {{ border-color: {out_color}; }}
      }}
    </style>
    <div style="background:#0d1117;border-radius:12px;padding:18px 16px;
                font-family:'Segoe UI',system-ui,sans-serif;border:1px solid #21262d;">

      <div style="display:flex;align-items:center;justify-content:center;
                  gap:10px;overflow-x:auto;">

        <!-- INPUT -->
        <div style="background:#161b22;border:1.5px solid #388bfd;border-radius:10px;
                    padding:12px 14px;min-width:130px;max-width:150px;
                    text-align:center;flex-shrink:0;">
          <div style="color:#484f58;font-size:9px;font-weight:600;letter-spacing:1.5px;
                      text-transform:uppercase;margin-bottom:6px;">Input</div>
          <div style="color:#c9d1d9;font-size:11px;line-height:1.5;
                      margin-bottom:6px;word-break:break-word;">"{text_preview}"</div>
          <div style="color:#e3b341;font-size:13px;">{stars} ({rating}★)</div>
        </div>

        <!-- Arrow -->
        <div style="color:#30363d;font-size:22px;flex-shrink:0;">&#8594;</div>

        <!-- MODEL -->
        <div style="background:#1a1630;border:1.5px solid #7c5cbf;border-radius:10px;
                    padding:12px 14px;min-width:140px;text-align:center;flex-shrink:0;">
          <div style="color:#484f58;font-size:9px;font-weight:600;letter-spacing:1.5px;
                      text-transform:uppercase;margin-bottom:6px;">Model</div>
          <div style="color:#b99fec;font-size:13px;font-weight:600;
                      margin-bottom:5px;">{model_name}</div>
          <div style="color:#484f58;font-size:10px;line-height:1.5;">
            Klasifikasi langsung<br>teks + rating &#8594; label
          </div>
        </div>

        <!-- Arrow -->
        <div style="color:#30363d;font-size:22px;flex-shrink:0;">&#8594;</div>

        <!-- OUTPUT -->
        <div style="background:{out_bg};border:1.5px solid {out_color};border-radius:10px;
                    padding:12px 14px;min-width:130px;text-align:center;flex-shrink:0;
                    animation:pulse_{anim_key} 2s ease-in-out infinite;">
          <div style="color:#484f58;font-size:9px;font-weight:600;letter-spacing:1.5px;
                      text-transform:uppercase;margin-bottom:6px;">Output</div>
          <div style="color:{out_color};font-size:15px;font-weight:700;
                      margin-bottom:4px;">{out_label}</div>
          <div style="background:#21262d;border-radius:3px;height:3px;
                      overflow:hidden;margin:8px 0 5px;">
            <div style="background:{out_color};height:100%;
                        width:{confidence:.0f}%;border-radius:3px;"></div>
          </div>
          <div style="color:#484f58;font-size:10px;">konfiden {confidence:.1f}%</div>
        </div>
      </div>

      <div style="margin-top:12px;padding-top:10px;border-top:1px solid #21262d;
                  color:#484f58;font-size:10px;text-align:center;line-height:1.6;">
        Model memprediksi label konsistensi secara langsung dari teks + rating &mdash;
        bukan melalui deteksi sentimen terpisah.
      </div>
    </div>
    """
    components.html(html, height=220)


# ══════════════════════════════════════════════════════════════
# PAGE 1: Dashboard Dataset
# ══════════════════════════════════════════════════════════════
if PAGE == "Dashboard Dataset":
    st.title("Dashboard Dataset")
    st.markdown("Review aplikasi mobile Indonesia dari Google Play Store — 5 aplikasi, scraping Juni 2026.")

    df = load_data()
    if df is None:
        no_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Review", f"{len(df):,}")
    c2.metric("Inkonsisten", f"{df['label'].sum():,}")
    c3.metric("% Inkonsisten", f"{df['label'].mean()*100:.2f}%")
    c4.metric("Jumlah Aplikasi", str(df["app"].nunique()))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Label")
        counts = df["label"].map({0: "Konsisten", 1: "Inkonsisten"}).value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.4,
                     color_discrete_map={"Konsisten": "#2ecc71", "Inkonsisten": "#e74c3c"})
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("% Inkonsisten per Aplikasi")
        app_stat = (df.groupby("app")["label"]
                      .agg(inkonsisten="sum", total="count")
                      .assign(pct=lambda x: x["inkonsisten"] / x["total"] * 100)
                      .sort_values("pct"))
        fig = px.bar(app_stat.reset_index(), x="pct", y="app", orientation="h",
                     color="pct", color_continuous_scale="RdYlGn_r",
                     text=app_stat["pct"].round(2).astype(str).values,
                     labels={"pct": "% Inkonsisten", "app": ""})
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribusi Rating per Aplikasi")
        rd = df.groupby(["app", "rating"]).size().reset_index(name="n")
        fig = px.bar(rd, x="app", y="n", color="rating", barmode="stack",
                     color_continuous_scale="RdYlGn",
                     labels={"n": "Jumlah Review", "app": "", "rating": "Rating"})
        fig.update_layout(height=350, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Distribusi Segmen Teks")
        seg_cnt = df["segment"].value_counts().reindex(["Pendek", "Sedang", "Panjang"])
        fig = px.bar(x=["Pendek\n(<20 kata)", "Sedang\n(20-50 kata)", "Panjang\n(>50 kata)"],
                     y=seg_cnt.values,
                     color=["Pendek", "Sedang", "Panjang"],
                     color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
                     text=seg_cnt.values,
                     labels={"x": "", "y": "Jumlah Review"})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=350, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Sentimen LLM per Aplikasi")
    sd = df.groupby(["app", "text_sentiment"]).size().reset_index(name="n")
    fig = px.bar(sd, x="app", y="n", color="text_sentiment", barmode="group",
                 color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"},
                 labels={"n": "Jumlah Review", "app": "", "text_sentiment": "Sentimen"})
    fig.update_layout(height=350, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2: Eksplorasi Data
# ══════════════════════════════════════════════════════════════
elif PAGE == "Eksplorasi Data":
    st.title("Eksplorasi Data")

    df = load_data()
    if df is None:
        no_data()

    with st.sidebar:
        st.markdown("### Filter")
        apps = ["Semua"] + sorted(df["app"].unique().tolist())
        sel_app    = st.selectbox("Aplikasi", apps)
        sel_rating = st.multiselect("Rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        sel_label  = st.radio("Label", ["Semua", "Konsisten", "Inkonsisten"])
        sel_seg    = st.multiselect("Segmen", ["Pendek", "Sedang", "Panjang"],
                                    default=["Pendek", "Sedang", "Panjang"])

    fdf = df.copy()
    if sel_app != "Semua":
        fdf = fdf[fdf["app"] == sel_app]
    if sel_rating:
        fdf = fdf[fdf["rating"].isin(sel_rating)]
    if sel_label == "Konsisten":
        fdf = fdf[fdf["label"] == 0]
    elif sel_label == "Inkonsisten":
        fdf = fdf[fdf["label"] == 1]
    if sel_seg:
        fdf = fdf[fdf["segment"].isin(sel_seg)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Ditampilkan", f"{len(fdf):,}")
    c2.metric("Inkonsisten", f"{fdf['label'].sum():,}")
    c3.metric("% Inkonsisten", f"{fdf['label'].mean()*100:.2f}%" if len(fdf) else "—")

    disp = fdf[["app", "rating", "text", "text_sentiment", "label", "word_count", "segment"]].copy()
    disp["label"] = disp["label"].map({0: "✅ Konsisten", 1: "❌ Inkonsisten"})
    st.dataframe(
        disp.rename(columns={"app": "Aplikasi", "rating": "Rating", "text": "Teks Review",
                              "text_sentiment": "Sentimen LLM", "label": "Label",
                              "word_count": "Kata", "segment": "Segmen"}),
        height=460, use_container_width=True,
    )
    st.download_button(
        "Download CSV (hasil filter)",
        fdf.to_csv(index=False).encode("utf-8-sig"),
        "filtered_reviews.csv", "text/csv",
    )

# ══════════════════════════════════════════════════════════════
# PAGE 3: Skenario 1
# ══════════════════════════════════════════════════════════════
elif PAGE == "Skenario 1: Encoding Rating":
    st.title("Skenario 1: Encoding Rating ke IndoBERT")

    st.markdown("""
    **Pertanyaan:** Dari tiga cara mengintegrasikan informasi rating ke dalam IndoBERT, mana yang paling efektif?

    | Variant | Cara Encoding | Input ke Model |
    |---|---|---|
    | **A** | Fitur Numerik | `teks` → `[CLS]` ⊕ `(rating/5.0)` |
    | **B** | Rating Embedding | `teks` → `[CLS]` ⊕ `Embedding(rating, dim=32)` |
    | **C** | Token di Teks | `[RATING_X] teks` → encoder bersama |

    **Setup:** `indobenchmark/indobert-base-p1` · MAX_LEN=128 · EPOCHS=5 · LR=2e-5 · BATCH=32
    · Dataset balanced 52.284 sampel (rasio 1:3) · Split 70/15/15
    """)

    st.markdown("---")
    st.subheader("Hasil Test Set")

    summary = pd.DataFrame([
        {"Variant": S1[v]["name"], "Deskripsi": S1[v]["desc"],
         "Test Accuracy": S1[v]["acc"], "Test F1-Score": S1[v]["f1"]}
        for v in ["A", "B", "C"]
    ])
    st.dataframe(
        summary.style.highlight_max(subset=["Test Accuracy", "Test F1-Score"], color="#d4edda"),
        hide_index=True, use_container_width=True,
    )

    names = [S1[v]["name"] for v in ["A", "B", "C"]]
    fig = go.Figure([
        go.Bar(name="Accuracy", x=names, y=[S1[v]["acc"] for v in ["A","B","C"]],
               text=[f"{S1[v]['acc']:.4f}" for v in ["A","B","C"]], textposition="outside"),
        go.Bar(name="F1-Score", x=names, y=[S1[v]["f1"] for v in ["A","B","C"]],
               text=[f"{S1[v]['f1']:.4f}" for v in ["A","B","C"]], textposition="outside"),
    ])
    fig.update_layout(barmode="group", yaxis_range=[0, 1.12],
                      title="Test Accuracy & F1-Score per Variant", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Detail per Variant")
    tabs = st.tabs(["Variant A", "Variant B", "Variant C (terbaik)"])
    for tab, v in zip(tabs, ["A", "B", "C"]):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    loss_acc_chart(S1[v]["history"], 5, S1[v]["name"]),
                    use_container_width=True,
                )
            with col2:
                st.markdown(f"**Classification Report**")
                st.dataframe(report_df(S1[v]["report"]), hide_index=True, use_container_width=True)

    st.info("**Kesimpulan:** Variant C terbaik (F1=0.9678) — token `[RATING_X]` di awal teks memungkinkan self-attention encoder memperhatikan konteks rating secara langsung bersama seluruh teks.")

# ══════════════════════════════════════════════════════════════
# PAGE 4: Skenario 2
# ══════════════════════════════════════════════════════════════
elif PAGE == "Skenario 2: Perbandingan Model":
    st.title("Skenario 2: Perbandingan Arsitektur Model")

    st.markdown("""
    **Pertanyaan:** Seberapa besar gap performa antara model pretrained, DL non-pretrained, dan baseline klasik?

    | Model | Kategori | Handle Rating |
    |---|---|---|
    | TF-IDF + LR / SVC / NB | Baseline Klasik | Fitur numerik terpisah |
    | TextCNN | DL non-pretrained | Di-concat ke output CNN |
    | IndoBERT Variant C | DL pretrained | Token `[RATING_X]` di awal teks |

    **Setup baseline** mengikuti paper arXiv:2605.03439: `ngram_range=(1,2)`, `sublinear_tf=True`,
    `min_df=2`, `max_features=50000`, `class_weight='balanced'`.
    Dataset balanced 52.284 sampel · Split 70/15/15
    """)

    st.markdown("---")
    st.subheader("Hasil Test Set")

    keys = ["LR", "SVC", "NB", "CNN", "BERT"]
    summary = pd.DataFrame([
        {"Model": S2[k]["name"], "Kategori": S2[k]["cat"],
         "Accuracy": S2[k]["acc"], "F1 Weighted": S2[k]["f1_w"], "F1 Macro": S2[k]["f1_m"]}
        for k in keys
    ])
    st.dataframe(
        summary.style.highlight_max(subset=["Accuracy", "F1 Weighted", "F1 Macro"], color="#d4edda")
                     .highlight_min(subset=["Accuracy", "F1 Weighted", "F1 Macro"], color="#f8d7da"),
        hide_index=True, use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        names = [S2[k]["name"] for k in keys]
        fig = go.Figure([
            go.Bar(name="Accuracy",    x=names, y=[S2[k]["acc"]   for k in keys],
                   text=[f"{S2[k]['acc']:.3f}"   for k in keys], textposition="outside"),
            go.Bar(name="F1 Weighted", x=names, y=[S2[k]["f1_w"]  for k in keys],
                   text=[f"{S2[k]['f1_w']:.3f}"  for k in keys], textposition="outside"),
            go.Bar(name="F1 Macro",    x=names, y=[S2[k]["f1_m"]  for k in keys],
                   text=[f"{S2[k]['f1_m']:.3f}"  for k in keys], textposition="outside"),
        ])
        fig.update_layout(barmode="group", yaxis_range=[0, 1.12], height=420,
                          title="Accuracy & F1 per Model")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        best_klasik = max(S2[k]["f1_w"] for k in ["LR","SVC","NB"])
        gap_bert_cnn  = S2["BERT"]["f1_w"] - S2["CNN"]["f1_w"]
        gap_cnn_base  = S2["CNN"]["f1_w"]  - best_klasik
        st.markdown("### Gap F1 Weighted")
        st.metric("IndoBERT vs TextCNN", f"{gap_bert_cnn:+.4f}")
        st.metric("TextCNN vs Best Baseline", f"{gap_cnn_base:+.4f}", delta_color="inverse")
        st.markdown("""
        **Temuan utama:**
        - IndoBERT unggul jauh berkat pretraining pada korpus bahasa Indonesia besar
        - TextCNN dan baseline klasik memiliki performa yang mirip
        - Multinomial NB paling lemah karena sangat bias ke kelas mayoritas
        """)

    st.markdown("---")
    st.subheader("Training Curve")
    tab_cnn, tab_bert = st.tabs(["TextCNN (10 epoch)", "IndoBERT Var C (5 epoch)"])
    with tab_cnn:
        st.plotly_chart(loss_acc_chart(S2["CNN"]["history"], 10, "TextCNN"), use_container_width=True)
    with tab_bert:
        st.plotly_chart(loss_acc_chart(S2["BERT"]["history"], 5, "IndoBERT Variant C"), use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Report per Model")
    for k in keys:
        with st.expander(f"{S2[k]['name']} — Acc={S2[k]['acc']:.4f} | F1 Macro={S2[k]['f1_m']:.4f}"):
            st.dataframe(report_df(S2[k]["report"]), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5: Skenario 3
# ══════════════════════════════════════════════════════════════
elif PAGE == "Skenario 3: Panjang Teks":
    st.title("Skenario 3: Pengaruh Panjang Teks")

    st.markdown("""
    **Pertanyaan:** Di segmen panjang teks mana IndoBERT Variant C paling dan paling tidak akurat?

    **Model:** IndoBERT Variant C (terbaik dari Skenario 1 & 2)
    · Dataset balanced 52.284 sampel · Split 70/15/15
    """)

    st.markdown("---")
    st.subheader("Training Curve")
    ep = list(range(1, 6))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Accuracy & F1"])
    fig.add_trace(go.Scatter(x=ep, y=S3_HIST["train_loss"], name="Train Loss",
                             mode="lines+markers", line=dict(color="#2ecc71")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=S3_HIST["val_loss"],   name="Val Loss",
                             mode="lines+markers", line=dict(color="#2ecc71", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=S3_HIST["train_acc"],  name="Train Acc",
                             mode="lines+markers", line=dict(color="#3498db")), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=S3_HIST["val_acc"],    name="Val Acc",
                             mode="lines+markers", line=dict(color="#3498db", dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=S3_HIST["val_f1"],     name="Val F1",
                             mode="lines+markers", line=dict(color="#e74c3c", dash="dot")), row=1, col=2)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Ringkasan Performa per Segmen")

    segs = list(S3.keys())
    seg_df = pd.DataFrame([
        {"Segmen": s, "N Sample": S3[s]["n"], "N Inkonsisten": S3[s]["n_incon"],
         "Accuracy": S3[s]["acc"], "F1 Weighted": S3[s]["f1_w"],
         "F1 Macro": S3[s]["f1_m"], "F1 Inkonsisten": S3[s]["f1_i"]}
        for s in segs
    ])
    st.dataframe(
        seg_df.style
              .highlight_max(subset=["Accuracy","F1 Weighted","F1 Macro","F1 Inkonsisten"], color="#d4edda")
              .highlight_min(subset=["Accuracy","F1 Weighted","F1 Macro","F1 Inkonsisten"], color="#f8d7da"),
        hide_index=True, use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure([
            go.Bar(name="Accuracy",    x=segs, y=[S3[s]["acc"]   for s in segs],
                   text=[f"{S3[s]['acc']:.4f}"   for s in segs], textposition="outside"),
            go.Bar(name="F1 Weighted", x=segs, y=[S3[s]["f1_w"]  for s in segs],
                   text=[f"{S3[s]['f1_w']:.4f}"  for s in segs], textposition="outside"),
        ])
        fig.update_layout(barmode="group", yaxis_range=[0.93, 1.0],
                          title="Accuracy & F1 Weighted per Segmen", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure([
            go.Bar(name="F1 Macro",       x=segs, y=[S3[s]["f1_m"] for s in segs],
                   text=[f"{S3[s]['f1_m']:.4f}" for s in segs], textposition="outside"),
            go.Bar(name="F1 Inkonsisten", x=segs, y=[S3[s]["f1_i"] for s in segs],
                   text=[f"{S3[s]['f1_i']:.4f}" for s in segs], textposition="outside"),
        ])
        fig.update_layout(barmode="group", yaxis_range=[0.88, 0.98],
                          title="F1 Macro & F1 Inkonsisten per Segmen", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Report & Analisis Error per Segmen")
    for s in segs:
        with st.expander(f"Segmen {s} — n={S3[s]['n']:,} | FN={S3[s]['fn']} | FP={S3[s]['fp']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Classification Report:**")
                st.dataframe(report_df(S3[s]["report"]), hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**Contoh False Negative** _(inkonsisten diprediksi konsisten)_:")
                st.dataframe(pd.DataFrame(FN_EXAMPLES[s]), hide_index=True, use_container_width=True)

    st.info("""
    **Kesimpulan:**
    - **Panjang** terbaik (F1 Inkonsisten=0.9401) — cukup sinyal linguistik untuk menangkap kontradiksi
    - **F1 Inkonsisten terendah** di Pendek (0.9251) — konteks teks singkat paling terbatas
    - Perbedaan antar segmen sangat kecil (<1%), model robust di semua panjang teks
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 6: Demo Prediksi
# ══════════════════════════════════════════════════════════════
elif PAGE == "Demo Prediksi":
    st.title("Demo Prediksi Inkonsistensi")
    st.markdown("Masukkan teks review dan rating untuk mendeteksi inkonsistensi antara sentimen teks dan bintang yang diberikan.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Input")
        text_input = st.text_area(
            "Teks Review",
            height=160,
            placeholder="Contoh: Aplikasi sangat bagus dan mudah digunakan, tidak ada kendala sama sekali!",
        )
        rating_input = st.slider("Rating Bintang", 1, 5, 5)
        st.markdown("&nbsp;&nbsp;" + "⭐" * rating_input + "☆" * (5 - rating_input))

        use_bert = st.checkbox("IndoBERT S3 (terbaik)", value=True)
        use_lr   = st.checkbox("TF-IDF + Logistic Regression", value=True)
        use_cnn  = st.checkbox("TextCNN", value=True)

        predict_btn = st.button("Prediksi", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil Prediksi")

        if predict_btn:
            if not text_input.strip():
                st.warning("Masukkan teks review terlebih dahulu.")
            else:
                if use_bert:
                    bert_model, tokenizer = load_indobert_s3()
                    if bert_model is None:
                        st.info("**IndoBERT S3** — model belum tersedia. Jalankan `skenario3_text_length.ipynb` terlebih dahulu.")
                    else:
                        pred, probs = predict_bert(text_input, rating_input, bert_model, tokenizer)
                        label = "INKONSISTEN" if pred == 1 else "KONSISTEN"
                        fn = st.error if pred == 1 else st.success
                        fn(f"**IndoBERT S3:** {label}  \nKonfiden: {probs[pred]*100:.1f}%")
                        c1, c2 = st.columns(2)
                        c1.metric("P(Konsisten)",   f"{probs[0]*100:.1f}%")
                        c2.metric("P(Inkonsisten)", f"{probs[1]*100:.1f}%")
                        st.progress(float(probs[1]), text=f"Prob. Inkonsisten: {probs[1]*100:.1f}%")
                        with st.expander("Logic Flow — cara model memutuskan"):
                            render_logic_flow(text_input, rating_input, pred, probs, "IndoBERT S3")
                        st.markdown("---")

                if use_lr:
                    tfidf_m, lr_m = load_tfidf_lr()
                    if tfidf_m is None:
                        st.info("**TF-IDF + LR** — model belum tersedia. Jalankan `skenario2_model_comparison.ipynb` terlebih dahulu.")
                    else:
                        pred, probs = predict_lr(text_input, rating_input, tfidf_m, lr_m)
                        label = "INKONSISTEN" if pred == 1 else "KONSISTEN"
                        fn = st.error if pred == 1 else st.success
                        fn(f"**TF-IDF + LR:** {label}  \nKonfiden: {probs[pred]*100:.1f}%")
                        c1, c2 = st.columns(2)
                        c1.metric("P(Konsisten)",   f"{probs[0]*100:.1f}%")
                        c2.metric("P(Inkonsisten)", f"{probs[1]*100:.1f}%")
                        st.progress(float(probs[1]), text=f"Prob. Inkonsisten: {probs[1]*100:.1f}%")
                        with st.expander("Logic Flow — cara model memutuskan"):
                            render_logic_flow(text_input, rating_input, pred, probs, "TFIDF LR")
                        st.markdown("---")

                if use_cnn:
                    cnn_m, cnn_ckpt = load_textcnn()
                    if cnn_m is None:
                        st.info("**TextCNN** — model belum tersedia. Jalankan `skenario2_model_comparison.ipynb` terlebih dahulu.")
                    else:
                        pred, probs = predict_cnn(text_input, rating_input, cnn_m, cnn_ckpt)
                        label = "INKONSISTEN" if pred == 1 else "KONSISTEN"
                        fn = st.error if pred == 1 else st.success
                        fn(f"**TextCNN:** {label}  \nKonfiden: {probs[pred]*100:.1f}%")
                        c1, c2 = st.columns(2)
                        c1.metric("P(Konsisten)",   f"{probs[0]*100:.1f}%")
                        c2.metric("P(Inkonsisten)", f"{probs[1]*100:.1f}%")
                        st.progress(float(probs[1]), text=f"Prob. Inkonsisten: {probs[1]*100:.1f}%")
                        with st.expander("Logic Flow — cara model memutuskan"):
                            render_logic_flow(text_input, rating_input, pred, probs, "TextCNN")

    st.markdown("---")
    with st.expander("Contoh kasus untuk dicoba"):
        st.markdown("""
        | Rating | Teks | Ekspektasi |
        |---|---|---|
        | 1★ | "Aplikasi sangat bagus dan membantu sekali, fiturnya lengkap dan tidak ada lag sama sekali!" | Inkonsisten (rating rendah, teks positif) |
        | 5★ | "Aplikasi sangat mengecewakan, sering error dan data saya hilang. Sangat tidak rekomendasikan!" | Inkonsisten (rating tinggi, teks negatif) |
        | 1★ | "Tidak bisa dibuka sama sekali, sudah coba uninstall berkali-kali tapi tetap error terus." | Konsisten (rating rendah, teks negatif) |
        | 5★ | "Sangat puas dengan layanannya, pengiriman cepat dan barang sesuai deskripsi. Recommended!" | Konsisten (rating tinggi, teks positif) |
        """)
