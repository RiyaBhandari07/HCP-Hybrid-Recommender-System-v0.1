#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
import tensorflow as tf
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ---------------- Page Config ---------------------
st.set_page_config(page_title="HCP Content Recommender", layout="wide")

# ---------------- Header Banner -------------------

st.markdown(
    """
    <style>
    .header-banner {
        background: linear-gradient(90deg, #0077B6 0%, #0096C7 50%, #00B4D8 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: white;
    }
    .header-banner p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    </style>

    <div class="header-banner">
        <h1>🏥 RWE Analytics | Next Best Action Recommender</h1>
        <p>Personalized Content & Channel Recommendations for Healthcare Professionals</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 1. Load CSV Data
# -------------------------------
@st.cache_data
def load_data():
    scientific_content_df = pd.read_csv("C:/Users/bhand/Desktop/Data Science - My Collection/Deep Learning Project - 1/Data/Data for Recommender System/HCP_Hybrid_Recommendation_System/artifacts/scientific_content.csv")
    hcp_interaction_df = pd.read_csv("C:/Users/bhand/Desktop/Data Science - My Collection/Deep Learning Project - 1/Data/Data for Recommender System/HCP_Hybrid_Recommendation_System/artifacts/hcp_interaction_data.csv")
    return scientific_content_df, hcp_interaction_df

scientific_content_df, hcp_interaction_df = load_data()

# Paths
# ---------------- Paths ----------------
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"

MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_lstm_model.keras")
CONTENT_META = os.path.join(ARTIFACTS_DIR, "content_meta.csv")
CONTENT_PAD = os.path.join(ARTIFACTS_DIR, "content_pad.npy")
CID2IX = os.path.join(ARTIFACTS_DIR, "content_id_to_ix.pkl")
HCP_OHE = os.path.join(ARTIFACTS_DIR, "hcp_ohe.pkl")
CHAN_OHE = os.path.join(ARTIFACTS_DIR, "chan_ohe.pkl")
HCP_MAP = os.path.join(ARTIFACTS_DIR, "hcp_spec_map.pkl")
CHANNELS_JSON = os.path.join(ARTIFACTS_DIR, "channels.json")
HCP_LIST = os.path.join(ARTIFACTS_DIR, "hcp_list.json")
TOPIC_COLS = os.path.join(ARTIFACTS_DIR, "topic_cols.json")
CONFIG = os.path.join(ARTIFACTS_DIR, "config.json")


# ---------------- Load Artifacts ----------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    content_meta = pd.read_csv(CONTENT_META)
    content_pad = np.load(CONTENT_PAD)
    with open(CID2IX, "rb") as f:
        content_id_to_ix = pickle.load(f)
    with open(HCP_OHE, "rb") as f:
        hcp_ohe = pickle.load(f)
    with open(CHAN_OHE, "rb") as f:
        chan_ohe = pickle.load(f)
    with open(HCP_MAP, "rb") as f:
        hcp_spec_map = pickle.load(f)
    with open(CHANNELS_JSON, "r") as f:
        channels = json.load(f)
    with open(HCP_LIST, "r") as f:
        hcp_list = json.load(f)
    with open(TOPIC_COLS, "r") as f:
        topic_cols = json.load(f)
    with open(CONFIG, "r") as f:
        config = json.load(f)

    # Precompute content features
    content_lda_feats = content_meta[topic_cols].values.astype(np.float32)
    return {
        "model": model,
        "content_meta": content_meta,
        "content_pad": content_pad,
        "content_id_to_ix": content_id_to_ix,
        "hcp_ohe": hcp_ohe,
        "chan_ohe": chan_ohe,
        "hcp_spec_map": hcp_spec_map,
        "channels": channels,
        "hcp_list": hcp_list,
        "topic_cols": topic_cols,
        "content_lda_feats": content_lda_feats,
        "config": config
    }

art = load_artifacts()
model = art["model"]
content_meta = art["content_meta"]
content_pad = art["content_pad"]
content_id_to_ix = art["content_id_to_ix"]
hcp_ohe = art["hcp_ohe"]
chan_ohe = art["chan_ohe"]
hcp_spec_map = art["hcp_spec_map"]
CHANNELS = art["channels"]
hcp_list = art["hcp_list"]
topic_cols = art["topic_cols"]
content_lda_feats = art["content_lda_feats"]
cfg = art["config"]
MAX_SEQ_LEN = cfg["MAX_SEQ_LEN"]

# ---------------- Utilities ----------------
def clean_title(title):
    """Clean up title text safely."""
    if not isinstance(title, str):
        return ""
    return (
        title.replace("[", "")
             .replace("]", "")
             .strip()
    )

content_ids = scientific_content_df['content_id'].astype(str).values
content_text_seqs = content_pad  # sequences aligned with content_df
n_topics = len(topic_cols)
hcp_spec_map = {str(row['hcp_id']): row['hcp_specialty'] for _, row in hcp_interaction_df.iterrows()}

# ---------------- Inference ----------------
def recommend_top_n_content(hcp_id, top_n=5, channels=None, candidate_content_ids=None):
    if channels is None:
        channels = CHANNELS
    if candidate_content_ids is None:
        candidate_content_ids = content_ids

    hcp_spec = hcp_spec_map[hcp_id]
    hcp_vec = hcp_ohe.transform(pd.DataFrame([[hcp_spec]], columns=['hcp_specialty'])).astype(np.float32)

    rows = []
    for cid in candidate_content_ids:
        idx = content_id_to_ix[cid]
        seq = np.expand_dims(content_text_seqs[idx], 0)             # shape (1, seq_len)
        lda_feat = content_lda_feats[idx].reshape(1, -1)           # shape (1, n_topics)
        
        # Repeat features for all channels
        num_channels = len(channels)
        seq_mat = np.repeat(seq, num_channels, axis=0)
        lda_mat = np.repeat(lda_feat, num_channels, axis=0)
        hcp_mat = np.repeat(hcp_vec, num_channels, axis=0)
        
        # Transform channels correctly
        ch_ohe_mat = np.vstack([
        chan_ohe.transform(pd.DataFrame([[ch]], columns=['campaign_channel'])).astype(np.float32)
         for ch in CHANNELS
          ])
        
        preds = model.predict([seq_mat, lda_mat, hcp_mat, ch_ohe_mat], verbose=0).ravel()
        best_idx = preds.argmax()
        best_chan = channels[best_idx]
        best_prob = preds[best_idx]
        title = scientific_content_df.loc[scientific_content_df['content_id'] == cid, 'Title'].values[0]
        
        rows.append((cid, clean_title(title), best_chan, best_prob))

    # Sort top-N
    rows = sorted(rows, key=lambda x: x[3], reverse=True)[:top_n]
    return rows

# Function to find best channel for a given HCP + content
def best_channel_for(hcp_id, content_id):
    """
    Returns the best channel, predicted probability, and content title
    for a given HCP and content.
    """
    hcp_spec = hcp_spec_map[hcp_id]
    hcp_vec = hcp_ohe.transform(pd.DataFrame([[hcp_spec]], columns=['hcp_specialty'])).astype(np.float32)
    
    idx = content_id_to_ix[content_id]
    seq = np.expand_dims(content_text_seqs[idx], 0)              # shape (1, seq_len)
    lda_feat = content_lda_feats[idx].reshape(1, -1)            # shape (1, n_topics)
    
    num_channels = len(CHANNELS)
    seq_mat = np.repeat(seq, num_channels, axis=0)
    lda_mat = np.repeat(lda_feat, num_channels, axis=0)
    hcp_mat = np.repeat(hcp_vec, num_channels, axis=0)
    
    ch_ohe_mat = np.vstack([
    chan_ohe.transform(pd.DataFrame([[ch]], columns=['campaign_channel'])).astype(np.float32)
    for ch in CHANNELS
    ])
    
    preds = model.predict([seq_mat, lda_mat, hcp_mat, ch_ohe_mat], verbose=0).ravel()
    best_idx = preds.argmax()
    
    best_channel = CHANNELS[best_idx]
    best_prob = preds[best_idx]
    title = scientific_content_df.loc[scientific_content_df['content_id'] == content_id, 'Title'].values[0]
    
    return best_channel, best_prob, clean_title(title)

# Function to compute KPIs for a given HCP + optional channel
def compute_kpis(hcp_id, top_n=5, channel=None):
    """
    Returns a dictionary of KPIs with top-N titles + probs only.
    """
    # Get top-N content
    top_content = recommend_top_n_content(
        hcp_id, top_n=top_n, channels=[channel] if channel else None
    )
    if not top_content:
        return {}
    
    # Each item in top_content = (cid, title, channel, prob)
    top_content_with_title = [
        (clean_title(title), round(prob, 3))
        for cid, title, ch, prob in top_content
    ]
    
    # Predicted scores
    pred_scores = np.array([prob for _, prob in top_content_with_title])
    
    # Best channel score for first content
    best_channel_score = None
    if top_content_with_title:
        cid_first = top_content[0][0]   # content_id from original tuple
        _, best_channel_score, _ = best_channel_for(hcp_id, cid_first)
    
    kpi = {
        'avg_pred_engagement': round(pred_scores.mean(), 3),
        'high_engagement_pct': round((pred_scores > 0.5).sum() / len(pred_scores) * 100, 1),
        'top_channel_score': round(best_channel_score, 3) if best_channel_score else None,
        'top_n_count': len(top_content_with_title),
        'top_content_list': top_content_with_title  # now only (title, prob)
    }
    return kpi


# ------------------- Plotly Theme Helper -------------------
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "tealrose"
px.defaults.height = 500

def style_plot(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#0F172A", size=18)),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ---------------- Streamlit Layout ----------------
st.markdown(
    """
    <div style="font-size:16px; color:#475569; margin-top:-10px;">
        Powered by Hybrid LSTM + Topic Modeling | Built with Streamlit, TensorFlow & Plotly
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

st.subheader("💡 HCP Engagement & Recommendation Dashboard")
st.markdown(
    """
    <div style="font-size:17px; color:#334155;">
        This interactive dashboard predicts <b>Next Best Content (NBC)</b> and <b>Next Best Channel (NBC)</b>
        for HCPs using engagement patterns and content embeddings.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
# Sidebar
st.sidebar.header("Controls")
selected_hcp = st.sidebar.selectbox("Select HCP ID", options=["-- select --"] + hcp_list)
top_n = st.sidebar.slider("Top Number of Content", 3, 10, 5)
channels_select = st.sidebar.multiselect( "Select Channel(s):", options= CHANNELS,  default=None)

# ---------------- Main Display ----------------
if selected_hcp == "-- select --":
    st.info("Choose an HCP ID from the sidebar to view recommendations.")
else:
    st.markdown(f"**HCP: {selected_hcp}** —> **Specialty: {hcp_spec_map.get(selected_hcp, 'Unknown')}**")

        # --- Banner-style Overall Summary Metrics ---
    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content: space-around;
            background: linear-gradient(90deg, #48CAE4 0%, #00B4D8 50%, #0096C7 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        ">
            <div style="text-align:center">
                <h3 style="margin:0; font-size:1.2rem;">👨‍⚕️ Total HCPs</h3>
                <p style="margin:0; font-size:1.5rem; font-weight:700;">{len(hcp_list)}</p>
            </div>
            <div style="text-align:center">
                <h3 style="margin:0; font-size:1.2rem;">📚 Total Contents</h3>
                <p style="margin:0; font-size:1.5rem; font-weight:700;">{len(content_ids)}</p>
            </div>
            <div style="text-align:center">
                <h3 style="margin:0; font-size:1.2rem;">📡 Total Channels</h3>
                <p style="margin:0; font-size:1.5rem; font-weight:700;">{len(CHANNELS)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
  
    # Safely get first selected channel if only one is selected
    selected_channel = channels_select[0] if channels_select and len(channels_select) == 1 else None

    # Compute KPIs
    kpi = compute_kpis(selected_hcp, top_n=top_n, channel=selected_channel)

    kpi_container = st.container()
    with kpi_container:
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    
        # Formatting with 4 decimal points
        avg_engagement = kpi.get("avg_pred_engagement", None)
        high_eng_pct = kpi.get("high_engagement_pct", None)
        top_channel_score = kpi.get("top_channel_score", None)
        top_n_count = kpi.get("top_n_count", None)

        kcol1.metric("💠 Avg Pred Engagement", f"{avg_engagement:.3f}" if avg_engagement is not None else "N/A")
        kcol2.metric("📈 High Engagement % (>0.5)", f"{high_eng_pct:.2f}%" if high_eng_pct is not None else "N/A")
        kcol3.metric("📊 Top Channel Score", f"{top_channel_score:.3f}" if top_channel_score is not None else "N/A")
        kcol4.metric("📚 Top-N Count", f"{top_n_count:.3f}" if top_n_count is not None else "N/A")

        st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("📚 Top Recommended Content (Title and Probability)")
    top_df = pd.DataFrame(kpi.get("top_content_list", []), columns=["Title","Predicted Prob"])
    # Format the probability column to 4 decimal places
    if not top_df.empty:
        top_df["Predicted Prob"] = top_df["Predicted Prob"].apply(lambda x: f"{x:.3f}")
    st.table(top_df) 

    if kpi.get("top_content_list"):
        first_cid = recommend_top_n_content(selected_hcp, top_n=1)[0][0]
        best_chan, best_prob, best_title = best_channel_for(selected_hcp, first_cid)
        st.info(f"Best channel for top content **{best_title}** → **{best_chan}** (Pred: {best_prob:.3f})")

    with st.expander("Show full Top-N details (content id, title, channel, prob)"):
        details = recommend_top_n_content(selected_hcp, top_n=top_n, channels=channels_select if channels_select else None)
        det_df = pd.DataFrame(details, columns=["content_id", "Title", "Channel", "Predicted Prob"])
        if not det_df.empty:
            det_df["Predicted Prob"] = det_df["Predicted Prob"].apply(lambda x: f"{x:.3f}")
        st.dataframe(det_df)

# ---------------- Visualization Section ----------------
st.markdown("---")
st.subheader("📊 Engagement Insights & Content Analysis")

if 'top_df' in locals() and not top_df.empty:
    # 1️⃣ BAR CHART
    fig_bar = px.bar(
        top_df, x="Title", y="Predicted Prob", color="Predicted Prob", text="Predicted Prob"
    )
    fig_bar = style_plot(fig_bar, "Engagement Probabilities (Top-N Content)")
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # 2️⃣ SCATTER PLOT
    top_titles = " ".join(top_df["Title"].astype(str).tolist())
    vectorizer = CountVectorizer(stop_words="english", max_features=10)
    X = vectorizer.fit_transform([top_titles])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().flatten()))
    word_df = pd.DataFrame({"Word": list(word_freq.keys()), "Frequency": list(word_freq.values())}).sort_values(by="Frequency", ascending=False)
    fig_scatter = px.scatter(word_df, x="Word", y="Frequency", size="Frequency", color="Frequency")
    fig_scatter = style_plot(fig_scatter, "Top 10 Words in Recommended Titles")

    # 3️⃣ WORD CLOUD
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="coolwarm").generate(top_titles)
    fig_wc, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("Word Cloud of Top Recommended Content", fontsize=14)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### ☁️ Keyword Cloud Overview")
    st.pyplot(fig_wc)
else:
    st.info("No recommendations available to generate visualizations.")

