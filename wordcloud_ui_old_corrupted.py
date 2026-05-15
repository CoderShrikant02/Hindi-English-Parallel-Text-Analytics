from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import matplotlib.pyplot as plt
import streamlit as st
from datasets import load_dataset
from wordcloud import STOPWORDS, WordCloud

st.set_page_config(page_title="IITB EN-HI Word Cloud", page_icon="☁️", layout="wide")

DEFAULT_EN_STOPWORDS = set(STOPWORDS).union(
    {
        "would",
        "could",
        "also",
        "one",
        "two",
        "us",
        "like",
        "get",
        "got",
        "make",
        "made",
        "using",
    }
)

DEFAULT_HI_STOPWORDS = {
    "है",
    "हैं",
    "था",
    "थे",
    "थी",
    "को",
    "का",
    "की",
    "के",
    "में",
    "पर",
    "और",
    "से",
    "यह",
    "वह",
    "तो",
    "भी",
    "नहीं",
    "एक",
    "इस",
    "उस",
    "लिए",
    "कर",
    "रहा",
    "रही",
    "रहे",
}


def clean_text(text: str, language: str) -> str:
    text = text.lower().strip()
    if language == "English":
        text = re.sub(r"[^a-z\s]", " ", text)
    else:
        text = re.sub(r"[^\u0900-\u097f\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def top_words(texts: Iterable[str], n: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())
    return counter.most_common(n)


@st.cache_data(show_spinner=False)
def fetch_texts(split_expr: str, sample_size: int, language: str) -> list[str]:
    ds = load_dataset("cfilt/iitb-english-hindi", split=split_expr)
    max_rows = min(sample_size, len(ds))
    key = "en" if language == "English" else "hi"

    collected: list[str] = []
    for i in range(max_rows):
        raw = ds[i].get("translation", {}).get(key, "")
        text = clean_text(raw, language)
        if text:
            collected.append(text)
    return collected


st.title("Word Cloud Experiment: cfilt/iitb-english-hindi")
st.caption("Dataset loaded from Hugging Face - Interactive word cloud generator.")

with st.sidebar:
    st.header("Experiment Controls")
    split_expr = st.selectbox("Split", ["train", "validation", "test"], index=0)
    sample_size = st.slider("Sample Size", min_value=1000, max_value=50000, value=12000, step=1000)
    language = st.radio("Language", ["English", "Hindi"], index=1)
    max_words = st.slider("Max Words in Cloud", min_value=50, max_value=500, value=200, step=25)
    width = st.number_input("Width", min_value=400, max_value=3000, value=1400, step=100)
    height = st.number_input("Height", min_value=300, max_value=2000, value=800, step=100)
    bg_color = st.selectbox("Background Color", ["white", "black", "ivory", "mintcream"], index=0)
    extra_stopwords = st.text_area(
        "Extra Stopwords (comma separated)",
        placeholder="example, words, to, ignore",
    )
    hindi_font_path = st.text_input(
        "Hindi Font Path (optional)",
        value="",
        help="For Hindi word clouds - provide .ttf font path (example: C:/Windows/Fonts/mangal.ttf)",
    )

run = st.button("Generate Word Cloud", type="primary", use_container_width=True)

if run:
    with st.spinner("Loading dataset and generating word cloud..."):
        texts = fetch_texts(split_expr, sample_size, language)

        if not texts:
            st.error("No text data found. Try changing split or sample size.")
            st.stop()

        base_sw = DEFAULT_EN_STOPWORDS if language == "English" else DEFAULT_HI_STOPWORDS
        user_sw = {w.strip().lower() for w in extra_stopwords.split(",") if w.strip()}
        stopwords = set(base_sw).union(user_sw)

        font_path = hindi_font_path.strip() or None
        if language == "Hindi" and not font_path:
            st.info("Tip: If Hindi words appear as boxes/squares, set the Hindi font path in the sidebar.")

        full_text = " ".join(texts)
        wc = WordCloud(
            width=int(width),
            height=int(height),
            background_color=bg_color,
            max_words=max_words,
            stopwords=stopwords,
            collocations=False,
            font_path=font_path,
        ).generate(full_text)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Quick Stats")
        st.metric("Language", language)
        st.metric("Loaded Sentences", f"{len(texts):,}")
        st.metric("Unique Tokens", f"{len(set(full_text.split())):,}")

        st.subheader("Top 20 Words")
        top20 = top_words(texts, 20)
        st.table({"word": [w for w, _ in top20], "count": [c for _, c in top20]})

st.markdown("---")
st.markdown(
    "Run command: `streamlit run wordcloud_ui.py`  \\\nOptional downloader: `python download_iitb_dataset.py --split train[:5000] --out iitb_sample.csv`"
)
