import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from wordcloud import STOPWORDS, WordCloud

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="English-Hindi Translation Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-label {
        font-size: 13px;
        color: #a6adc8;
        margin-bottom: 6px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #cdd6f4;
    }
    .metric-sub {
        font-size: 12px;
        color: #6c7086;
        margin-top: 4px;
    }
    .badge-good  { color: #a6e3a1; font-weight: 700; }
    .badge-mid   { color: #f9e2af; font-weight: 700; }
    .badge-bad   { color: #f38ba8; font-weight: 700; }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #cdd6f4;
        border-left: 4px solid #89b4fa;
        padding-left: 12px;
        margin-bottom: 16px;
    }
    .translation-box {
        background: #181825;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 3px solid #89b4fa;
    }
    .trans-label {
        font-size: 11px;
        color: #6c7086;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .trans-text { color: #cdd6f4; font-size: 15px; }
    .score-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: SAFE IMAGE LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def show_image(filename, caption="", image_width=900):
    path = Path(filename)
    if path.exists():
        st.image(str(path), caption=caption, width=image_width)
    else:
        st.warning(f"⚠️ Image not found: `{filename}` – make sure it's in the same folder as app.py")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: SAFE JSON LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_json(filename):
    path = Path(filename)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


DEFAULT_EN_STOPWORDS = set(STOPWORDS).union(
    {"would", "could", "also", "one", "two", "us", "like", "get", "got", "make", "made", "using"}
)
DEFAULT_HI_STOPWORDS = {
    "है", "हैं", "था", "थे", "थी", "को", "का", "की", "के", "में", "पर", "और",
    "से", "यह", "वह", "तो", "भी", "नहीं", "एक", "इस", "उस", "लिए", "कर", "रहा", "रही", "रहे",
}


def clean_text_wc(text, language):
    text = str(text).lower().strip()
    if language == "English":
        text = re.sub(r"[^a-z\s]", " ", text)
    else:
        text = re.sub(r"[^\u0900-\u097f\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def top_words_wc(texts, n=20):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    return counter.most_common(n)


@st.cache_data(show_spinner=False)
def load_iitb_texts(split_expr, sample_size, language):
    ds = load_dataset("cfilt/iitb-english-hindi", split=split_expr)
    key = "en" if language == "English" else "hi"
    max_rows = min(sample_size, len(ds))

    texts = []
    for i in range(max_rows):
        value = ds[i].get("translation", {}).get(key, "")
        cleaned = clean_text_wc(value, language)
        if cleaned:
            texts.append(cleaned)
    return texts


@st.cache_resource(show_spinner=False)
def load_en_hi_model(model_id="Helsinki-NLP/opus-mt-en-hi"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model


def translate_en_to_hi(text, max_new_tokens=128, num_beams=4):
    tokenizer, model = load_en_hi_model()
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌐 English-Hindi Analyzer")
    st.markdown("**SMA Internal Assessment**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview",
         "📈 EDA & Vocabulary",
         "🔤 Translation Samples",
         "📉 Evaluation Scores",
         "🔍 Error Analysis",
         "🧠 Topic Modeling"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** Helsinki-NLP/opus-mt-en-hi")
    st.markdown("**Dataset:** cfilt/iitb-english-hindi")
    st.markdown("**Train:** 50,000 pairs")
    st.markdown("**Test:** 2,507 pairs")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🌐 English → Hindi Translation Analyzer")
    st.markdown("##### Powered by Helsinki-NLP · IIT Bombay Corpus · SMA IA Project")
    st.markdown("---")

    # Load scores
    scores = load_json("evaluation_scores.json")

    if scores:
        bleu  = scores.get("BLEU", "–")
        chrf  = scores.get("chrF", "–")
        meteor = scores.get("METEOR", "–")
    else:
        bleu, chrf, meteor = "–", "–", "–"
        st.warning("evaluation_scores.json not found in this folder.")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BLEU Score</div>
            <div class="metric-value">{bleu}</div>
            <div class="metric-sub">Scale: 0-100</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">chrF Score</div>
            <div class="metric-value">{chrf}</div>
            <div class="metric-sub">Scale: 0-100</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">METEOR</div>
            <div class="metric-value">{meteor}</div>
            <div class="metric-sub">Scale: 0-1</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Sentences</div>
            <div class="metric-value">2,507</div>
            <div class="metric-sub">Official test split</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Project Pipeline")
    steps = [
        ("1️⃣", "Setup & Load Dataset", "IIT Bombay English-Hindi corpus via HuggingFace"),
        ("2️⃣", "Preprocessing & Cleaning", "IndicNLP normalizer + regex cleaning"),
        ("3️⃣", "EDA", "Sentence length, vocab size, TTR, word frequency"),
        ("4️⃣", "Machine Translation", "Helsinki-NLP/opus-mt-en-hi (MarianMT)"),
        ("5️⃣", "Evaluation", "BLEU, chrF, METEOR scores"),
        ("6️⃣", "Cross-Lingual Analysis", "SVO vs SOV, TTR, word length comparison"),
        ("7️⃣", "Error Analysis", "Good / Medium / Bad categorization via chrF"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"**{icon} {title}** – {desc}")

    st.markdown("---")
    st.info("💡 **Key Insight:** BLEU of 6.91 is low but expected for English→Hindi. Hindi has complex morphology and a different script, so BLEU always underestimates quality. chrF (27.09) is the more reliable metric here.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA & VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA & Vocabulary":
    st.markdown('<div class="section-header">EDA & Vocabulary Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📏 Sentence Lengths",
        "📊 Word Frequency",
        "📚 Vocabulary Coverage",
        "🧮 Type-Token Ratio"
    ])

    with tab1:
        st.markdown("#### Sentence Length Distribution (EN vs HI)")
        show_image("sentence_lengths.png")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**English Stats**")
            st.markdown("- Avg: **4.42 words**\n- Max: 115 words\n- Min: 0 words")
        with col2:
            st.markdown("**Hindi Stats**")
            st.markdown("- Avg: **4.02 words**\n- Max: 113 words\n- Min: 0 words")

    with tab2:
        st.markdown("#### Top 20 Most Frequent Words")
        show_image("word_frequency.png")
        st.markdown("""
        | Language | Vocab Size | Top Word |
        |----------|-----------|----------|
        | English  | 4,459 unique words | "the" (12,754×) |
        | Hindi    | 4,226 unique words | के (6,300×) |
        """)

    with tab3:
        st.markdown("#### Vocabulary Coverage (Total vs Unique Words)")
        show_image("vocabulary_coverage.png")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("English Total Words", "2,20,778")
            st.metric("English Unique Words", "4,459")
        with col2:
            st.metric("Hindi Total Words", "2,00,892")
            st.metric("Hindi Unique Words", "4,226")

    with tab4:
        st.markdown("#### Type-Token Ratio (Vocabulary Richness)")
        show_image("ttr_comparison.png")
        st.markdown("""
        **What is TTR?** → Unique words ÷ Total words. Higher = richer vocabulary.
        
        | Language | TTR Score | Interpretation |
        |----------|-----------|----------------|
        | English  | 0.0202    | Baseline |
        | **Hindi**    | **0.0210**    | ✅ Richer vocabulary – due to complex morphology |
        """)
        st.info("Hindi has higher TTR because of agglutinative morphology – one root word can have many inflected forms.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – TRANSLATION SAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔤 Translation Samples":
    st.markdown('<div class="section-header">Translation Samples</div>', unsafe_allow_html=True)

    data = load_json("translations.json")

    if data is None:
        st.error("translations.json not found. Place it in the same folder as app.py.")
    else:
        src_list  = data.get("source_en", [])
        pred_list = data.get("predicted_hi", [])
        ref_list  = data.get("reference_hi", [])

        total = len(src_list)
        st.markdown(f"**Total translations loaded:** {total:,}")

        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            idx = st.number_input("Go to sentence #", min_value=1, max_value=total, value=1, step=1)
        with col2:
            st.markdown(f"Showing sentence **{idx}** of **{total}**")

        i = idx - 1
        st.markdown(f"""
        <div class="translation-box">
            <div class="trans-label">🇬🇧 Source English</div>
            <div class="trans-text">{src_list[i]}</div>
        </div>
        <div class="translation-box" style="border-left-color:#a6e3a1;">
            <div class="trans-label">🤖 Predicted Hindi</div>
            <div class="trans-text">{pred_list[i]}</div>
        </div>
        <div class="translation-box" style="border-left-color:#f9e2af;">
            <div class="trans-label">✅ Reference Hindi</div>
            <div class="trans-text">{ref_list[i]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🔍 Quick Lookup – Search by Keyword")
        keyword = st.text_input("Search in English source sentences:", placeholder="e.g. road, technology, police")
        if keyword:
            matches = [(i, s) for i, s in enumerate(src_list) if keyword.lower() in s.lower()]
            if matches:
                st.markdown(f"Found **{len(matches)}** matches:")
                for mi, ms in matches[:10]:
                    with st.expander(f"#{mi+1} – {ms[:80]}..."):
                        st.markdown(f"**Predicted:** {pred_list[mi]}")
                        st.markdown(f"**Reference:** {ref_list[mi]}")
            else:
                st.warning("No matches found.")

        st.markdown("---")
        st.markdown("#### Live English → Hindi Translation (Helsinki Model)")
        st.caption("Model: Helsinki-NLP/opus-mt-en-hi (Transformers)")

        user_input_en = st.text_area(
            "Enter English Sentence",
            value="India is making rapid progress in AI research.",
            height=90,
        )
        c_live1, c_live2 = st.columns(2)
        with c_live1:
            beam_size = st.slider("Beam size", 1, 8, 4, 1, key="live_beam_size")
        with c_live2:
            max_tokens = st.slider("Max new tokens", 32, 256, 128, 8, key="live_max_tokens")

        run_live_translate = st.button("Translate with Model", key="live_translate_btn")
        if run_live_translate:
            try:
                with st.spinner("Loading model and translating... first run can take time."):
                    live_hi = translate_en_to_hi(
                        user_input_en.strip(),
                        max_new_tokens=max_tokens,
                        num_beams=beam_size,
                    )
                st.success("Translation generated")
                st.markdown(
                    f"""
                    <div class="translation-box">
                        <div class="trans-label">🇬🇧 Source English</div>
                        <div class="trans-text">{user_input_en}</div>
                    </div>
                    <div class="translation-box" style="border-left-color:#a6e3a1;">
                        <div class="trans-label">🤖 Model Output (Hindi)</div>
                        <div class="trans-text">{live_hi}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(
                    "Live model run failed. Install dependencies and ensure internet is available for first download."
                )
                st.code(
                    ".\\.venv\\Scripts\\python.exe -m pip install transformers sentencepiece torch",
                    language="bash",
                )
                st.caption(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – EVALUATION SCORES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Evaluation Scores":
    st.markdown('<div class="section-header">Evaluation Scores</div>', unsafe_allow_html=True)

    scores = load_json("evaluation_scores.json")
    if scores is None:
        st.error("evaluation_scores.json not found.")
    else:
        bleu   = scores.get("BLEU", 0)
        chrf   = scores.get("chrF", 0)
        meteor = scores.get("METEOR", 0)

        # Score table
        df = pd.DataFrame({
            "Metric": ["BLEU", "chrF", "METEOR"],
            "Score": [bleu, chrf, meteor],
            "Scale": ["0-100", "0-100", "0-1"],
            "Meaning": [
                "Low – but normal for English→Hindi",
                "Moderate – Hindi script character matching",
                "Decent – considers synonyms & word order"
            ],
            "Reliability for Hindi": ["⚠️ Underestimates", "✅ Best metric", "✅ Good"]
        })
        st.dataframe(df, width='stretch', hide_index=True)

        st.markdown("---")
        # Visual score bars
        st.markdown("#### Score Visualization")
        fig, ax = plt.subplots(figsize=(8, 3))
        metrics = ["BLEU\n(÷10 for scale)", "chrF\n(÷10 for scale)", "METEOR\n(×10 for scale)"]
        values  = [bleu/10, chrf/10, meteor*10]
        colors  = ["#89b4fa", "#a6e3a1", "#f9e2af"]
        bars = ax.barh(metrics, values, color=colors, edgecolor="#313244", height=0.4)
        ax.set_xlim(0, 10)
        ax.set_facecolor("#1e1e2e")
        fig.patch.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.spines[:].set_color("#313244")
        for bar, val in zip(bars, [bleu, chrf, meteor]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color="#cdd6f4", fontsize=11, fontweight='bold')
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 📋 Why is BLEU So Low?")
        st.markdown("""
        BLEU works by matching **exact word sequences** (n-grams) between predicted and reference translations.
        
        For English→Hindi this is problematic because:
        
        - Hindi has **complex morphology** – same word has many valid forms
        - **Script difference** – BLEU can't handle Devanagari partial matches  
        - **Word order** – Hindi is SOV, English is SVO – reordering penalizes BLEU heavily
        - **Multiple valid translations** – BLEU only checks against one reference
        
        **chrF is more reliable here** because it works at the character level and handles script-based languages better.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Error Analysis":
    st.markdown('<div class="section-header">Error Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🥧 Quality Distribution", "✅ Good Examples", "❌ Bad Examples"])

    with tab1:
        st.markdown("#### Translation Quality Distribution (English→Hindi)")
        show_image("error_categories.png")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<p class="badge-good">✅ Good (chrF ≥ 50)</p>', unsafe_allow_html=True)
            st.metric("Count", "132", "5.3% of test set")
        with col2:
            st.markdown('<p class="badge-mid">⚠️ Medium (chrF 20-50)</p>', unsafe_allow_html=True)
            st.metric("Count", "1,644", "65.6% of test set")
        with col3:
            st.markdown('<p class="badge-bad">❌ Bad (chrF < 20)</p>', unsafe_allow_html=True)
            st.metric("Count", "731", "29.2% of test set")

        if Path("error_analysis.png").exists():
            st.markdown("---")
            st.markdown("#### Detailed Error Analysis Chart")
            show_image("error_analysis.png")

    with tab2:
        st.markdown("#### ✅ Good Translations (chrF ≥ 50)")
        good_examples = [
            {
                "src": "a black box in your car?",
                "pred": "अपनी कार में एक ब्लैक बॉक्स?",
                "ref": "आपकी कार में ब्लैक बॉक्स?",
                "score": 69.25
            },
            {
                "src": "the technology is there to do it.",
                "pred": "प्रौद्योगिकी यह करने के लिए है.",
                "ref": "ऐसा करने के लिए प्रौद्योगिकी है।",
                "score": 67.07
            },
        ]
        for ex in good_examples:
            with st.expander(f"chrF: {ex['score']} – {ex['src'][:60]}"):
                st.markdown(f"""
                <div class="translation-box">
                    <div class="trans-label">🇬🇧 Source</div>
                    <div class="trans-text">{ex['src']}</div>
                </div>
                <div class="translation-box" style="border-left-color:#a6e3a1;">
                    <div class="trans-label">🤖 Predicted</div>
                    <div class="trans-text">{ex['pred']}</div>
                </div>
                <div class="translation-box" style="border-left-color:#f9e2af;">
                    <div class="trans-label">✅ Reference</div>
                    <div class="trans-text">{ex['ref']}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### ❌ Bad Translations (chrF < 20)")
        st.warning("""
        **Why Do Bad Translations Happen?**
        - Long complex sentences with multiple clauses
        - Domain-specific vocabulary (legal, political terms)
        - Idiomatic expressions that don't translate literally
        - Named entities (organization names, places)
        - The SVO→SOV word order restructuring fails for complex sentences
        """)

        st.markdown("---")
        st.markdown("### 📬 Cross-Lingual Analysis Summary")
        cross_data = {
            "Feature": ["Avg Sentence Length", "Avg Word Length", "TTR (Vocab Richness)", "Word Order"],
            "English": ["3.08 words", "4.53 chars", "0.0202", "SVO"],
            "Hindi": ["3.43 words", "4.16 chars", "0.0210", "SOV"],
            "Implication": [
                "Hindi sentences slightly longer",
                "English words slightly longer",
                "Hindi has richer vocabulary",
                "Key reason MT struggles English→Hindi"
            ]
        }
        st.dataframe(pd.DataFrame(cross_data), width='stretch', hide_index=True)

elif page == "🧠 Topic Modeling":
    st.markdown('<div class="section-header">Topic Modeling (LDA)</div>', unsafe_allow_html=True)
    st.caption("Unsupervised topic discovery on cfilt/iitb-english-hindi")

    c1, c2, c3 = st.columns(3)
    with c1:
        split_expr = st.selectbox("Split", ["train", "validation", "test"], index=0, key="tm_split")
    with c2:
        sample_size = st.slider("Sample Size", 1000, 30000, 8000, 1000, key="tm_sample")
    with c3:
        language = st.radio("Language", ["English", "Hindi"], horizontal=True, key="tm_lang")

    c4, c5 = st.columns(2)
    with c4:
        n_topics = st.slider("Number of Topics", 2, 12, 5, 1, key="tm_topics")
    with c5:
        n_top_words = st.slider("Top Words per Topic", 5, 20, 10, 1, key="tm_words")

    run_tm = st.button("Run Topic Modeling", type="primary")

    if run_tm:
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            st.error("scikit-learn is required for topic modeling.")
            st.code("pip install scikit-learn", language="bash")
            st.stop()

        with st.spinner("Loading dataset and fitting LDA..."):
            texts = load_iitb_texts(split_expr, sample_size, language)

            if len(texts) < 50:
                st.warning("Not enough cleaned text for topic modeling. Increase sample size.")
                st.stop()

            stop_words = list(DEFAULT_EN_STOPWORDS) if language == "English" else list(DEFAULT_HI_STOPWORDS)
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=3,
                stop_words=stop_words,
                max_features=5000,
            )
            dtm = vectorizer.fit_transform(texts)

            if dtm.shape[1] == 0:
                st.warning("No vocabulary left after filtering. Try another language/split.")
                st.stop()

            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method="batch",
                max_iter=20,
            )
            doc_topic = lda.fit_transform(dtm)
            feature_names = vectorizer.get_feature_names_out()

        st.markdown("#### Topics (Top Keywords)")
        topic_rows = []
        for topic_idx, topic_vec in enumerate(lda.components_):
            top_idx = topic_vec.argsort()[-n_top_words:][::-1]
            words = [feature_names[i] for i in top_idx]
            topic_rows.append({"Topic": f"Topic {topic_idx + 1}", "Keywords": ", ".join(words)})
        st.dataframe(pd.DataFrame(topic_rows), width='stretch', hide_index=True)

        st.markdown("#### Dominant Topic Distribution")
        dominant = doc_topic.argmax(axis=1)
        counts = pd.Series(dominant).value_counts().sort_index()
        dist_df = pd.DataFrame(
            {
                "Topic": [f"Topic {i + 1}" for i in counts.index],
                "Documents": counts.values,
                "Share (%)": (counts.values / counts.values.sum() * 100).round(2),
            }
        )
        st.dataframe(dist_df, width='stretch', hide_index=True)

        st.markdown("#### Sample Documents with Dominant Topic")
        sample_show = min(8, len(texts))
        preview_df = pd.DataFrame(
            {
                "Text": [texts[i][:140] + ("..." if len(texts[i]) > 140 else "") for i in range(sample_show)],
                "Dominant Topic": [f"Topic {int(dominant[i]) + 1}" for i in range(sample_show)],
            }
        )
        st.dataframe(preview_df, width='stretch', hide_index=True)

# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<center style='color:#6c7086; font-size:13px;'>SMA IA · English→Hindi Translation Analysis · Helsinki-NLP/opus-mt-en-hi · IIT Bombay Corpus</center>",
    unsafe_allow_html=True
)
