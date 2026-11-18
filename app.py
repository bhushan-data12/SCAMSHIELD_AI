"""
app.py
=============================================================================
ScamShield AI — SMS Spam Detection Using Machine Learning
Streamlit Web Interface & Multi-Algorithm Comparison Platform

Algorithms Included & Compared:
1. Multinomial Naive Bayes (Probabilistic NLP Baseline)
2. Logistic Regression (High-Precision Linear Classifier)
3. Support Vector Machine (Linear SVM with Calibrated Probabilities)
4. Soft Voting Ensemble (Hybrid Consensus Model - Highest Accuracy & F1)
=============================================================================
"""

import os
import re
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ScamShield AI - SMS Spam Detection & Model Comparison",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, sleek, and intuitive dashboard
st.markdown("""
<style>
    /* Global styles */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Result cards */
    .result-box-spam {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0;
    }
    .result-box-ham {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0;
    }
    .result-box-warning {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 15px 0;
    }
    
    /* Badges & metric cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }
    .badge-yellow {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .badge-blue {
        background-color: #e0e7ff;
        color: #4338ca;
        border: 1px solid #a5b4fc;
    }

    /* Message highlight box */
    .highlight-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 14px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "sms_spam_model.joblib"

# ---------------------------------------------------------------------------
# Suspicious Keyword & Phishing URL Scanner
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "free", "win", "won", "winner", "cash", "prize", "reward", "lottery",
    "claim", "urgent", "urgently", "immediately", "act now", "action required",
    "otp", "pin", "password", "cvv", "verify", "suspended", "locked", "blocked",
    "congratulations", "congrats", "bonus", "call now", "gift card", "refund",
    "kyc", "bank account", "debit card", "credit card", "tax refund", "wire transfer",
    "click here", "deposit", "selected", "unclaimed", "exclusive offer"
]

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".click", ".buzz", ".club", ".work", ".link",
    ".loan", ".gq", ".cf", ".tk", ".ml", ".cc", ".icu", ".vip"
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "ow.ly",
    "rebrand.ly", "shorturl.at", "rb.gy", "goo.gl"
]


def scan_suspicious_patterns(text):
    """
    Checks for common scam keywords and web links in the input text.
    Returns detected keywords and detected URLs.
    """
    detected_words = []
    text_lower = text.lower()

    for word in SUSPICIOUS_KEYWORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            detected_words.append(word)

    # URL detection
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)'
    found_urls = re.findall(url_pattern, text)
    # Filter out common punctuation or single abbreviations
    clean_urls = [u.rstrip('.,!?:;') for u in found_urls if '.' in u and not u.endswith('.')]

    return detected_words, clean_urls


def inspect_url(url):
    """
    Analyzes an individual URL for phishing and deception indicators.
    """
    url_lower = url.lower()
    flags = []
    risk_score = 0

    if url_lower.startswith("http://"):
        flags.append("Insecure Protocol (HTTP instead of HTTPS)")
        risk_score += 25

    if any(shortener in url_lower for shortener in URL_SHORTENERS):
        flags.append("URL Shortener used (hides the real destination)")
        risk_score += 35

    if any(url_lower.endswith(tld) or (tld + "/") in url_lower for tld in SUSPICIOUS_TLDS):
        flags.append("High-risk / free top-level domain frequently used in phishing")
        risk_score += 30

    if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
        flags.append("Uses raw IP address instead of domain name")
        risk_score += 40

    if "@" in url:
        flags.append("Contains '@' symbol (often used to obscure malicious redirect)")
        risk_score += 35

    # Check for fake banking keywords in domain
    banking_terms = ["sbi", "hdfc", "icici", "paytm", "kyc", "bank", "login", "secure", "update", "verify"]
    found_terms = [t for t in banking_terms if t in url_lower]
    if len(found_terms) >= 2 and not any(legit in url_lower for legit in ["sbi.co.in", "hdfcbank.com", "icicibank.com"]):
        flags.append(f"Suspicious banking keywords combined in URL: `{', '.join(found_terms)}`")
        risk_score += 40

    return {
        'url': url,
        'flags': flags,
        'risk_score': min(risk_score, 100)
    }


def highlight_keywords_in_text(text, keywords):
    """
    Highlights suspicious scam words in HTML using colored badges.
    """
    if not keywords:
        return text

    highlighted = text
    # Sort by length descending to prevent partial replacements
    sorted_kw = sorted(keywords, key=len, reverse=True)
    for kw in sorted_kw:
        pattern = re.compile(rf'(\b{re.escape(kw)}\b)', re.IGNORECASE)
        highlighted = pattern.sub(r'<span class="badge-pill badge-red">\1</span>', highlighted)
    return highlighted


# ---------------------------------------------------------------------------
# Load Model Function (Backward compatible & multi-model support)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """
    Loads all trained model pipelines and attached evaluation metrics.
    Supports both single pipeline models and multi-model bundles.
    """
    if not os.path.exists(MODEL_FILE):
        return None, {}, None

    data = joblib.load(MODEL_FILE)

    if isinstance(data, dict) and 'models' in data:
        models = data['models']
        default_model = data.get('pipeline') or next(iter(models.values()))
        metrics = data.get('metrics', {})
        return default_model, models, metrics
    elif hasattr(data, 'predict'):
        # Legacy single pipeline
        models = {'Multinomial Naive Bayes': data}
        metrics = getattr(data, 'metrics_', {})
        return data, models, metrics
    else:
        return None, {}, None


# ---------------------------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
            <div class="main-title">🛡️ ScamShield AI</div>
            <div class="sub-title">Intelligent SMS Spam & Scam Detection with Multi-Algorithm Machine Learning Benchmarking</div>
        </div>
        <div>
            <span class="badge-pill badge-blue">Naive Bayes</span>
            <span class="badge-pill badge-blue">Logistic Regression</span>
            <span class="badge-pill badge-blue">Linear SVM</span>
            <span class="badge-pill badge-green">Voting Ensemble</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Navigation & Model Selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📌 Navigation Menu")
    app_mode = st.radio(
        "Select Module:",
        [
            "🔍 Single SMS Predictor",
            "⚖️ Algorithm Comparison & Benchmark",
            "📁 Batch CSV / TXT Classifier",
            "🔗 Phishing URL Inspector",
            "📖 About & Viva Defense Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Configuration")

    # Load model
    default_model, all_models, metrics_bundle = load_trained_model()

    if all_models:
        model_names = list(all_models.keys())
        # Default to Soft Voting Ensemble or first
        default_idx = model_names.index("Soft Voting Ensemble") if "Soft Voting Ensemble" in model_names else 0
        selected_model_name = st.selectbox(
            "Active Classification Model:",
            model_names,
            index=default_idx,
            help="Choose which algorithm powers the predictor. You can also compare all of them side-by-side!"
        )
        active_model = all_models[selected_model_name]
    else:
        selected_model_name = "None"
        active_model = None

    st.markdown("---")
    st.markdown("### 🎯 Standard Decision Threshold")
    STANDARD_THRESHOLD = 50.0  # Standard statistical threshold for binary classification
    spam_threshold = STANDARD_THRESHOLD

    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 14px; margin-bottom: 6px;">
        <div style="font-weight: 700; color: #1e293b; font-size: 0.95rem;">Standard Cutoff: 50.0%</div>
        <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">
            Based on the standard binary classification boundary: messages with Spam Probability ≥ <b>50%</b> are classified as <b>Scam</b>; below 50% are <b>Normal</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset & Tech Stack")
    st.markdown("""
    - **Dataset:** UCI SMS Spam Collection (5,574 messages)
    - **Feature Extraction:** TF-IDF (1-2 N-grams, 5,000 features)
    - **Algorithms:** MNB, LogReg, Linear SVM, Ensemble
    - **Frontend:** Streamlit 1.56
    """)

# Handle case where model is missing or needs retraining
if active_model is None:
    st.warning("⚠️ **Model bundle (`sms_spam_model.joblib`) not found or needs retraining!**")
    st.info("Train the machine learning models now (takes ~5 seconds on the UCI dataset).")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        if st.button("🚀 Train All Models Now", use_container_width=True, type="primary"):
            with st.spinner("Training Naive Bayes, Logistic Regression, Linear SVM, and Ensemble..."):
                from train_model import train_and_evaluate
                train_and_evaluate()
                st.cache_resource.clear()
                st.success("✅ Models trained and saved successfully! Reloading...")
                st.rerun()
    with col_t2:
        st.write("Alternatively, run this in your terminal:")
        st.code("python train_model.py")
    st.stop()


# ---------------------------------------------------------------------------
# MODULE 1: SINGLE SMS PREDICTOR
# ---------------------------------------------------------------------------
if app_mode == "🔍 Single SMS Predictor":
    st.subheader(f"🔍 Single SMS Predictor — Powered by `{selected_model_name}`")
    st.caption("Enter an SMS message to detect phishing scams, OTP harvesting, fake lottery winnings, and fraudulent links.")

    # Quick demonstration sample buttons
    st.write("**Quick Test Samples:**")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])

    if "sms_input" not in st.session_state:
        st.session_state["sms_input"] = ""

    with col_s1:
        if st.button("📥 Normal Ham", use_container_width=True):
            st.session_state["sms_input"] = "Hey, are you free for the project discussion in the college library at 3 PM today?"
    with col_s2:
        if st.button("🚨 Urgent Lottery", use_container_width=True):
            st.session_state["sms_input"] = "URGENT! You have won a £1,000 cash prize or a free holiday. Call 09061701461 now to claim your reward immediately!"
    with col_s3:
        if st.button("🏦 Banking Phishing", use_container_width=True):
            st.session_state["sms_input"] = "Dear customer, your bank account has been suspended due to unverified KYC. Click http://bit.ly/sbi-verify-kyc now to update your details."
    with col_s4:
        if st.button("🎁 Gift Card Scam", use_container_width=True):
            st.session_state["sms_input"] = "Congratulations! You have been selected for a free $500 Amazon gift card. Enter your PIN and password at http://claim-free-gift.xyz"
    with col_s5:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["sms_input"] = ""

    # Message text area
    user_sms = st.text_area(
        "Type or paste SMS message below:",
        value=st.session_state["sms_input"],
        height=120,
        placeholder="Example: Congratulations! Your account has been credited with $1,000. Click http://... to withdraw."
    )

    col_btn1, col_btn2 = st.columns([1.5, 4.5])
    with col_btn1:
        predict_clicked = st.button("🔎 Analyze Message", type="primary", use_container_width=True)
    with col_btn2:
        compare_all_toggle = st.checkbox("⚡ Show live predictions from all 4 models simultaneously", value=True)

    if predict_clicked:
        if not user_sms.strip():
            st.warning("⚠️ Please enter or select an SMS message to test.")
        else:
            # 1. Prediction with active model
            probabilities = active_model.predict_proba([user_sms])[0]
            prob_ham = probabilities[0] * 100
            prob_spam = probabilities[1] * 100

            # Apply user-configured threshold
            is_scam = prob_spam >= spam_threshold
            prediction = 1 if is_scam else 0

            # 2. Pattern and URL scanning
            detected_words, detected_urls = scan_suspicious_patterns(user_sms)

            # 3. Determine Overall Risk Category based on standard 50% threshold & patterns
            if is_scam:
                if prob_spam >= 75 or (len(detected_words) >= 2 or detected_urls):
                    risk_level = "HIGH-RISK SCAM"
                    risk_class = "result-box-spam"
                    risk_color = "#b91c1c"
                    risk_icon = "🚨"
                else:
                    risk_level = "SPAM / SCAM MESSAGE"
                    risk_class = "result-box-spam"
                    risk_color = "#dc2626"
                    risk_icon = "🚨"
            else:
                if prob_spam >= 35 and (detected_words or detected_urls):
                    risk_level = "SUSPICIOUS (CHECK SENDER)"
                    risk_class = "result-box-warning"
                    risk_color = "#b45309"
                    risk_icon = "⚠️"
                else:
                    risk_level = "LEGITIMATE (NORMAL HAM)"
                    risk_class = "result-box-ham"
                    risk_color = "#15803d"
                    risk_icon = "✅"

            # 4. Display Primary Result Card
            threshold_diff = prob_spam - 50.0
            diff_label = f"+{threshold_diff:.1f}% above standard cutoff" if threshold_diff >= 0 else f"{abs(threshold_diff):.1f}% below standard cutoff"

            st.markdown(f"""
            <div class="{risk_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <h3 style="color: {risk_color}; margin: 0;">
                        {risk_icon} {risk_level}
                    </h3>
                    <span style="font-weight: 700; font-size: 1.05rem; color: {risk_color};">
                        Spam Probability: {prob_spam:.1f}% | Cutoff: 50.0% ({diff_label})
                    </span>
                </div>
                <p style="margin: 8px 0 0 0; color: #334155; font-size: 0.95rem;">
                    Evaluated by <b>{selected_model_name}</b> | Decision: <b>{'🚨 Flagged as Scam' if is_scam else '✅ Marked as Legitimate'}</b> (Normal: <b>{prob_ham:.1f}%</b> | Spam: <b>{prob_spam:.1f}%</b>)
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence Visual Progress Bar with Standard Threshold Reference
            col_bar1, col_bar2 = st.columns([3, 1])
            with col_bar1:
                st.write("**Spam Probability Gauge** (Standard 50% Threshold):")
                st.progress(min(int(prob_spam), 100))
                if threshold_diff >= 0:
                    st.caption(f"🚨 **Scam flagged:** Probability **{prob_spam:.1f}%** is **{threshold_diff:.1f}% higher** than the standard 50.0% cutoff.")
                else:
                    st.caption(f"✅ **Clear:** Probability **{prob_spam:.1f}%** is **{abs(threshold_diff):.1f}% safely below** the standard 50.0% cutoff.")
            with col_bar2:
                if is_scam:
                    st.markdown(f"<span class='badge-pill badge-red'>🚨 SCAM: {prob_spam:.1f}%</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-pill badge-green'>✅ NORMAL: {prob_ham:.1f}%</span>", unsafe_allow_html=True)

            # 5. Live Multi-Model Comparison Table
            if compare_all_toggle and len(all_models) > 1:
                st.markdown("#### ⚡ Live Multi-Algorithm Consensus (Standard 50% Cutoff)")
                comparison_rows = []
                for m_name, m_pipeline in all_models.items():
                    probs = m_pipeline.predict_proba([user_sms])[0]
                    m_spam_prob = probs[1] * 100
                    m_is_scam = m_spam_prob >= 50.0
                    verdict = "🚨 Scam (≥50%)" if m_is_scam else "✅ Normal (<50%)"
                    confidence = f"{probs[1 if m_is_scam else 0] * 100:.1f}%"
                    comparison_rows.append({
                        "Algorithm": m_name,
                        "Verdict": verdict,
                        "Spam Probability": f"{m_spam_prob:.1f}%",
                        "Confidence": confidence,
                        "Selected": "👉 Active" if m_name == selected_model_name else ""
                    })

                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # 6. Explainable AI & Keyword Highlighting
            col_exp1, col_exp2 = st.columns([1.2, 1])
            with col_exp1:
                st.markdown("#### 🔍 Explainable Pattern Highlighter")
                highlighted_html = highlight_keywords_in_text(user_sms, detected_words)
                st.markdown(f"""
                <div class="highlight-box">
                    {highlighted_html}
                </div>
                """, unsafe_allow_html=True)

                if detected_words:
                    st.write("**Detected Trigger Words:**")
                    badges_html = "".join([f'<span class="badge-pill badge-red">{w}</span>' for w in detected_words])
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.info("No explicit scam trigger keywords found in vocabulary list.")

            with col_exp2:
                st.markdown("#### 🛡️ Actionable Safety Advice")
                if is_scam:
                    st.error("""
                    - **Do not click links** or call unverified phone numbers in this SMS.
                    - **Never share OTPs**, bank passwords, PINs, or card CVVs.
                    - Official banks never ask for credentials via standard SMS.
                    - Report scam SMS to your telecom provider or national cyber fraud helpline (e.g. 1930).
                    """)
                else:
                    st.success("""
                    - This message appears to be standard conversational text.
                    - Always maintain caution when asked to transfer money or install unexpected applications.
                    """)

            # 7. URL Inspection if present
            if detected_urls:
                st.markdown("---")
                st.markdown("#### 🔗 Extracted Links & Phishing Inspection")
                for u in detected_urls:
                    analysis = inspect_url(u)
                    u_risk = analysis['risk_score']
                    u_flags = analysis['flags']

                    col_u1, col_u2 = st.columns([2, 1])
                    with col_u1:
                        st.markdown(f"**URL:** `{u}`")
                        if u_flags:
                            for f in u_flags:
                                st.write(f"- ⚠️ {f}")
                        else:
                            st.write("- ✅ No prominent phishing indicators detected in URL syntax.")
                    with col_u2:
                        st.metric("URL Risk Score", f"{u_risk}/100")


# ---------------------------------------------------------------------------
# MODULE 2: ALGORITHM COMPARISON & BENCHMARK
# ---------------------------------------------------------------------------
elif app_mode == "⚖️ Algorithm Comparison & Benchmark":
    st.subheader("⚖️ Machine Learning Algorithm Comparison & Viva Defense Benchmark")
    st.write("Comprehensive evaluation of **4 models** on the 20% stratified holdout test set (1,115 SMS messages).")

    metrics_dict = metrics_bundle.get('all_metrics', {})
    table_data = metrics_bundle.get('comparison_table', [])

    if table_data:
        comp_df = pd.DataFrame(table_data)

        # Highlight highest values in table
        st.markdown("### 📊 Performance Summary Table")
        st.dataframe(
            comp_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Visual Comparison Bar Charts
        st.markdown("### 📈 Visual Metric Comparison Across Algorithms")
        chart_col1, chart_col2 = st.columns(2)

        models_list = comp_df['Algorithm'].tolist()
        short_names = [
            m.replace("Support Vector Machine (Linear SVM)", "Linear SVM")
             .replace("Soft Voting Ensemble", "Voting Ensemble")
             .replace("Multinomial Naive Bayes", "Naive Bayes")
            for m in models_list
        ]

        with chart_col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            x = np.arange(len(short_names))
            width = 0.35

            ax1.bar(x - width/2, comp_df['Accuracy (%)'], width, label='Accuracy', color='#3b82f6')
            ax1.bar(x + width/2, comp_df['F1-Score (%)'], width, label='F1-Score', color='#10b981')
            ax1.set_ylabel('Percentage (%)')
            ax1.set_title('Accuracy vs F1-Score Comparison', fontsize=11, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
            ax1.set_ylim(85, 101)
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            ax1.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig1)

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(x - width/2, comp_df['Precision (%)'], width, label='Precision (Spam)', color='#8b5cf6')
            ax2.bar(x + width/2, comp_df['Recall (%)'], width, label='Recall (Spam)', color='#f59e0b')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Precision vs Recall Trade-off', fontsize=11, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
            ax2.set_ylim(75, 102)
            ax2.grid(axis='y', linestyle='--', alpha=0.5)
            ax2.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("---")

        # Side-by-Side Confusion Matrices
        st.markdown("### 🔲 Side-by-Side Confusion Matrices")
        st.write("Evaluating False Positives (Normal messages falsely blocked) vs False Negatives (Spam messages missed).")

        cm_cols = st.columns(len(table_data))
        for idx, row in enumerate(table_data):
            with cm_cols[idx]:
                name = row['Algorithm']
                tn = row['True Negatives']
                fp = row['False Positives']
                fn = row['False Negatives']
                tp = row['True Positives']

                short_title = name.replace("Support Vector Machine (Linear SVM)", "Linear SVM")\
                                  .replace("Soft Voting Ensemble", "Voting Ensemble")\
                                  .replace("Multinomial Naive Bayes", "Naive Bayes")

                fig_cm, ax_cm = plt.subplots(figsize=(3, 2.5))
                cax = ax_cm.matshow([[tn, fp], [fn, tp]], cmap=plt.cm.Blues, alpha=0.75)
                ax_cm.text(0, 0, str(tn), va='center', ha='center', fontsize=11, fontweight='bold')
                ax_cm.text(1, 0, str(fp), va='center', ha='center', fontsize=11, fontweight='bold', color='red' if fp > 0 else 'black')
                ax_cm.text(0, 1, str(fn), va='center', ha='center', fontsize=11, fontweight='bold', color='orange')
                ax_cm.text(1, 1, str(tp), va='center', ha='center', fontsize=11, fontweight='bold', color='green')
                ax_cm.set_xticks([0, 1])
                ax_cm.set_yticks([0, 1])
                ax_cm.set_xticklabels(['Pred H', 'Pred S'], fontsize=8)
                ax_cm.set_yticklabels(['Act H', 'Act S'], fontsize=8)
                ax_cm.set_title(short_title, fontsize=9, pad=8, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_cm)

                st.caption(f"**FP (False Alarm):** {fp}\n\n**FN (Missed Spam):** {fn}")

        st.markdown("---")

        # Deep Academic Analysis: Which model does better and why?
        st.markdown("### 🏆 Which Algorithm Does Better & Why? (Viva Defense Analysis)")
        st.markdown("""
        In real-world cybersecurity, no single metric tells the whole story. Here is the rigorous evaluation of which algorithm performs better and the engineering trade-offs between them:

        #### 1. 🥇 The Best Overall: **Soft Voting Ensemble (98.74% Accuracy, 95.14% F1-Score)**
        - **Why it wins:** Combines probabilistic reasoning (Naive Bayes), linear boundary optimization (Logistic Regression), and maximum-margin hyperplane separation (Linear SVM).
        - **Consensus Benefit:** Outlier errors from one algorithm are smoothed out by probability consensus, producing the highest overall accuracy and lowest combined error count.

        #### 2. 🛡️ The Best for User Trust: **Logistic Regression (100.00% Precision)**
        - **Why it matters:** Zero False Positives! Out of 966 legitimate messages, it correctly approved all 966 without a single false alarm.
        - **Use-Case:** Critical when a user **cannot afford to miss a legitimate bank OTP or doctor appointment message**.
        - **Trade-off:** Lower recall (83.22%), meaning slightly more spam messages sneak through.

        #### 3. 🎯 The Best for Maximum Spam Catch: **Linear SVM (92.62% Recall)**
        - **Why it matters:** Highest recall among individual classifiers. It successfully intercepted 138 out of 149 real spam messages (only 11 missed).
        - **Why it works on text:** Text classification involves high-dimensional sparse TF-IDF vectors (5,000 features). Support Vector Machines excel in high-dimensional spaces by finding the maximum-margin hyperplane.

        #### 4. ⚡ The Best Lightweight Baseline: **Multinomial Naive Bayes (98.57% Accuracy)**
        - **Why it matters:** Exceptional speed and low computational overhead. Trains in milliseconds and provides 98.54% precision.
        - **Academic Note:** Assumes conditional feature independence given the class, which simplifies probability calculations via Bayes' Theorem.
        """)
    else:
        st.info("Metrics data not found. Please train models using `python train_model.py`.")


# ---------------------------------------------------------------------------
# MODULE 3: BATCH CSV / TXT CLASSIFIER
# ---------------------------------------------------------------------------
elif app_mode == "📁 Batch CSV / TXT Classifier":
    st.subheader("📁 Batch File Classifier (CSV / TXT)")
    st.write("Upload a file containing multiple SMS messages to classify them in bulk and download the labeled results.")

    uploaded_file = st.file_uploader(
        "Upload CSV or TXT file:",
        type=["csv", "txt"],
        help="Upload a CSV with text messages or a plain TXT file with one message per line."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
                text_columns = df_upload.select_dtypes(include=['object']).columns.tolist()
                if not text_columns:
                    st.error("No text columns found in the uploaded CSV.")
                    st.stop()
                selected_col = st.selectbox("Select column containing SMS messages:", text_columns)
                messages_to_test = df_upload[selected_col].dropna().astype(str).tolist()
            else:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
                lines = [line.strip() for line in stringio.readlines() if line.strip()]
                messages_to_test = lines
                df_upload = pd.DataFrame({"message": messages_to_test})
                selected_col = "message"

            st.write(f"Loaded **{len(messages_to_test)}** messages for classification.")

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner(f"Classifying {len(messages_to_test)} messages with {selected_model_name}..."):
                    probs = active_model.predict_proba(messages_to_test)
                    spam_probs = [p[1] * 100 for p in probs]
                    preds = [1 if sp >= 50.0 else 0 for sp in spam_probs]
                    labels = ["Spam" if p == 1 else "Normal (Ham)" for p in preds]

                    df_results = df_upload.copy()
                    df_results['Prediction'] = labels
                    df_results['Spam_Probability_%'] = [round(sp, 2) for sp in spam_probs]

                    spam_count = int(sum(p == 1 for p in preds))
                    ham_count = int(sum(p == 0 for p in preds))
                    total_cnt = len(messages_to_test)

                    st.info("🎯 **Standard Decision Threshold:** Evaluated at standard **50.0%** cutoff. Messages with Spam Probability ≥ 50% are marked as Spam.")

                    # Summary cards
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Total Messages", total_cnt)
                    with col_b2:
                        st.metric("🚨 Detected Spam (≥50%)", f"{spam_count} ({spam_count/total_cnt*100:.1f}%)")
                    with col_b3:
                        st.metric("✅ Legitimate (<50%)", f"{ham_count} ({ham_count/total_cnt*100:.1f}%)")

                    # Preview table
                    st.markdown("### 📋 Classification Results Preview")
                    st.dataframe(df_results.head(100), use_container_width=True)

                    # CSV Download
                    csv_buffer = io.BytesIO()
                    df_results.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Full Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"scamshield_results_{selected_model_name.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ---------------------------------------------------------------------------
# MODULE 4: PHISHING URL INSPECTOR
# ---------------------------------------------------------------------------
elif app_mode == "🔗 Phishing URL Inspector":
    st.subheader("🔗 Phishing URL & Deceptive Domain Inspector")
    st.write("Inspect hyperlinks extracted from SMS and emails for common phishing techniques, URL shorteners, and fraudulent domains.")

    sample_urls = [
        "http://bit.ly/sbi-verify-kyc-urgent",
        "http://192.168.1.105/bank-login/update",
        "https://secure-account-update.xyz/login.php",
        "https://www.google.com"
    ]

    selected_sample = st.selectbox("Or choose a sample link to test:", [""] + sample_urls)

    input_url = st.text_input(
        "Enter URL to inspect:",
        value=selected_sample if selected_sample else "",
        placeholder="http://example-verify-login.xyz/update"
    )

    if st.button("🔎 Analyze Link", type="primary"):
        if not input_url.strip():
            st.warning("Please enter a URL to inspect.")
        else:
            result = inspect_url(input_url.strip())
            risk_score = result['risk_score']
            flags = result['flags']

            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                if risk_score >= 50:
                    st.error(f"🚨 **High Risk Phishing Indicator ({risk_score}/100)**")
                elif risk_score >= 25:
                    st.warning(f"⚠️ **Suspicious Link Attributes ({risk_score}/100)**")
                else:
                    st.success(f"✅ **Low Risk ({risk_score}/100)** — No immediate structural red flags found.")

                if flags:
                    st.write("**Identified Risk Factors:**")
                    for f in flags:
                        st.markdown(f"- 🔴 {f}")
                else:
                    st.write("- Uses standard domain formatting with no common phishing tricks.")

            with col_res2:
                st.metric("Deception Risk Score", f"{risk_score}/100")
                st.progress(risk_score)


# ---------------------------------------------------------------------------
# MODULE 5: ABOUT & VIVA DEFENSE GUIDE
# ---------------------------------------------------------------------------
elif app_mode == "📖 About & Viva Defense Guide":
    st.subheader("📖 Academic Viva Defense & Technical Architecture Guide")

    st.markdown("""
    ### 1. Project Overview & Problem Statement
    SMS-based phishing (**Smishing**) is a prevalent form of cyber attack where malicious actors impersonate financial institutions, government bodies, or lottery organizations to steal credentials, banking OTPs, and personal identity data. 

    ScamShield AI solves this problem by using supervised machine learning and natural language processing (NLP) to classify text messages, identify suspicious patterns, and provide actionable security intelligence.

    ---

    ### 2. Theoretical Background of Implemented Algorithms

    #### A. Feature Extraction: TF-IDF (Term Frequency - Inverse Document Frequency)
    Raw text cannot be processed directly by machine learning algorithms. TF-IDF transforms textual messages into numerical feature vectors by balancing term frequency against inverse document frequency:
    $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
    - **Why it is superior to Bag of Words (CountVectorizer):** It penalizes common stop-words and gives higher weights to rare, domain-specific words (e.g. *lottery, prize, urgent*).

    #### B. Multinomial Naive Bayes (MNB)
    - **Foundation:** Bayes' Theorem with the "naive" assumption of conditional feature independence given class $y$:
      $$P(y \\mid x_1, \\dots, x_n) \\propto P(y) \\prod_{i=1}^n P(x_i \\mid y)$$
    - **Laplace Smoothing ($\\alpha = 0.1$):** Prevents zero-probability traps for words unseen in the training vocabulary.
    - **Advantage:** Lightning-fast training and inference, negligible memory footprint.

    #### C. Logistic Regression (LR)
    - **Foundation:** Models the log-odds of a message being spam using the standard sigmoid function:
      $$\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad z = w^T x + b$$
    - **Why it performs well:** In text classification, linear separation often works effectively because TF-IDF vectors are high-dimensional. L2 regularization prevents overfitting.
    - **Key Result:** Achieved **100% Precision** on the test set, producing zero false alarms.

    #### D. Support Vector Machine (Linear SVM)
    - **Foundation:** Finds the optimal hyperplane that maximizes the geometric margin between Normal and Spam classes:
      $$\\min_{w, b} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^N \\xi_i$$
    - **Probability Calibration:** Since standard SVM computes signed distance to hyperplane rather than probabilities, we apply Platt Scaling via `CalibratedClassifierCV` (sigmoid cross-validation).
    - **Key Result:** Achieved the **highest Spam Recall (92.62%)**, intercepting the most spam messages.

    #### E. Soft Voting Ensemble
    - Aggregates predicted class probabilities from all three diverse classifiers:
      $$P_{\\text{ensemble}}(y) = \\frac{1}{3} \\sum_{m=1}^3 P_m(y)$$
    - **Key Result:** Achieved the highest overall accuracy (**98.74%**) and F1-Score (**95.14%**).

    ---

    ### 3. Top 8 Frequently Asked Viva Questions & Answers

    **Q1: Why did you choose Naive Bayes as the baseline algorithm?**  
    *Answer:* Multinomial Naive Bayes is the classic gold-standard baseline for text categorization because text word counts follow a multinomial distribution. It computes fast, works exceptionally well with sparse TF-IDF matrices, and has minimal memory overhead.

    **Q2: Which algorithm performed best, and why?**  
    *Answer:* The **Soft Voting Ensemble** performed best overall with **98.74% accuracy** and an **F1-score of 95.14%** because it combines the probabilistic nature of Naive Bayes, the high precision of Logistic Regression, and the maximum margin classification of Linear SVM.

    **Q3: Why is Precision considered more critical than Recall in spam detection?**  
    *Answer:* A False Positive means a legitimate message (such as a critical banking OTP or job interview email) is incorrectly marked as spam and missed by the user. False Negatives (spam slipping into inbox) are annoying, but False Positives can cause catastrophic communication loss. Logistic Regression achieved **100% precision (0 false positives)**.

    **Q4: How did you handle data imbalance?**  
    *Answer:* The UCI dataset has 86.6% Ham and 13.4% Spam. We used **Stratified Train-Test Splitting** (`stratify=y`), ensuring the exact same class ratio exists in both training (80%) and testing (20%) sets. We also evaluated F1-score and Precision-Recall rather than relying solely on accuracy.

    **Q5: What is n-gram and why did you use `(1, 2)`?**  
    *Answer:* N-grams capture sequences of $N$ words. Unigrams `(1)` capture isolated words like *"prize"*, while bigrams `(2)` capture contextual combinations like *"cash prize"* and *"call now"*, which carry significantly stronger spam indicators than isolated words.

    **Q6: What is Laplace smoothing in Naive Bayes?**  
    *Answer:* If a word appears in a test SMS that never appeared in the spam training set, its conditional probability would be zero. Because Naive Bayes multiplies probabilities, a single zero would zero out the entire spam score. Laplace smoothing (adding $\\alpha = 0.1$) prevents this zero-probability trap.

    **Q7: How does Linear SVM obtain probability scores?**  
    *Answer:* A standard Linear SVM outputs signed geometric distances from the decision hyperplane, not calibrated probabilities. We wrapped the model in Scikit-Learn's `CalibratedClassifierCV`, which applies Platt's logistic sigmoid transformation across cross-validation folds.

    **Q8: Can this model detect scams in other languages or audio?**  
    *Answer:* The current model is trained on English text. For multi-language or regional dialects, we can fine-tune multilingual models (e.g., mBERT or XLM-RoBERTa). For voice/audio, Speech-to-Text (STT) models like Whisper can transcribe audio before passing it to ScamShield AI.

    **Q9: What is the Decision Threshold and why is it adjustable?**  
    *Answer:* In binary classification, the default decision threshold is 0.50 (50%). Any message with $P(\\text{Spam}) \\ge 0.50$ is flagged as scam. Adjusting this threshold allows tuning between Precision and Recall:
    - **Higher threshold (e.g., 70%):** Maximizes **Precision** (eliminates false alarms so important messages/OTPs are never blocked).
    - **Lower threshold (e.g., 35%):** Maximizes **Recall** (catches all potential scams for high-security environments).
    """)
