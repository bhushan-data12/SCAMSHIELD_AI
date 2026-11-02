"""
app.py
=============================================================================
ScamShield AI — SMS Spam Detection Using Machine Learning
Streamlit Web Interface & Prediction Application

This application allows a user to:
1. Enter an SMS text message or choose a quick demonstration sample.
2. Click "Predict" to classify the message as 'Spam' or 'Normal (Ham)'
   using a trained Multinomial Naive Bayes model with TF-IDF features.
3. View detected suspicious keywords/patterns and a safety recommendation.
4. Review model performance metrics (Accuracy, Precision, Confusion Matrix).
5. Read an 'About' section explaining the project workflow for viva.
=============================================================================
"""

import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ScamShield AI - SMS Spam Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling for a clean, professional, beginner-friendly interface
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .result-box-spam {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 16px;
        margin: 15px 0;
    }
    .result-box-ham {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 16px;
        margin: 15px 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "sms_spam_model.joblib"

# ---------------------------------------------------------------------------
# Suspicious Keyword Scanner (Simple & Explainable)
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "free", "win", "won", "winner", "cash", "prize", "reward", "lottery",
    "claim", "urgent", "urgently", "immediately", "act now", "action required",
    "otp", "pin", "password", "cvv", "verify", "suspended", "locked", "blocked",
    "congratulations", "congrats", "bonus", "call now", "gift card", "refund"
]

def scan_suspicious_patterns(text):
    """
    Checks for common scam keywords and web links in the input text.
    Returns a list of detected suspicious words/patterns.
    """
    detected = []
    text_lower = text.lower()

    # Check for keywords
    for word in SUSPICIOUS_KEYWORDS:
        # Match whole word boundaries
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            detected.append(word)

    # Check for web URLs or link shorteners
    if re.search(r'https?://|www\.|\.ly/|\.com/|\.top|\.xyz', text_lower):
        detected.append("link/URL (http/https/link shortener)")

    return detected

# ---------------------------------------------------------------------------
# Load Model Function
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """
    Loads the trained model pipeline and attached evaluation metrics.
    """
    if not os.path.exists(MODEL_FILE):
        return None, None

    model_data = joblib.load(MODEL_FILE)
    if hasattr(model_data, 'predict'):
        pipeline = model_data
        metrics = getattr(model_data, 'metrics_', None)
    elif isinstance(model_data, dict):
        pipeline = model_data.get('pipeline') or model_data.get('model')
        metrics = model_data.get('metrics')
    else:
        pipeline = None
        metrics = None

    return pipeline, metrics

# ---------------------------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ ScamShield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">SMS Spam Detection Using Machine Learning (Multinomial Naive Bayes + TF-IDF)</div>', unsafe_allow_html=True)

# Sidebar with quick project info and navigation
with st.sidebar:
    st.header("📌 Project Menu")
    app_mode = st.radio(
        "Go to:",
        ["🔍 SMS Predictor", "📊 Model Dashboard", "📖 About Project"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Algorithm:** Multinomial Naive Bayes")
    st.markdown("**Features:** TF-IDF Vectorizer")
    st.markdown("**Dataset:** UCI SMS Spam Collection")
    st.markdown("**Status:** Ready for Viva Demo")

# Load model
model, metrics = load_trained_model()

# Handle case where model is missing
if model is None:
    st.warning("⚠️ **Model file (`sms_spam_model.joblib`) not found!**")
    st.info("The machine learning model needs to be trained first.")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        if st.button("🚀 Train Model Now", use_container_width=True):
            with st.spinner("Training model on UCI dataset... please wait a few seconds."):
                from train_model import train_and_evaluate
                model = train_and_evaluate()
                st.cache_resource.clear()
                st.success("✅ Model trained and saved successfully! Reloading...")
                st.rerun()
    with col_t2:
        st.write("Alternatively, run this command in your PowerShell terminal:")
        st.code("python train_model.py")
    st.stop()


# ---------------------------------------------------------------------------
# SECTION 1: SMS PREDICTOR
# ---------------------------------------------------------------------------
if app_mode == "🔍 SMS Predictor":
    st.subheader("✉️ Enter SMS Message to Test")

    # Sample test buttons to assist presentation during college viva
    st.write("**Quick Test Samples:**")
    col_s1, col_s2, col_s3 = st.columns([1.5, 1.5, 3])
    
    # Session state for text input
    if "sms_text" not in st.session_state:
        st.session_state["sms_text"] = ""

    with col_s1:
        if st.button("📥 Load Normal Sample", use_container_width=True):
            st.session_state["sms_text"] = "Hey, are you free for the project discussion in the college library at 3 PM today?"
    with col_s2:
        if st.button("🚨 Load Spam Sample", use_container_width=True):
            st.session_state["sms_text"] = "URGENT! You have won a £1,000 cash prize or a free holiday. Call 09061701461 now to claim your reward immediately!"
    with col_s3:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state["sms_text"] = ""

    # Message text area
    user_sms = st.text_area(
        "Type or paste SMS message below:",
        value=st.session_state["sms_text"],
        height=110,
        placeholder="Example: Congratulations! You have won a free gift card. Click link to claim..."
    )

    # Predict button
    if st.button("🔎 Predict SMS", type="primary", use_container_width=False):
        if not user_sms.strip():
            st.warning("Please enter an SMS message to test.")
        else:
            # 1. Machine Learning Prediction using TF-IDF + Multinomial Naive Bayes
            prediction = model.predict([user_sms])[0]
            # Get probability scores (Normal vs Spam)
            probabilities = model.predict_proba([user_sms])[0]
            prob_ham = probabilities[0] * 100
            prob_spam = probabilities[1] * 100

            # 2. Suspicious pattern check
            detected_patterns = scan_suspicious_patterns(user_sms)

            # 3. Display Result
            if prediction == 1:
                # SPAM RESULT
                st.markdown(f"""
                <div class="result-box-spam">
                    <h3 style="color: #b91c1c; margin: 0;">🚨 Prediction: SPAM MESSAGE</h3>
                    <p style="margin: 6px 0 0 0; color: #4b5563;">
                        Confidence Score: <b>{prob_spam:.1f}% Spam</b> (Normal: {prob_ham:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Suspicious words warning
                if detected_patterns:
                    st.warning(f"⚠️ **Suspicious patterns detected:** `{', '.join(detected_patterns)}`")
                else:
                    st.info("ℹ️ Flagged as Spam primarily by machine learning vocabulary patterns.")

                # Safety recommendation
                st.error("🛡️ **Safety Recommendation:** Do not click links, do not reply, and never share OTPs or banking passwords.")

            else:
                # NORMAL (HAM) RESULT
                st.markdown(f"""
                <div class="result-box-ham">
                    <h3 style="color: #15803d; margin: 0;">✅ Prediction: NORMAL (HAM) MESSAGE</h3>
                    <p style="margin: 6px 0 0 0; color: #4b5563;">
                        Confidence Score: <b>{prob_ham:.1f}% Normal</b> (Spam: {prob_spam:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Suspicious words note if any
                if detected_patterns:
                    st.warning(f"⚠️ **Note:** Contains words like `{', '.join(detected_patterns)}`, but overall message structure appears normal.")
                else:
                    st.success("✅ **Pattern Check:** No common scam triggers or phishing links detected.")

                # Safety recommendation
                st.info("💡 **Safety Recommendation:** This message appears legitimate, but always verify unfamiliar senders before sharing personal information.")


# ---------------------------------------------------------------------------
# SECTION 2: MODEL DASHBOARD
# ---------------------------------------------------------------------------
elif app_mode == "📊 Model Dashboard":
    st.subheader("📊 Model Performance & Evaluation Dashboard")
    st.write("These metrics are calculated on the test dataset (20% holdout split) during model training.")

    if metrics:
        # Display 4 Key Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        with col2:
            st.metric("Spam Precision", f"{metrics['precision'] * 100:.2f}%")
        with col3:
            st.metric("Spam Recall", f"{metrics['recall'] * 100:.2f}%")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

        st.markdown("---")

        # Dataset Split Summary & Confusion Matrix
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown("### 📁 Dataset Summary")
            st.write(f"- **Total SMS Messages:** {metrics.get('total_samples', 5574)}")
            st.write(f"- **Normal (Ham) Messages:** {metrics.get('ham_count', 4827)} ({metrics.get('ham_count', 4827)/metrics.get('total_samples', 5574)*100:.1f}%)")
            st.write(f"- **Spam Messages:** {metrics.get('spam_count', 747)} ({metrics.get('spam_count', 747)/metrics.get('total_samples', 5574)*100:.1f}%)")
            st.write(f"- **Training Samples (80%):** {metrics.get('train_samples', 4459)}")
            st.write(f"- **Testing Samples (20%):** {metrics.get('test_samples', 1115)}")
            st.info("**Stratified Splitting:** Maintained equal Spam/Ham ratio in both training and test sets.")

        with col_right:
            st.markdown("### 🔲 Confusion Matrix")
            tn = metrics.get('tn', 0)
            fp = metrics.get('fp', 0)
            fn = metrics.get('fn', 0)
            tp = metrics.get('tp', 0)

            # Confusion Matrix Table
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual Normal", "Actual Spam"],
                columns=["Predicted Normal", "Predicted Spam"]
            )
            st.table(cm_df)

            # Simple visual plot of the Confusion Matrix
            fig, ax = plt.subplots(figsize=(4, 3))
            cax = ax.matshow([[tn, fp], [fn, tp]], cmap=plt.cm.Blues, alpha=0.8)
            for i in range(2):
                for j in range(2):
                    val = [[tn, fp], [fn, tp]][i][j]
                    ax.text(j, i, str(val), va='center', ha='center', fontsize=12, fontweight='bold')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred Normal', 'Pred Spam'])
            ax.set_yticklabels(['Act Normal', 'Act Spam'])
            plt.title("Confusion Matrix Heatmap", fontsize=10, pad=10)
            st.pyplot(fig, use_container_width=False)

            st.caption(f"• **True Negatives ({tn}):** Correctly identified Normal messages\n"
                       f"• **True Positives ({tp}):** Correctly identified Spam messages\n"
                       f"• **False Positives ({fp}):** Normal messages falsely marked as Spam\n"
                       f"• **False Negatives ({fn}):** Spam messages missed")
    else:
        st.info("Metrics not found in model file. Re-run `python train_model.py` to refresh.")


# ---------------------------------------------------------------------------
# SECTION 3: ABOUT PROJECT
# ---------------------------------------------------------------------------
elif app_mode == "📖 About Project":
    st.subheader("📖 About ScamShield AI")
    st.markdown("""
    ### 1. Problem Statement
    SMS scams and spam messages trick users into clicking phishing links, sharing one-time passwords (OTPs),
    or sending money. Traditional keyword filters fail against changing spam text. ScamShield AI uses
    **Machine Learning** to automatically identify spam messages with high accuracy.

    ### 2. Project Objectives
    1. **Accurate Classification:** Classify incoming SMS as either **Spam** or **Normal (Ham)**.
    2. **Explainable Warnings:** Detect suspicious trigger words (e.g., *urgent, lottery, verify*) to inform users.
    3. **Lightweight & Fast:** Provide a simple, fast interface suitable for demonstration and viva presentation.

    ### 3. Machine Learning Architecture
    The system follows a simple 4-step pipeline:
    ```
    SMS Message -> Text Cleaning -> TF-IDF Vectorizer -> Multinomial Naive Bayes -> Prediction
    ```
    - **TF-IDF (Term Frequency - Inverse Document Frequency):** Converts text into numerical values, giving higher importance to distinctive words.
    - **Multinomial Naive Bayes:** Calculates the probability that a message belongs to Spam or Normal class based on Bayes' Theorem:
      $$P(\\text{Spam} \\mid \\text{Words}) = \\frac{P(\\text{Words} \\mid \\text{Spam}) \\cdot P(\\text{Spam})}{P(\\text{Words})}$$

    ### 4. Dataset
    - **Source:** UCI Machine Learning Repository - SMS Spam Collection.
    - **Size:** 5,574 real SMS messages tagged as legitimate (Ham) or Spam.
    """)
