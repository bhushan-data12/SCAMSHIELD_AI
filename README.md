# 🛡️ ScamShield AI — Multi-Algorithm SMS Spam & Scam Detection System

> **Algorithms Implemented & Compared:** Soft Voting Ensemble, Multinomial Naive Bayes, Linear Support Vector Machine (Linear SVM), Logistic Regression  
> **Feature Extraction:** TF-IDF Vectorizer (Unigram + Bigram, 5,000 features) | **Frontend:** Streamlit 1.56  
> **Dataset:** UCI Machine Learning Repository — SMS Spam Collection (5,574 messages)  

---

## 📌 1. Project Overview & Architecture

ScamShield AI is an intelligent cybersecurity application that identifies fraudulent SMS messages, phishing lures, and deceptive web links. The system is designed for **academic defense, viva presentation, and practical cybersecurity evaluation**.

### Key Modules:
1. **🔍 Single SMS Predictor** — Real-time classification with confidence scores, risk gauge, keyword highlighting, and **simultaneous multi-model consensus**.
2. **⚖️ Algorithm Comparison & Benchmark** — Side-by-side performance comparison of 4 algorithms with accuracy/F1 charts, precision/recall trade-off plots, and individual confusion matrices.
3. **📁 Batch CSV/TXT Classifier** — Bulk classification of uploaded files with summary statistics and CSV export.
4. **🔗 Phishing URL & Deceptive Domain Inspector** — Analyzes extracted hyperlinks for URL shorteners, raw IP hosts, suspicious TLDs, and fraudulent banking subdomains.
5. **📖 Viva Defense & Architecture Guide** — In-depth mathematical formulas (Bayes' Theorem, Sigmoid, SVM Margin), architectural diagrams, and top viva Q&A.

---

## 📊 2. Comprehensive Algorithm Benchmark & Comparison

Evaluated on an **unseen holdout test set of 1,115 messages** (966 Normal, 149 Spam) using an 80/20 Stratified Split:

| Algorithm | Test Accuracy | Spam Precision | Spam Recall | F1-Score | False Positives (Alarms) | False Negatives (Missed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Soft Voting Ensemble** | **98.74%** | **98.56%** | **91.95%** | **95.14%** | **2** | **12** |
| 🥈 **Multinomial Naive Bayes** | **98.57%** | **98.54%** | **90.60%** | **94.41%** | **2** | **14** |
| 🥉 **Linear SVM (Calibrated)** | **98.39%** | **95.17%** | **92.62%** | **93.88%** | **7** | **11** |
| 🔒 **Logistic Regression** | **97.76%** | **100.00%** | **83.22%** | **90.84%** | **0** | **25** |

### 🏆 Which Algorithm Performs Better and Why?

1. **Soft Voting Ensemble wins Overall (Highest Accuracy & F1):**
   - Combines the complementary decision boundaries of Naive Bayes (probabilistic), Logistic Regression (linear), and SVM (maximum-margin hyperplane).
   - Probability averaging acts as a consensus filter that prevents individual model mistakes.

2. **Logistic Regression wins for Zero False Alarms (100% Precision):**
   - Out of 966 legitimate messages, it achieved **0 false positives**. 
   - Ideal for scenarios where accidentally blocking an OTP or banking alert is unacceptable.

3. **Linear SVM wins for Maximum Spam Catch (92.62% Recall):**
   - High-dimensional text spaces (5,000 TF-IDF features) are linearly separable. SVM finds the optimal maximum-margin hyperplane, catching 138 out of 149 spam texts.

4. **Multinomial Naive Bayes wins for Speed & Simplicity:**
   - Extremely fast training and evaluation with 98.57% accuracy, making it the premier lightweight baseline.

---

## 💻 3. How to Run on Windows (PowerShell)

### Step 1: Open PowerShell in Project Directory
```powershell
cd c:\Users\Asus\Downloads\PROJECT_BSA
```

### Step 2: Activate Virtual Environment (if created)
```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Train & Benchmark All Algorithms
```powershell
python train_model.py
```
> Trains Naive Bayes, Logistic Regression, Linear SVM, and the Soft Voting Ensemble, evaluates test metrics, and exports `sms_spam_model.joblib`.

### Step 5: Launch the Web Dashboard
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎙️ 4. Two-Minute Presentation Script for Faculty & Viva

> *"Good morning respected examiners.*
>
> *Today, I am presenting **ScamShield AI** — a machine learning system for SMS spam detection and comparative algorithm benchmarking.*
>
> *SMS fraud (smishing, fake KYC updates, lottery winnings, and OTP harvesting) causes millions in financial losses every year. To address this, we developed a system that not only classifies messages with high accuracy, but also benchmarks multiple machine learning paradigms.*
>
> *Our engineering pipeline:*
> 1. *We utilized the benchmark **UCI SMS Spam Collection dataset** containing 5,574 messages.*
> 2. *We processed text using an **80/20 Stratified Split** and **TF-IDF vectorization** with unigram and bigram feature extraction.*
> 3. *We trained and benchmarked 3 diverse algorithms:*
>    - *A probabilistic model: **Multinomial Naive Bayes***
>    - *A linear classifier: **Logistic Regression***
>    - *A maximum-margin geometric classifier: **Support Vector Machine***
>    - *And a hybrid consensus model: **Soft Voting Ensemble***
>
> *Our findings:*
> - *The **Soft Voting Ensemble** achieved the highest overall accuracy of **98.74%** and an F1-score of **95.14%**.*
> - *The **Logistic Regression** model achieved a perfect **100% Spam Precision**, resulting in zero false alarms on legitimate messages.*
> - *The **Support Vector Machine** achieved the highest **Spam Recall (92.62%)**, intercepting the most spam.*
>
> *Our web interface also provides an Explainable AI keyword highlighter, batch file classification, and a phishing URL inspector.*
> *Thank you, and I look forward to your questions."*

---

## ❓ 5. Viva Q&A Cheat Sheet

1. **Why use both Precision and Recall instead of just Accuracy?**  
   *Answer:* The dataset is imbalanced (86.6% Ham, 13.4% Spam). A dummy model predicting everything as 'Normal' would achieve 86.6% accuracy while catching 0% spam. Precision and Recall evaluate performance on the minority spam class accurately.

2. **Why does Logistic Regression achieve 100% Precision?**  
   *Answer:* Logistic Regression applies L2 weight penalty, setting conservative decision boundaries for the spam class. It only flags messages as spam when there is overwhelming textual evidence, preventing false alarms.

3. **Why did you use CalibratedClassifierCV on Linear SVM?**  
   *Answer:* Linear SVM outputs signed distance from the decision hyperplane rather than probabilities. CalibratedClassifierCV applies Platt scaling (logistic sigmoid calibration) so we obtain reliable probability scores for confidence gauges and soft voting.

4. **What is the purpose of the Soft Voting Ensemble?**  
   *Answer:* Soft voting averages the predicted class probabilities across Naive Bayes, Logistic Regression, and SVM. This reduces model variance and minimizes individual misclassifications, yielding superior accuracy (98.74%).
