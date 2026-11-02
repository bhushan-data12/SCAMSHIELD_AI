# ScamShield AI: Intelligent Scam Message Analysis and Risk Assessment

**Department of Artificial Intelligence and Data Science**  
**Second-Year Engineering Academic Project Report**  
**Core Algorithm:** Multinomial Naive Bayes (Supervised Machine Learning)  
**Technologies:** Python 3.10 | scikit-learn | TF-IDF | Streamlit | Plotly  

---

## Abstract

With the rapid expansion of mobile communications, short messaging services (SMS) and instant chat applications have become primary channels for smishing, financial fraud, and credential harvesting. This project presents **ScamShield AI**, an intelligent, explainable message risk assessment system developed using Python, scikit-learn, and Streamlit.

The project is built around a single, highly effective supervised learning algorithm: **Multinomial Naive Bayes**, paired with sublinear **TF-IDF (unigram and bigram)** feature extraction. Trained on the benchmark **UCI SMS Spam Collection** (5,574 messages), the model achieved **98.65% test accuracy**, **98.55% spam precision**, **91.28% spam recall**, and a **94.77% spam F1-score** under an 80/20 stratified split, producing only 2 false positives out of 966 legitimate messages (99.79% specificity).

To ensure defense-in-depth, ScamShield AI integrates a 6-category deterministic regular expression engine scanning for urgency, prize lures, credential harvesting, unusual payment demands, URL shorteners, and account security/KYC notices. Model decisions are explained through word-level log-likelihood ratios, and combined into a transparent heuristic risk tier (**Low**, **Moderate**, **Elevated**, **High Risk**). The system runs entirely locally on a standard Windows laptop in VS Code without requiring paid APIs or deep learning infrastructure.

---

## 1. Problem Statement & Motivation

### 1.1 Problem Statement
*How can a second-year engineering student build an accurate, lightweight, and explainable message risk assessment application using a single foundational machine learning algorithm that runs locally in real time?*

### 1.2 Motivation
1. **Academic Clarity:** Focusing on **Multinomial Naive Bayes** allows a second-year student to understand and clearly explain every mathematical step—from Bayes' Theorem and Laplace smoothing to TF-IDF weights and log-likelihood ratios.
2. **False Positive Sensitivity:** In mobile messaging, discarding a legitimate message (such as a bank OTP or family emergency alert) has severe consequences. Naive Bayes achieved **98.55% precision**, keeping false alarms to a minimum.
3. **Defense-in-Depth:** Combining the statistical ML classifier with a rule engine provides layered protection against emerging threats that historical datasets might not capture.

---

## 2. Algorithm & Technical Formulation

### 2.1 Multinomial Naive Bayes
Multinomial Naive Bayes applies Bayes' Theorem under the feature conditional independence assumption:
$$P(\text{Spam} \mid \mathbf{x}) \propto P(\text{Spam}) \prod_{i=1}^{m} P(x_i \mid \text{Spam})$$

To prevent zero-probability errors when an unseen test word appears in a message, Laplace smoothing is applied:
$$\theta_{\text{Spam}, i} = \frac{N_{\text{Spam}, i} + \alpha}{N_{\text{Spam}} + \alpha |V|}$$
where $\alpha = 0.1$ was tuned for optimal precision on spam, $|V|$ is the vocabulary size (5,000 features), and $N_{\text{Spam}, i}$ is the frequency of word $i$ in spam messages.

### 2.2 TF-IDF Feature Extraction
Term Frequency-Inverse Document Frequency converts unstructured text strings into numerical vectors:
$$\text{TF-IDF}(t, d, D) = (1 + \ln \text{TF}(t, d)) \times \left( \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1 \right)$$
- `ngram_range=(1, 2)`: Captures both individual words ("prize") and critical phrases ("claim now", "cash prize").
- `sublinear_tf=True`: Dampens the effect of repetitive word usage.
- **Leakage Prevention:** The vectorizer was fitted strictly on `X_train` after stratified splitting.

---

## 3. Dataset & Empirical Evaluation Results

### 3.1 UCI SMS Spam Collection
- Total Messages: **5,574**
- Ham (Legitimate): **4,827 (86.6%)**
- Spam: **747 (13.4%)**
- Split: **80% Train (4,459) / 20% Test (1,115)** with stratified sampling.

### 3.2 Evaluation Metrics (Holdout Test Set)

| Metric | Score | Practical Meaning |
| :--- | :---: | :--- |
| **Accuracy** | **98.65%** | Percentage of all test messages correctly classified |
| **Spam Precision** | **98.55%** | When flagged as Spam, 98.55% are truly spam (only 2 false alarms) |
| **Spam Recall** | **91.28%** | 136 out of 149 actual spam messages caught |
| **Spam F1-Score** | **94.77%** | Harmonic mean balancing precision and recall |
| **Ham Specificity**| **99.79%** | 964 out of 966 legitimate messages correctly cleared |

### 3.3 Confusion Matrix (966 Ham, 149 Spam)
$$\begin{pmatrix} \text{TN} = 964 & \text{FP} = 2 \\ \text{FN} = 13 & \text{TP} = 136 \end{pmatrix}$$
- **False Positives (FP = 2):** Only 2 legitimate messages were incorrectly flagged as spam.
- **False Negatives (FN = 13):** 13 spam messages were missed.

---

## 4. Multi-Signal Pattern Engine & Explainability

### 4.1 Six Rule Categories
1. **Urgency & Coercion (+20 pts):** "immediately", "within 24 hours", "account suspended".
2. **Unexpected Prizes & Rewards (+15 pts):** "congratulations", "you have won", "cash prize".
3. **Credential Harvesting (+35 pts - Critical):** "OTP", "PIN", "password", "CVV".
4. **Unusual Payments (+25 pts):** "wire transfer", "Western Union", "bitcoin", "processing fee".
5. **Suspicious Links & Shorteners (+25 pts):** `bit.ly`, `tinyurl`, raw IP addresses, `.xyz`.
6. **Account Security / KYC (+20 pts):** "update KYC", "verify identity", "unauthorized login".

### 4.2 Composite Risk Synthesis
$$\text{Composite Risk Score} = (P(\text{Spam} \mid \mathbf{x}) \times 50\%) + \left(\frac{\text{Rule Points}}{100} \times 50\%\right)$$
- 🟢 **LOW RISK (0–24%):** ML Ham + 0 rules triggered. Routine communication.
- 🟡 **MODERATE RISK (25–49%):** Mild promotional language or single urgency word.
- 🟠 **ELEVATED RISK (50–74%):** Strong spam signals or multiple suspicious patterns.
- 🔴 **HIGH RISK (75–100%):** Critical phishing markers, OTP solicitation, or payment demands.

### 4.3 Word Explainability
Calculates the log-likelihood ratio for each active token:
$$\text{Log-Ratio}(w) = \ln P(w \mid \text{Spam}) - \ln P(w \mid \text{Ham})$$
- **Positive:** Words pushing toward **Spam** (e.g., `claim`, `prize`, `txt`, `urgent`).
- **Negative:** Words anchoring in **Ham** (e.g., `ok`, `meeting`, `tomorrow`, `home`).

---

## 5. Top Viva Examination Questions & Answers

1. **Q: Why did you choose Multinomial Naive Bayes as your single algorithm?**  
   *A:* Multinomial Naive Bayes is the standard foundation for text classification. It operates on word count/TF-IDF distributions using Bayes' theorem. It achieved 98.65% accuracy and 98.55% precision on our dataset, runs with microsecond speed, and requires no heavy hardware.
2. **Q: What is the mathematical equation of Bayes' Theorem?**  
   *A:* $P(\text{Spam} \mid \text{Message}) = \frac{P(\text{Spam}) \prod P(\text{word}_i \mid \text{Spam})}{P(\text{Message})}$. It multiplies the prior probability of spam by the likelihood of each word appearing in spam under the conditional independence assumption.
3. **Q: What is Laplace smoothing and why is $\alpha=0.1$ used?**  
   *A:* Laplace smoothing adds a small pseudo-count ($\alpha$) to every word count so that an unseen word does not result in a zero-probability error. We set $\alpha=0.1$ to prevent over-smoothing.
4. **Q: What is TF-IDF and why use unigrams and bigrams?**  
   *A:* TF-IDF balances term frequency with inverse document frequency, giving high weight to words specific to spam and downweighting common words like 'the'. Unigrams capture single words ('prize'), while bigrams capture phrases like 'cash prize' or 'claim now'.
5. **Q: Why is high Precision more important than high Recall in spam filtering?**  
   *A:* A False Positive means a legitimate personal or banking message is mistakenly flagged as spam, which can cause severe harm to the user. High precision guarantees that when the model flags a message as spam, it is almost certainly spam (our model achieved 98.55% precision).
6. **Q: How did you prevent data leakage?**  
   *A:* We performed stratified train-test splitting (80% train, 20% test) before vectorization. The TfidfVectorizer was fitted strictly on `X_train` and used only to transform `X_test`.
7. **Q: Why combine ML with a Rule-Based Engine?**  
   *A:* Machine learning models rely on historical training data. If a scammer uses new wording or spelling, the ML model might not catch it. The rule engine scans for active structural threats (like asking for an OTP or using a bit.ly link), providing defense-in-depth.
