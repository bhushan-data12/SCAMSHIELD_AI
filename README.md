# 🛡️ ScamShield AI — SMS Spam Detection Using Machine Learning

> A simple, lightweight, and honest machine learning mini-project for college presentation and viva defense.  
> **Algorithm:** Multinomial Naive Bayes | **Feature Extraction:** TF-IDF Vectorizer | **Interface:** Streamlit  

---

## 📌 1. Project Overview & Essential Files

This project takes an SMS text message as input and predicts whether it is **Spam** or **Normal (Ham)** using a trained Machine Learning model. It also checks for suspicious keywords/links, displays a short safety recommendation, and provides an interactive performance dashboard with a confusion matrix.

The project is structured with minimal files to keep it simple, clean, and easy to explain:
- **`app.py`** — Streamlit web interface and prediction logic.
- **`train_model.py`** — Dataset loading, preprocessing, 80/20 train-test split, model training, evaluation, and saving.
- **`requirements.txt`** — List of required Python libraries.
- **`sms_spam_model.joblib`** — Serialized trained pipeline containing the TF-IDF vectorizer, Naive Bayes classifier, and calculated evaluation metrics.
- **`SMSSpamCollection`** — Real public SMS dataset from the UCI Machine Learning Repository.
- **`README.md`** — Execution instructions, viva prep, presentation script, and project documentation.

---

## 💻 2. How to Run on Windows (VS Code PowerShell)

Follow these exact commands in your VS Code PowerShell terminal:

### Step 1: Open Terminal in Project Folder
Open VS Code, press ``Ctrl + ` `` to open the PowerShell terminal in your project directory (`PROJECT_BSA`).

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
*(You will see `(venv)` appear at the start of your terminal prompt).*

### Step 3: Install Required Libraries
```powershell
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model
```powershell
python train_model.py
```
> **What this does:** Loads the 5,574 SMS messages from `SMSSpamCollection`, splits them 80/20 into train/test sets, fits the TF-IDF + Multinomial Naive Bayes pipeline, evaluates real test accuracy and confusion matrix, and saves everything to `sms_spam_model.joblib`.

### Step 5: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Your default browser will automatically open:
👉 **`http://localhost:8501`**

---

## 📊 3. Honest Model Evaluation Results

*(Calculated on 1,115 unseen holdout test messages: 966 Normal, 149 Spam)*

| Evaluation Metric | Actual Result | Simple Explanation |
| :--- | :---: | :--- |
| **Accuracy** | **98.57%** | Out of 1,115 test messages, 1,099 were correctly classified. |
| **Spam Precision** | **98.54%** | When the model flags a message as Spam, it is correct 98.5% of the time (only 2 false alarms!). |
| **Spam Recall** | **90.60%** | The model successfully caught 135 out of 149 real spam messages. |
| **F1-Score** | **94.41%** | The harmonic mean balancing precision and recall. |

### Confusion Matrix Breakdown
| | Predicted Normal (Ham) | Predicted Spam |
| :--- | :---: | :---: |
| **Actual Normal** | **964 (True Negatives)** | **2 (False Positives)** |
| **Actual Spam** | **14 (False Negatives)** | **135 (True Positives)** |

- **True Negatives (964):** Legitimate messages correctly classified as Normal.
- **True Positives (135):** Spam messages correctly caught.
- **False Positives (2):** Only 2 legitimate messages were mistakenly flagged as spam.
- **False Negatives (14):** 14 spam messages slipped past as normal.

---

## 🎯 4. College Presentation Materials

### A. Problem Statement
With the widespread use of mobile phones, fraudulent SMS messages (phishing, fake lottery winnings, KYC traps, and OTP harvesting) cause severe financial loss and identity theft. Manual filtering or simple keyword blacklists are unreliable because fraudsters continually change their phrasing. An automated, intelligent system is needed to accurately detect spam messages before users become victims.

### B. Three Project Objectives
1. **Accurate Binary Classification:** Build and train a machine learning model to classify SMS messages as either **Spam** or **Normal (Ham)** with high accuracy and low false positives.
2. **Explainable Warnings & Safety Advice:** Highlight detected suspicious patterns (e.g., urgency, prize lures, links) and provide actionable safety recommendations to protect users.
3. **Interactive & Demonstrable Interface:** Deliver a clean, responsive Streamlit dashboard showing real-time predictions, confidence scores, and model performance metrics for academic defense.

### C. Short Abstract
ScamShield AI is an SMS spam detection system built using supervised machine learning in Python. The system uses the public UCI SMS Spam Collection dataset consisting of 5,574 real-world messages. Text data is preprocessed and converted into numerical feature vectors using **TF-IDF (Term Frequency - Inverse Document Frequency)**. A **Multinomial Naive Bayes** classifier is trained on an 80% training set and evaluated on an unseen 20% test set, achieving **98.57% accuracy** and **98.54% spam precision**. An interactive Streamlit web application allows users to test custom messages, view confidence probabilities, detect suspicious trigger words, and inspect confusion matrix metrics.

### D. Simple Explanation of Algorithm and Dataset
- **The Dataset:** The UCI SMS Spam Collection contains 5,574 labeled English SMS messages (4,827 legitimate Ham messages and 747 Spam messages).
- **TF-IDF Vectorizer:** Machines cannot understand raw text. TF-IDF transforms text into numbers by measuring how frequently a word appears in a message relative to how common it is across all messages, giving higher weight to informative words (e.g., *lottery*, *prize*, *urgent*).
- **Multinomial Naive Bayes:** A probabilistic classification algorithm based on Bayes' Theorem:
  $$P(\text{Spam} \mid \text{Words}) = \frac{P(\text{Words} \mid \text{Spam}) \times P(\text{Spam})}{P(\text{Words})}$$
  It assumes that individual words contribute independently to the probability of the message being spam. Despite this "naive" assumption, it is exceptionally fast, requires low compute power, and works well for text categorization.

---

## 🎙️ 5. Two-Minute Presentation Script for Faculty

> *"Good morning respected faculty members and examiners.*
>
> *Today, I am presenting my machine learning project: **ScamShield AI — SMS Spam Detection Using Machine Learning**.*
>
> *SMS phishing and fraudulent texts are among the most common social engineering attacks today. Attackers trick users into revealing OTPs, bank credentials, or clicking malicious links.*
>
> *To solve this, I built a complete end-to-end classification system using Python, Scikit-learn, and Streamlit.*
>
> *Our workflow is straightforward:*
> 1. *First, we use the benchmark **UCI SMS Spam Collection dataset**, which contains 5,574 real SMS messages.*
> 2. *We clean the text and perform an **80/20 stratified train-test split**.*
> 3. *Next, we convert raw text into numerical features using **TF-IDF vectorization**, which gives higher weight to distinctive words.*
> 4. *We train a **Multinomial Naive Bayes** classifier. We selected Naive Bayes because it is mathematically grounded in Bayes' Theorem, computationally lightweight, and proven to excel at text categorization.*
> 5. *On our holdout test set of 1,115 messages, our model achieved an **accuracy of 98.57%** and a **spam precision of 98.54%**, with only 2 false positives.*
>
> *Let me demonstrate the web application:*
> *(Click 'Load Normal Sample' -> Predict -> Show Normal result)*
> *Here, a legitimate college message is correctly identified as Normal with 99% confidence.*
> *(Click 'Load Spam Sample' -> Predict -> Show Spam result)*
> *And here, an urgent cash prize message is immediately flagged as Spam, showing the suspicious keywords detected and a safety recommendation.*
>
> *In conclusion, ScamShield AI provides a fast, accurate, and explainable solution to prevent SMS fraud. Thank you, and I am now ready for your questions."*

---

## ❓ 6. Ten Basic Viva Questions & Easy Answers

**Q1: What is the main objective of your project?**  
**A:** The objective is to automatically classify incoming SMS messages as either Spam or Normal (Ham) using machine learning, protecting mobile users from scams.

**Q2: Which machine learning algorithm did you use, and why?**  
**A:** I used **Multinomial Naive Bayes**. It is the standard benchmark algorithm for text classification because it uses word frequency counts, calculates probabilities using Bayes' Theorem, trains in seconds, and performs reliably on sparse text data.

**Q3: What does the 'Naive' in Naive Bayes mean?**  
**A:** It is called 'naive' because it assumes all features (words) are conditionally independent given the class label. Even though words in human language are often related, this assumption works very well in practice for spam filtering.

**Q4: What is TF-IDF and why did you use it?**  
**A:** TF-IDF stands for **Term Frequency - Inverse Document Frequency**. Machine learning models cannot process raw strings directly. TF-IDF converts text into numerical vectors. Term Frequency measures how often a word occurs in a message, while Inverse Document Frequency reduces the weight of common words (like 'the', 'is') and increases the weight of rare, informative words (like 'prize', 'claim').

**Q5: Which dataset did you use, and what is its composition?**  
**A:** I used the **UCI SMS Spam Collection dataset**. It contains 5,574 labeled messages: 4,827 (86.6%) legitimate messages (Ham) and 747 (13.4%) spam messages.

**Q6: What is the difference between Precision and Recall in spam detection?**  
**A:** 
- **Precision** measures: *"Of all messages the model flagged as Spam, how many were actually spam?"* High precision means very few false alarms (legitimate messages wrongly blocked).
- **Recall** measures: *"Of all actual spam messages out there, how many did the model catch?"*

**Q7: In spam detection, which is worse: a False Positive or a False Negative?**  
**A:** A **False Positive** is usually worse. If an important legitimate message (like a bank OTP, interview call, or family emergency) is mistakenly flagged as spam, the user might miss critical information. Our model has a very low false positive rate (only 2 out of 966 normal messages).

**Q8: What is a Confusion Matrix?**  
**A:** A confusion matrix is a 2x2 table that evaluates the performance of a classification model by comparing actual labels against predicted labels. It shows True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

**Q9: Why did you use Streamlit for the front-end?**  
**A:** Streamlit is a Python-native framework that allows developers to create clean, interactive web interfaces directly in Python without needing HTML, CSS, or JavaScript. It is ideal for rapid prototyping and academic demonstrations.

**Q10: How can this project be improved in the future?**  
**A:** Future enhancements could include supporting multi-lingual messages (like Hindi or Hinglish), integrating an SMS gateway API for real-time mobile filtering, and expanding the dataset with recent WhatsApp and UPI scam formats.

---

## 🧪 7. Two Sample SMS Messages for Testing

1. **Normal / Legitimate (Ham) Message:**
   > *"Hey, are you free for the project discussion in the college library at 3 PM today?"*  
   - **Expected Result:** Normal (Ham), Confidence ~99%, No scam triggers.

2. **Spam / Scam Message:**
   > *"URGENT! You have won a £1,000 cash prize or a free holiday. Call 09061701461 now to claim your reward immediately!"*  
   - **Expected Result:** Spam, Confidence ~99%, Triggers: `urgent`, `won`, `cash`, `prize`, `free`, `claim`.
