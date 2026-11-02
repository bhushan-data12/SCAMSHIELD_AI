"""
train_model.py
=============================================================================
ScamShield AI — SMS Spam Detection Using Machine Learning
Academic Mini-Project for College Presentation & Viva

Workflow:
1. Load SMS Spam Collection dataset (UCI Machine Learning Repository).
2. Clean and preprocess text data.
3. Split into 80% Training and 20% Testing sets.
4. Extract numerical features using TF-IDF (Term Frequency - Inverse Document Frequency).
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate model on test data (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Save the trained pipeline to 'sms_spam_model.joblib'.
=============================================================================
"""

import os
import urllib.request
import zipfile
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Constant paths
DATASET_PATH = "SMSSpamCollection"
DATASET_ALT_PATH = os.path.join("data", "SMSSpamCollection")
MODEL_OUTPUT_PATH = "sms_spam_model.joblib"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def ensure_dataset():
    """
    Ensures that the SMS Spam Collection dataset is available locally.
    If not found in root or 'data/' folder, downloads and extracts it from UCI.
    """
    if os.path.exists(DATASET_PATH):
        return DATASET_PATH
    if os.path.exists(DATASET_ALT_PATH):
        return DATASET_ALT_PATH

    print("[*] Dataset not found locally. Downloading from UCI Machine Learning Repository...")
    zip_path = "sms_spam_collection.zip"
    try:
        urllib.request.urlretrieve(UCI_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("[+] Download and extraction successful.")
        return DATASET_PATH
    except Exception as e:
        print(f"[-] Error downloading dataset: {e}")
        print("[-] Please ensure 'SMSSpamCollection' is placed in the project folder.")
        raise


def load_and_preprocess_data(filepath):
    """
    Loads tab-separated SMS data into a Pandas DataFrame.
    Maps labels: 'ham' -> 0 (Normal), 'spam' -> 1 (Spam).
    """
    print(f"\n[*] Loading dataset from: {filepath}")
    # Dataset has two columns separated by tab: label and message text
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['label', 'message'],
        quoting=3,
        encoding='latin-1'
    )

    # Drop any null or empty rows
    df = df.dropna(subset=['message', 'label']).copy()

    total = len(df)
    ham_count = int((df['label'] == 'ham').sum())
    spam_count = int((df['label'] == 'spam').sum())

    print(f"    - Total SMS messages : {total}")
    print(f"    - Normal (Ham) count : {ham_count} ({ham_count / total * 100:.1f}%)")
    print(f"    - Spam count         : {spam_count} ({spam_count / total * 100:.1f}%)")

    # Convert textual labels into binary numeric targets (0 = Normal, 1 = Spam)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    return df, total, ham_count, spam_count


def train_and_evaluate():
    """
    Main training function:
    1. Loads dataset
    2. Performs 80/20 train/test split
    3. Builds TF-IDF + Multinomial Naive Bayes pipeline
    4. Evaluates on test set
    5. Saves model to 'sms_spam_model.joblib'
    """
    print("=" * 60)
    print(" ScamShield AI - Model Training Pipeline ")
    print("=" * 60)

    dataset_file = ensure_dataset()
    df, total, ham_count, spam_count = load_and_preprocess_data(dataset_file)

    X = df['message']
    y = df['target']

    # Step 1: Train-Test Split (80% training, 20% testing)
    # 'stratify=y' ensures both train and test sets have the same ratio of Spam to Ham
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[*] Train set size: {len(X_train)} messages")
    print(f"[*] Test set size : {len(X_test)} messages")

    # Step 2: Build Scikit-Learn Pipeline
    # TfidfVectorizer converts raw text into numerical feature weights
    # MultinomialNB applies Bayes' theorem to calculate probabilities
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # unigrams and bigrams
            max_features=5000
        )),
        ('nb', MultinomialNB(alpha=0.1))
    ])

    print("\n[*] Training Multinomial Naive Bayes model...")
    pipeline.fit(X_train, y_train)

    # Step 3: Evaluate on holdout test set
    print("[*] Evaluating model on unseen test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 45)
    print(" MODEL PERFORMANCE EVALUATION RESULTS")
    print("=" * 45)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}% (Spam)")
    print(f"  Recall    : {rec * 100:.2f}% (Spam)")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print("-" * 45)
    print(" Confusion Matrix:")
    print(f"  True Negatives  (Normal correctly identified): {tn}")
    print(f"  False Positives (Normal falsely flagged as Spam): {fp}")
    print(f"  False Negatives (Spam missed)                  : {fn}")
    print(f"  True Positives  (Spam correctly caught)       : {tp}")
    print("=" * 45)

    # Attach computed metrics to the pipeline object for the Streamlit dashboard
    pipeline.metrics_ = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total_samples': int(total),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'ham_count': int(ham_count),
        'spam_count': int(spam_count)
    }

    # Step 4: Save trained pipeline to disk
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print(f"\n[+] Trained model and metrics successfully saved to '{MODEL_OUTPUT_PATH}'!")
    return pipeline


if __name__ == '__main__':
    train_and_evaluate()
