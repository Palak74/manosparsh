import subprocess
import sys
import os
import zipfile
import joblib
import pandas as pd
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# File paths
SVM_ZIP = "SVM_Emotions.zip"
NB_ZIP = "nb_s.zip"
EXTRACT_FOLDER = "extracted_files"
SVM_FOLDER = os.path.join(EXTRACT_FOLDER, "SVM_Emotions")
NB_FOLDER = os.path.join(EXTRACT_FOLDER, "nb_s")
SVM_MODEL_PATH = os.path.join(SVM_FOLDER, "SVM_Emotions_model.pkl")
NB_MODEL_PATH = os.path.join(NB_FOLDER, "nb_s_model.pkl")
TRAIN_FILE = os.path.join(SVM_FOLDER, "train.txt")
TRAIN_CSV = os.path.join(NB_FOLDER, "twitter_training.csv")

# Extraction
os.makedirs(EXTRACT_FOLDER, exist_ok=True)
for zip_path, folder in [(SVM_ZIP, SVM_FOLDER), (NB_ZIP, NB_FOLDER)]:
    if not os.path.exists(folder):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_FOLDER)

# Load emotion data
def load_emotion_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split(";")
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(parts[1])
    return texts, labels

# SVM model (emotion)
if os.path.exists(SVM_MODEL_PATH):
    svm_vectorizer, svm_model = joblib.load(SVM_MODEL_PATH)
else:
    train_texts, train_labels = load_emotion_data(TRAIN_FILE)
    svm_vectorizer = TfidfVectorizer()
    X_train = svm_vectorizer.fit_transform(train_texts)
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X_train, train_labels)
    joblib.dump((svm_vectorizer, svm_model), SVM_MODEL_PATH)

# Naive Bayes (sentiment)
if os.path.exists(NB_MODEL_PATH):
    nb_model = joblib.load(NB_MODEL_PATH)
else:
    df = pd.read_csv(TRAIN_CSV, encoding="ISO-8859-1", usecols=["text", "sentiment"]).dropna()
    nb_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    nb_model.fit(df["text"], df["sentiment"])
    joblib.dump(nb_model, NB_MODEL_PATH)

# 🧠❤️ Manosparsh Logic
def manosparsh_logic(emotion_percentages, sentiment_result):
    emotion_weights = {
        'joy':       (0.6, 0.4),
        'sadness':   (0.9, 0.1),
        'anger':     (0.7, 0.3),
        'fear':      (0.8, 0.2),
        'love':      (0.7, 0.3),
        'surprise':  (0.4, 0.6),
        'neutral':   (0.5, 0.5)
    }

    emotional_score = 0
    practical_score = 0

    for emotion, value in emotion_percentages.items():
        emo_weight, prac_weight = emotion_weights.get(emotion.lower(), (0.5, 0.5))
        emotional_score += value * emo_weight
        practical_score += value * prac_weight

    total = emotional_score + practical_score
    if total != 0:
        emotional_score = (emotional_score / total) * 100
        practical_score = 100 - emotional_score

    adjustments = {
        'positive': (+5, -5),
        'negative': (-5, +5),
        'irrelevant': (0, 0)
    }
    emo_adj, prac_adj = adjustments.get(sentiment_result.lower(), (0, 0))

    emotional_score = max(0, min(100, emotional_score + emo_adj))
    practical_score = max(0, min(100, practical_score + prac_adj))

    emotional_score = round(emotional_score)
    practical_score = 100 - emotional_score

    return emotional_score, practical_score

# ✅ Updated Single Prediction
def predict_both(text):
    if not text.strip():
        return "<div style='color:red;'>Please enter a sentence. Textbox is empty.</div>", "", ""

    proba = svm_model.predict_proba(svm_vectorizer.transform([text]))[0]
    emotion_labels = svm_model.classes_

    # Plot emotion confidence as bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(emotion_labels, proba * 100, color="skyblue")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Emotion Prediction")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}%', va='center')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_html = f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'/>"

    sentiment_prediction = nb_model.predict([text])[0]
    sentiment_html = f"<div style='font-weight:bold; font-size: 18px; margin-top: 10px;'>Predicted Sentiment:</div><div style='font-size: 16px;'>{sentiment_prediction}</div>"

    emotion_percentages = {label: prob * 100 for label, prob in zip(emotion_labels, proba)}
    emo_score, prac_score = manosparsh_logic(emotion_percentages, sentiment_prediction)

    manosparsh_html = f"<div style='margin-top: 15px; font-weight: bold; font-size: 18px;'>🧠 Mind vs ❤️ Heart Prediction</div><div style='font-size: 16px;'>🧠 Practical Thinking: {prac_score}%<br>❤️ Emotional Thinking: {emo_score}%</div>"

    return img_html, sentiment_html, manosparsh_html

# 📄 Batch Prediction
def process_file(file):
    df = pd.read_csv(file.name, header=None, names=["Text"])
    rows = []
    for text in df["Text"]:
        proba = svm_model.predict_proba(svm_vectorizer.transform([text]))[0]
        emotion_labels = svm_model.classes_
        sentiment = nb_model.predict([text])[0]

        emotion_dict = {label: prob * 100 for label, prob in zip(emotion_labels, proba)}
        emo_score, prac_score = manosparsh_logic(emotion_dict, sentiment)

        row = {
            "Text": text,
            "Sentiment": sentiment,
            "Emotional Thinking (%)": emo_score,
            "Practical Thinking (%)": prac_score
        }
        for label in emotion_labels:
            row[label] = round(emotion_dict[label], 2)
        rows.append(row)

    return pd.DataFrame(rows)

# 📊 Model Evaluation
def model_performance():
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    train_texts, train_labels = load_emotion_data(TRAIN_FILE)
    X_test = svm_vectorizer.transform(train_texts)
    y_pred_svm = svm_model.predict(X_test)

    df = pd.read_csv(TRAIN_CSV, encoding="ISO-8859-1", usecols=["text", "sentiment"]).dropna()
    y_pred_nb = nb_model.predict(df["text"])
    y_true_nb = df["sentiment"].values

    # Confusion Matrices
    sns.heatmap(confusion_matrix(train_labels, y_pred_svm), annot=True, fmt="d",
                xticklabels=svm_model.classes_, yticklabels=svm_model.classes_, ax=axes[0, 0], cmap="Blues")
    axes[0, 0].set_title("SVM Emotion Confusion Matrix", fontsize=12)

    sns.heatmap(confusion_matrix(y_true_nb, y_pred_nb), annot=True, fmt="d",
                xticklabels=np.unique(y_true_nb), yticklabels=np.unique(y_true_nb), ax=axes[0, 1], cmap="Reds")
    axes[0, 1].set_title("Naive Bayes Sentiment Confusion Matrix", fontsize=12)

    # ROC Curve: SVM
    y_test_binarized = label_binarize(train_labels, classes=svm_model.classes_)
    y_score = svm_model.decision_function(X_test)
    for i in range(len(svm_model.classes_)):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, label=f"{svm_model.classes_[i]} (AUC = {roc_auc:.2f})")
    axes[1, 0].plot([0, 1], [0, 1], "k--")
    axes[1, 0].set_title("SVM Emotion ROC Curve", fontsize=12)
    axes[1, 0].legend(loc="lower right")

    # ROC Curve: Naive Bayes
    binarized_nb = label_binarize(y_true_nb, classes=np.unique(y_true_nb))
    nb_proba = nb_model.predict_proba(df["text"])
    for i in range(len(np.unique(y_true_nb))):
        fpr, tpr, _ = roc_curve(binarized_nb[:, i], nb_proba[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1, 1].plot(fpr, tpr, label=f"{np.unique(y_true_nb)[i]} (AUC = {roc_auc:.2f})")
    axes[1, 1].plot([0, 1], [0, 1], "k--")
    axes[1, 1].set_title("Naive Bayes Sentiment ROC Curve", fontsize=12)
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()

    report_svm = classification_report(train_labels, y_pred_svm, output_dict=False)
    report_nb = classification_report(y_true_nb, y_pred_nb, output_dict=False)

    return fig, f"**SVM Emotion Detection - Classification Report**\n\n```\n{report_svm}\n```", f"**Naive Bayes Sentiment Analysis - Classification Report**\n\n```\n{report_nb}\n```"

# 🖥️ Gradio Interfaces
iface = gr.Interface(
    fn=predict_both,
    inputs=gr.Textbox(label="Enter a sentence"),
    outputs=[
        gr.HTML(label="Emotion Detection (Visualized)"),
        gr.HTML(label="Sentiment Analysis"),
        gr.HTML(label="🧠❤️ Manosparsh Analysis")
    ],
    title="Sentiment & Emotion Detection + Manosparsh",
    description="Enter a sentence to get emotional + sentiment predictions and final Manosparsh analysis.",
    allow_flagging="never"
)

file_interface = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload a CSV file with sentences"),
    outputs=gr.Dataframe(type="pandas"),
    title="Batch Prediction with Manosparsh Output"
)

perf_interface = gr.Interface(
    fn=model_performance,
    inputs=[],
    outputs=[gr.Plot(label="Confusion Matrices & ROC Curves"),
             gr.Markdown(label="SVM Classification Report"),
             gr.Markdown(label="Naive Bayes Classification Report")],
    title="Model Performance (Accuracy Metrics)"
)

# Launch App
app = gr.TabbedInterface(
    [iface, file_interface, perf_interface],
    ["**Single Prediction**", "**Batch Prediction**", "**Model Performance**"]
)

if __name__ == "__main__":
    app.launch()