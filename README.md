
---
title: Sentiment
emoji: 🐨
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
---

# 🧠 Manosparsh - Heart vs Brain Percentage Analyzer

Manosparsh is a unique AI model that analyzes a user-written paragraph and determines how emotionally or practically (logically) a person is reacting in a situation. It uses **sentiment analysis** and **emotion detection** models to compute a percentage split between heart (emotion-driven) and brain (logic-driven) responses.

---

## 🚀 Live Demo

Try it out here 👉 [Hugging Face Space](https://huggingface.co/spaces/PalMomo/manopercent)  
Just enter your paragraph and get instant emotional vs practical insight.

---

## 📂 File Structure

manopercent/ │ ├── app.py # Main Gradio app file ├── README.md # This file ├── requirements.txt # Python dependencies ├── SVM_Emotions.zip # Contains the SVM emotion detection model ├── nb_s.zip # Contains the Naive Bayes sentiment analysis model ├── .gitattributes # Git LFS configuration

yaml
Copy
Edit

---

## 🧠 How It Works

- **Input**: A paragraph describing a situation or feeling.
- **Model 1**: Naive Bayes-based **sentiment analysis** (Positive / Negative / Neutral).
- **Model 2**: SVM-based **emotion detection** (Joy, Anger, Fear, etc.).
- **Logic**: Combines outputs from both models using weighted logic to calculate:
  - 💓 Heart (emotional) %
  - 🧠 Brain (logical) %

---

## 📦 Requirements

The models are zipped and auto-loaded on startup.  
Dependencies listed in `requirements.txt`:

```txt
gradio==4.25.0
scikit-learn
nltk
numpy
joblib
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
