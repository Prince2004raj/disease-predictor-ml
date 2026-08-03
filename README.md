# 🩺 Disease Predictor using Machine Learning

A desktop application that predicts possible diseases based on user-reported symptoms, using and comparing three different machine learning classification algorithms.

## 📌 About the Project

Disease Predictor helps simplify early symptom-based screening by letting a user select up to 5 symptoms and instantly compare predictions from three different ML models side-by-side — instead of relying on a single black-box answer.

It provides:
- Multi-model disease prediction (Decision Tree, Random Forest, Naive Bayes)
- Confidence-based, data-driven predictions across 41 diseases and 132 symptoms
- Severity-based alerts for life-threatening conditions
- General safety precautions for the predicted disease
- A built-in emergency contact reference

⚠️ **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice or diagnosis.

## ⚙️ Tech Stack

**Language:** Python

**Libraries:**
- Pandas — data loading & preprocessing
- Scikit-learn — model training (Decision Tree, Random Forest, Naive Bayes)
- NumPy — numerical operations
- Tkinter — desktop GUI

**Dataset:** 4,920 training records across 41 diseases and 132 symptoms

## ✨ Features

### 🧠 Multi-Model Prediction
- Predicts disease using three separate ML algorithms
- Lets users compare outputs across models instead of trusting one

### 🚨 Severity Alerts
- Automatically flags life-threatening predictions (e.g. Heart attack, Tuberculosis) with a visible red alert

### 📋 Precautions
- Displays relevant, general safety precautions for the predicted condition
- No prescriptive medical advice — directs users to seek professional care

### 🆘 Emergency Access
- Fixed on-screen emergency number for quick access during urgent situations

## 🏗️ Project Structure

```
disease-predictor-ml/
│
├── disease_predictor_v2.py     # Main application (GUI + prediction logic)
├── symptom_precaution.csv      # Precaution data mapped to each disease
├── Training.csv                # Training dataset
├── Testing.csv                 # Testing dataset
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 How to Run

1. Clone this repository
   ```
   git clone https://github.com/Prince2004raj/disease-predictor-ml.git
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Run the application
   ```
   python disease_predictor_v2.py
   ```

## 🧠 Skills Demonstrated

**Machine Learning**
- Multi-class classification with Decision Tree, Random Forest, and Naive Bayes
- Model evaluation using accuracy metrics
- Label encoding for categorical target variables

**Data Handling**
- Pandas-based data cleaning and preprocessing
- Dynamic feature extraction directly from dataset (avoids hardcoded mismatches)

**Application Development**
- Desktop GUI development with Tkinter
- Event-driven programming
- State management across user interactions

## 🔮 Planned Enhancements

- Web-based version with login & prediction history ("report card" for each check)
- Multi-language support (Hindi, English)
- Symptom search/autocomplete
- Nearby hospital lookup based on user location

## 👤 Author

**Prince Raj**
[GitHub](https://github.com/Prince2004raj)
