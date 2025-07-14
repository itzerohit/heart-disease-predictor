# heart-disease-predictor
Machine Learning app to predict heart disease based on patient health data.
# ❤️ Heart Disease Prediction App

A Machine Learning web application built with **Streamlit** that predicts whether a person is likely to have heart disease based on their medical information.

---

## 🔍 Project Overview

This project aims to use various ML models to predict heart disease based on patient data such as age, cholesterol levels, resting blood pressure, etc. It includes an interactive **Streamlit app** that takes user input and displays predictions from multiple trained models.

---

## 🚀 Demo

Try the live demo (if deployed):  
**[🔗 Live App](#)** ← *(Add your link here if you deploy on Streamlit Cloud or Render)*

---

## 📊 Features

- Accepts patient inputs through a simple UI
- Runs 5 ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - GridSearchCV-Optimized Random Forest
- Displays prediction result from each model
- Trained model files (`.pkl`) are used for fast loading

---

## 🧠 Models Used

The following models were trained on the [Heart Disease UCI dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction):

| Model               | Accuracy (example) |
|---------------------|--------------------|
| Logistic Regression | 86%                |
| Decision Tree       | 82%                |
| Random Forest       | 88%                |
| SVM                 | 85%                |
| GridSearch RF       | 89%                |

---
