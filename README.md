# 🧠 Disease Prediction Website

A web application that predicts the likelihood of major health conditions — **Diabetes, Heart Disease, Lung Cancer, and Stroke** — using trained machine learning models on health-related input data. Designed to provide early warning and awareness to users, potentially saving lives.

---

## 🚀 Live Demo
🌐 [Click here to try the website]((https://web-production-2e7c.up.railway.app/))

---

## 🛠️ Tech Stack

| Frontend | Backend | Machine Learning | Deployment |
|----------|---------|------------------|------------|
| HTML5, CSS3 (via templates) | Flask (Python) | scikit-learn, pandas, pickle | Render / Railway / Heroku |
| JavaScript (minimal) | Jinja2 | Trained `.pkl` models | Gunicorn + WSGI |

---

## 🎯 Track: AI/ML + Full Stack

We chose **AI/ML** for disease prediction using healthcare datasets and **Full Stack Web Development** to build an interactive user-facing web application.

---

## 💡 Problem Statement

Early detection of diseases like **heart disease, diabetes, lung cancer, and stroke** is critical to improve survival rates and reduce treatment costs. However, access to immediate diagnosis tools can be limited in rural and underserved areas.

This project solves the problem by providing:
- Instant, lightweight predictions via a web interface
- An intuitive design requiring only basic health inputs
- Deployable on cloud platforms with minimal infrastructure

---

## 🧩 Features

- ✅ Separate model for each disease
- ✅ Clean and responsive HTML templates for each prediction page
- ✅ Preprocessing pipeline using a trained `scaler.pkl`
- ✅ Live deployment with WSGI + Flask
- ✅ Simple chatbot testing script (`test_chatbot.py`) for automation

---

## 🧪 ML Models Used

Each disease has its own trained model (saved as `.pkl`):
- `diabetes_model.pkl`
- `heart_disease_model.pkl`
- `lung_cancer_model.pkl`
- `stroke_model.pkl`

All models were trained using cleaned and balanced datasets, with accuracy ranging between **85–95%** depending on the dataset.

---

## 💸 Business Model *(Optional / If Applicable)*

This project can be extended to:
- Clinics or diagnostic centers as a screening tool
- Health insurance firms for risk scoring
- Telemedicine integration for remote diagnostics

Currently, it's open-source and freely available.

---

## 🏆 Bounties / Challenges Addressed

- ✅ Choosing a light theme for our website showing allegiance to jedi
- ✅ adding animation features (parallax) to our website
- ✅ Creating a mascot for our team

---

## 📁 Project Structure
DISEASE-PREDICTION-FINAL/
│
├── templates/ # HTML pages for each disease
├── static/ # CSS, JS (optional)
├── app.py # Main Flask application
├── *.pkl # Trained ML models
├── scaler.pkl # Preprocessing scaler
├── simple_test.py # Script for manual model tests
├── test_chatbot.py # Chatbot test script
├── wsgi.py # For deployment with Gunicorn
├── requirements.txt # Project dependencies
└── README.md # You're reading it!


---

## 📌 How to Run Locally

(bash)
git clone https://github.com/your-username/disease-prediction-final
cd disease-prediction-final
pip install -r requirements.txt
python app.py

Open http://localhost:5000 in your browser.



## License

This project is licensed under the MIT License.
