# 🎓 Student Performance Predictor (GPA)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-predictor-gpa-app.streamlit.app/)
[![Model](https://img.shields.io/badge/Model-Linear%20Regression-blue)]()
[![Domain](https://img.shields.io/badge/Domain-EdTech-green)]()

## 🚀 Live Demo
The application is deployed and active! Click the button below to test it:
> **[👉 Launch Student GPA Predictor](https://student-performance-predictor-gpa-app.streamlit.app/)**

## 📋 Project Overview
Academic performance is influenced by a complex mix of environmental, behavioral, and demographic factors. This project utilizes **Machine Learning** to predict a student's GPA based on habits such as weekly study time, absences, parental support, and extracurricular activities.

The goal is to provide a predictive tool for **Educators and Parents** to identify "at-risk" students early and intervene before grades suffer.

## 📊 Key Insights (EDA)
Exploratory analysis of the dataset revealed distinct patterns affecting student success:

* **The Attendance Factor:** A strong negative correlation exists between **Absences** and GPA. Students with >15 absences rarely achieve a GPA above 2.5.
* **Study Habits:** Students studying **10+ hours/week** consistently maintain a GPA above 3.0.
* **Support Systems:** Higher levels of **Parental Support** and participation in **Extracurriculars** correlate positively with academic stability.

## 🤖 The Model & Logic
We trained a **Linear Regression** model using `Scikit-Learn` to predict the continuous GPA variable (0.0 - 4.0).

### Input Features
The model considers 12 key factors:
1.  **Academic:** Study Time Weekly, Absences, Tutoring.
2.  **Support:** Parental Support, Parental Education Level.
3.  **Activities:** Sports, Music, Volunteering, Extracurriculars.
4.  **Demographics:** Age, Gender, Ethnicity.

## 🛠️ Tech Stack
* **Python:** Core programming.
* **Streamlit:** Cloud-based frontend deployment.
* **Scikit-Learn:** Model training and Standardization (`StandardScaler`).
* **Pandas & Seaborn:** Data manipulation and EDA visualization.

