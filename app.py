import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data and Model components
@st.cache_data
def load_data():
    # Ensure this filename matches your GitHub repo file
    return pd.read_csv('Student_performance_data.csv')

df = load_data()

# Try to load model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.warning("Please ensure model.joblib and scaler.joblib are in the same folder as app.py.")

st.title("🎓 Student Performance Predictor (GPA)")
st.markdown("This app uses Machine Learning to predict student GPA based on academic and lifestyle factors.")

# Sidebar for Navigation
page = st.sidebar.selectbox("Navigate", ["Exploratory Data Analysis", "GPA Prediction"])

if page == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.write("Review the correlation between student habits and their GPA.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("GPA vs Absences")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Absences', y='GPA', ax=ax, color='red', alpha=0.5)
        st.pyplot(fig)
    
    with col_b:
        st.subheader("Weekly Study Time")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['StudyTimeWeekly'], kde=True, ax=ax2, color='blue')
        st.pyplot(fig2)

elif page == "GPA Prediction":
    st.header("🔮 Predict Student GPA")
    st.info("Fill in the student details below to estimate their GPA.")
    
    # Input Fields - Grouped for better UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal & Academic")
        age = st.slider("Age", 15, 18, 17)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3], help="0: Caucasian, 1: African American, 2: Asian, 3: Other")
        parent_edu = st.selectbox("Parental Education", [0, 1, 2, 3, 4], help="0: None, 1: High School, 2: Some College, 3: Bachelor, 4: Higher")
        study_time = st.number_input("Weekly Study Time (0-20 Hours)", 0.0, 20.0, 10.0)
        absences = st.number_input("Total Absences (0-30)", 0, 30, 5)

    with col2:
        st.subheader("Support & Activities")
        tutoring = st.radio("Tutoring Support", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        parent_support = st.selectbox("Parental Support Level", [0, 1, 2, 3, 4], help="0: None to 4: Very High")
        extracurricular = st.radio("Extracurricular Activities", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        sports = st.radio("Participates in Sports", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        music = st.radio("Participates in Music", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        volunteering = st.radio("Volunteering", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    # Prediction Logic
    if st.button("Calculate Estimated GPA"):
        # 1. Arrange all 12 inputs in the EXACT order of the training columns
        # Features: Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, 
        # Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering
        input_features = [
            age, gender, ethnicity, parent_edu, study_time, absences,
            tutoring, parent_support, extracurricular, sports, music, volunteering
        ]
        
        # 2. Convert to DataFrame to maintain column structure
        input_df = pd.DataFrame([input_features])
        
        # 3. Apply the fitted scaler and make prediction
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            
            # Display results
            st.success(f"### Predicted Student GPA: {prediction[0]:.2f}")
            
            # Visual feedback
            st.progress(min(prediction[0]/4.0, 1.0)) # GPA is usually on a 4.0 scale
            
        except Exception as e:
            st.error(f"Prediction failed. Check if the scaler matches the 12 features. Error: {e}")
