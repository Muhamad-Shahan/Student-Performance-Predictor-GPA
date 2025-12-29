import streamlit as st
import pandas as pd
import joblib

# Load Data and Model components
@st.cache_data
def load_data():
    # Ensure this filename matches your GitHub repo file
    return pd.read_csv('Student_performance_data.csv')

# Try to load model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.warning("Please ensure model.joblib and scaler.joblib are in the same folder as app.py.")
    st.stop() # Stop execution if model fails to load

# --- Page Configuration ---
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- Main Title and Description ---
st.title("🎓 Student GPA Predictor")
st.markdown("""
This application uses a machine learning model to estimate a student's Grade Point Average (GPA) based on various academic, demographic, and lifestyle factors.
Fill in the details below to generate a prediction.
""")
st.markdown("---")

# --- Sidebar Statistics ---
with st.sidebar:
    st.header("📊 GPA Statistics")
    st.markdown("Contextual data on average high school GPAs (4.0 scale):")
    
    st.subheader("By Demographics")
    st.markdown("""
    * **Overall Average:** ~3.00
    * **Female Students:** 3.10
    * **Male Students:** 2.90
    * **Asian/Pacific Islander:** 3.26
    * **White:** 3.09
    * **Hispanic:** 2.84
    * **Black:** 2.69
    """)
    
    st.subheader("By Parental Education")
    st.markdown("""
    * **High School Diploma:** ~2.60 - 2.80
    * **Bachelor's Degree:** ~3.00 - 3.20
    * **Graduate Degree:** ~3.20 - 3.40
    """)
    st.caption("Source: National Center for Education Statistics (NCES) & Recent Academic Studies")

# --- Input Form ---
st.header("📝 Student Profile")

with st.form("prediction_form"):
    # Group 1: Demographics & Background
    st.subheader("1. Demographics & Background")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 15, 18, 17)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        
    with col2:
        ethnicity = st.selectbox(
            "Ethnicity", 
            [0, 1, 2, 3], 
            format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x]
        )
        parent_edu = st.selectbox(
            "Parental Education Level", 
            [0, 1, 2, 3, 4], 
            format_func=lambda x: ["None", "High School", "Some College", "Bachelor's", "Higher"][x]
        )

    st.markdown("---")

    # Group 2: Academics & Habits
    st.subheader("2. Academic Habits")
    col3, col4 = st.columns(2)
    
    with col3:
        study_time = st.number_input("Weekly Study Time (Hours)", 0.0, 20.0, 10.0, step=0.5)
        absences = st.number_input("Total Absences (School Year)", 0, 30, 5)
        tutoring = st.radio("Receives Tutoring?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

    with col4:
        parent_support = st.selectbox(
            "Parental Support", 
            [0, 1, 2, 3, 4], 
            format_func=lambda x: ["None", "Low", "Moderate", "High", "Very High"][x]
        )
        
    st.markdown("---")

    # Group 3: Activities
    st.subheader("3. Extracurricular Activities")
    st.markdown("Check all that apply:")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        extracurricular = st.checkbox("General Extracurriculars")
    with col6:
        sports = st.checkbox("Sports Team")
    with col7:
        music = st.checkbox("Music/Arts")
    with col8:
        volunteering = st.checkbox("Volunteering")

    # Convert checkboxes to 0/1 for model
    extracurricular = 1 if extracurricular else 0
    sports = 1 if sports else 0
    music = 1 if music else 0
    volunteering = 1 if volunteering else 0

    st.markdown("###")
    
    # Submit Button
    submitted = st.form_submit_button("🔮 Predict GPA", type="primary", use_container_width=True)

# --- Prediction Logic ---
if submitted:
    # 1. Arrange inputs EXACTLY as model expects
    input_features = [
        age, gender, ethnicity, parent_edu, study_time, absences,
        tutoring, parent_support, extracurricular, sports, music, volunteering
    ]
    
    # 2. Create DataFrame
    input_df = pd.DataFrame([input_features], columns=[
        'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 
        'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering'
    ])
    
    # 3. Predict
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        # Clamp prediction between 0.0 and 4.0
        prediction = max(0.0, min(4.0, prediction))
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # Create columns for result and visual
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Estimated GPA", value=f"{prediction:.2f}")
        
        with res_col2:
            st.write("GPA Scale (0.0 - 4.0)")
            st.progress(prediction / 4.0)
            
            if prediction >= 3.5:
                st.success("🌟 Excellent! Keep up the great work!")
            elif prediction >= 3.0:
                st.info("✅ Good standing. Consistent effort pays off.")
            elif prediction >= 2.0:
                st.warning("⚠️ Average. Consider focusing on study habits or tutoring.")
            else:
                st.error("🚨 At Risk. Intervention or extra support may be needed.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
