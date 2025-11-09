#  STUDENT PERFORMANCE PREDICTION - FRONTEND (Streamlit)

import streamlit as st
import numpy as np
import joblib

model = joblib.load("student_grade_predictor.pkl")
features_list = joblib.load("model_features.pkl")

st.set_page_config(page_title=" Student Grade Predictor", layout="centered")
st.title(" Student Performance Prediction System")
st.markdown(
    """
    This web app predicts a student's *final grade (G3)*  
    based on academic, social, and lifestyle factors.  
    Enter details below and get an instant grade prediction 
    """
)

st.sidebar.header(" Enter Student Information")

school_map = {"GP (Gabriel Pereira)": 0, "MS (Mousinho da Silveira)": 1}
sex_map = {"Female": 0, "Male": 1}
address_map = {"Urban": 1, "Rural": 0}
Mjob_map = {"Teacher": 4, "Health": 2, "Services": 3, "At home": 0, "Other": 1}
reason_map = {"Course": 0, "Home": 1, "Reputation": 2, "Other": 3}
higher_map = {"Yes": 1, "No": 0}
internet_map = {"Yes": 1, "No": 0}


school = st.sidebar.selectbox("School", list(school_map.keys()))
sex = st.sidebar.selectbox("Gender", list(sex_map.keys()))
age = st.sidebar.slider("Age", 15, 22, 17)
address = st.sidebar.selectbox("Address Type", list(address_map.keys()))
Medu = st.sidebar.slider("Mother's Education (0=none, 4=Higher Ed)", 0, 4, 2)
Fedu = st.sidebar.slider("Father's Education (0=none, 4=Higher Ed)", 0, 4, 2)
Mjob = st.sidebar.selectbox("Mother's Job", list(Mjob_map.keys()))
reason = st.sidebar.selectbox("Reason for Choosing School", list(reason_map.keys()))
traveltime = st.sidebar.slider("Travel Time to School (1=<15min, 4=>1hr)", 1, 4, 2)
studytime = st.sidebar.slider("Weekly Study Time (1=<2hr, 4=>10hr)", 1, 4, 2)
failures = st.sidebar.slider("Number of Past Failures", 0, 4, 0)
higher = st.sidebar.selectbox("Wants Higher Education?", list(higher_map.keys()))
internet = st.sidebar.selectbox("Internet Access at Home?", list(internet_map.keys()))
freetime = st.sidebar.slider("Free Time (1=very little, 5=lots)", 1, 5, 3)
Dalc = st.sidebar.slider("Workday Alcohol Consumption (1=low, 5=high)", 1, 5, 1)
Walc = st.sidebar.slider("Weekend Alcohol Consumption (1=low, 5=high)", 1, 5, 2)
G1 = st.sidebar.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.sidebar.slider("Second Period Grade (G2)", 0, 20, 10)


input_data = np.array([
    school_map[school],
    sex_map[sex],
    age,
    address_map[address],
    Medu,
    Fedu,
    Mjob_map[Mjob],
    reason_map[reason],
    traveltime,
    studytime,
    failures,
    higher_map[higher],
    internet_map[internet],
    freetime,
    Dalc,
    Walc,
    G1,
    G2
]).reshape(1, -1)


if st.button(" Predict Final Grade"):
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(20, prediction))

    st.subheader(" Predicted Final Grade (G3):")
    st.metric(label="Estimated Grade", value=f"{prediction:.2f}")


    if prediction >= 15:
        st.success(" Excellent performance! Keep it up ")
    elif prediction >= 10:
        st.info(" Average performance. Some improvements can make a big difference!")
    else:
        st.error("⚠ Needs improvement. Focus more on studies and consistency.")


st.markdown("---")
st.caption("Developed by Neha | Empowered by AI | Predicting the Future of Learning")