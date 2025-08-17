import streamlit as st
import pandas as pd
import cloudpickle

with open("mh_pipeline.pkl", "rb") as f:
    pipe = cloudpickle.load(f)
from datetime import datetime

# Load pipeline




st.set_page_config(page_title="Mental Health Treatment Prediction", layout="centered")

st.title("🧠 Mental Health Treatment Prediction")
st.write(
    """
    This tool predicts whether someone might seek mental health treatment based on workplace and personal factors.  
    ⚠️ **Ethical Note:** This is a demo for educational purposes only.  
    It is **not a substitute for professional medical or psychological advice**.  
    If you are struggling, please seek support from a qualified professional. 💙
    """
)

# Collect inputs
st.header("Please answer the following questions:")

# --- Extra optional fields ---
timestamp = st.text_input("Timestamp (optional, leave blank for auto)", "")
comments = st.text_area("Additional Comments (optional)", "")

# --- Main survey fields ---
age = st.number_input("Age", min_value=10, max_value=100, step=1)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])

country = st.text_input("Country", "India")

state = st.text_input("State (if in USA, else leave blank)", "")

self_employed = st.selectbox("Are you self-employed?", ["Yes", "No"])

family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])

work_interfere = st.selectbox(
    "If you have a mental health condition, does it interfere with your work?",
    ["Never", "Rarely", "Sometimes", "Often"]
)

no_employees = st.selectbox(
    "Number of employees at your company?",
    ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
)

remote_work = st.selectbox("Do you work remotely (at least 50% of the time)?", ["Yes", "No"])

tech_company = st.selectbox("Is your employer primarily a tech company?", ["Yes", "No"])

benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])

care_options = st.selectbox(
    "Do you know the options for mental health care your employer provides?",
    ["Yes", "No", "Not sure"]
)

wellness_program = st.selectbox(
    "Has your employer ever discussed mental health as part of a wellness program?",
    ["Yes", "No", "Don't know"]
)

seek_help = st.selectbox(
    "Does your employer provide resources to learn about mental health?",
    ["Yes", "No", "Don't know"]
)

anonymity = st.selectbox(
    "Is your anonymity protected if you choose to take advantage of mental health treatment?",
    ["Yes", "No", "Don't know"]
)

leave = st.selectbox(
    "How easy is it for you to take medical leave for a mental health condition?",
    ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
)

mental_health_consequence = st.selectbox(
    "Do you think discussing a mental health issue with your employer would have negative consequences?",
    ["Yes", "No", "Maybe"]
)

phys_health_consequence = st.selectbox(
    "Do you think discussing a physical health issue with your employer would have negative consequences?",
    ["Yes", "No", "Maybe"]
)

coworkers = st.selectbox(
    "Would you be willing to discuss a mental health issue with your coworkers?",
    ["Yes", "No", "Some of them"]
)

supervisor = st.selectbox(
    "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
    ["Yes", "No", "Some of them"]
)

mental_health_interview = st.selectbox(
    "Would you bring up a mental health issue in a job interview?",
    ["Yes", "No", "Maybe"]
)

phys_health_interview = st.selectbox(
    "Would you bring up a physical health issue in a job interview?",
    ["Yes", "No", "Maybe"]
)

mental_vs_physical = st.selectbox(
    "Do you feel that your employer takes mental health as seriously as physical health?",
    ["Yes", "No", "Don't know"]
)

obs_consequence = st.selectbox(
    "Have you heard of or observed negative consequences for coworkers with mental health conditions at your workplace?",
    ["Yes", "No"]
)

# Collect all features in correct order (must match training pipeline)
input_data = pd.DataFrame([{
    "Timestamp": timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Age": age,
    "Gender": gender,
    "Country": country,
    "state": state,
    "self_employed": self_employed,
    "family_history": family_history,
    "treatment": "No",  # placeholder (target variable during training, not used in prediction)
    "work_interfere": work_interfere,
    "no_employees": no_employees,
    "remote_work": remote_work,
    "tech_company": tech_company,
    "benefits": benefits,
    "care_options": care_options,
    "wellness_program": wellness_program,
    "seek_help": seek_help,
    "anonymity": anonymity,
    "leave": leave,
    "mental_health_consequence": mental_health_consequence,
    "phys_health_consequence": phys_health_consequence,
    "coworkers": coworkers,
    "supervisor": supervisor,
    "mental_health_interview": mental_health_interview,
    "phys_health_interview": phys_health_interview,
    "mental_vs_physical": mental_vs_physical,
    "obs_consequence": obs_consequence,
    "comments": comments if comments else "No comments"
}])

if st.button("🔮 Predict"):
    try:
        prediction = pipe.predict(input_data.drop(columns=["treatment"]))[0]
        if prediction == "Yes":
            st.success("✅ The model predicts that the person is **likely to seek treatment**.")
        else:
            st.info("ℹ️ The model predicts that the person is **unlikely to seek treatment**.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.markdown("---")
st.caption("Made by Shivam 💡")
