import streamlit as st
import pandas as pd
import cloudpickle

# -------------------------------
# Load the trained pipeline safely
# -------------------------------
with open("mh_pipeline.pkl", "rb") as f:
    pipe = cloudpickle.load(f)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🧠 Mental Health Treatment Prediction")

st.markdown("""
This tool predicts whether a person may need **mental health treatment** based on workplace and personal factors.  
⚠️ **Disclaimer:** This is for **educational/demo purposes only** and should not replace professional medical advice.  
If you are struggling, please reach out to a qualified professional or helpline.
""")

# Collect inputs
st.header("Please answer the following questions:")

Timestamp = st.text_input("Timestamp (optional)", value="2023-01-01 00:00:00")
Age = st.number_input("Your Age", min_value=10, max_value=100, value=25)
Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
Country = st.text_input("Country", value="India")
state = st.text_input("State (optional)", value="NA")
self_employed = st.selectbox("Are you self-employed?", ["Yes", "No", "NA"])
family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])
work_interfere = st.selectbox("If you have a mental health condition, does it interfere with work?", 
                              ["Never", "Rarely", "Sometimes", "Often", "NA"])
no_employees = st.selectbox("Number of employees at your company", 
                            ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])
tech_company = st.selectbox("Is your employer a tech company?", ["Yes", "No"])
benefits = st.selectbox("Does your employer provide mental health benefits?", 
                        ["Yes", "No", "Don't know"])
care_options = st.selectbox("Do you know the care options available?", ["Yes", "No", "Not sure"])
wellness_program = st.selectbox("Does your employer provide wellness programs?", 
                                ["Yes", "No", "Don't know"])
seek_help = st.selectbox("Does your employer encourage you to seek help?", 
                         ["Yes", "No", "Don't know"])
anonymity = st.selectbox("Is anonymity protected when seeking help?", 
                         ["Yes", "No", "Don't know"])
leave = st.selectbox("How easy is it to take medical leave for mental health?", 
                     ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
mental_health_consequence = st.selectbox("Consequence of discussing mental health at work", 
                                         ["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Consequence of discussing physical health at work", 
                                       ["Yes", "No", "Maybe"])
coworkers = st.selectbox("Would you discuss mental health with coworkers?", 
                         ["Yes", "No", "Some of them"])
supervisor = st.selectbox("Would you discuss mental health with your supervisor?", 
                          ["Yes", "No", "Some of them"])
mental_health_interview = st.selectbox("Would you discuss mental health in an interview?", 
                                       ["Yes", "No", "Maybe"])
phys_health_interview = st.selectbox("Would you discuss physical health in an interview?", 
                                     ["Yes", "No", "Maybe"])
mental_vs_physical = st.selectbox("Is mental health as important as physical health?", 
                                  ["Yes", "No", "Don't know"])
obs_consequence = st.selectbox("Have you observed negative consequences for discussing mental health?", 
                               ["Yes", "No"])
comments = st.text_area("Additional comments (optional)", value="NA")

# Collect inputs into a DataFrame
input_data = pd.DataFrame([{
    "Timestamp": Timestamp,
    "Age": Age,
    "Gender": Gender,
    "Country": Country,
    "state": state,
    "self_employed": self_employed,
    "family_history": family_history,
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
    "comments": comments
}])

# Prediction
if st.button("🔮 Predict"):
    try:
        prediction = pipe.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ The model predicts that you **may require mental health treatment.**")
        else:
            st.success("✅ The model predicts that you are **less likely to require mental health treatment.**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.markdown("---")
st.markdown("👨‍💻 Made by Shivam")
