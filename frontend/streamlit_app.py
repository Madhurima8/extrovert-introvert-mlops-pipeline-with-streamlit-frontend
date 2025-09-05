import streamlit as st
import requests

st.title("Extrovert vs Introvert Classifier")
st.markdown("Fill in the following details to predict personality type.")

# Input fields (exact column names from schema)
payload = {
    "Time_spent_Alone": st.selectbox("Hours spent alone per day", [str(i) for i in range(12)]),
    "Stage_fear": st.selectbox("Do you have stage fear?", ["Yes", "No"]),
    "Social_event_attendance": st.selectbox("Social events attended per week", [str(i) for i in range(11)]),
    "Going_outside": st.selectbox("Days you go outside per week", [str(i) for i in range(8)]),
    "Drained_after_socializing": st.selectbox("Feel drained after socializing?", ["Yes", "No"]),
    "Friends_circle_size": st.selectbox("Number of close friends", [str(i) for i in range(16)]),
    "Post_frequency": st.selectbox("Social media post frequency", [str(i) for i in range(11)])
}

if st.button("Predict"):
    try:
        # Send POST request to FastAPI backend
        response = requests.post("http://extrovert_backend:8080/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Personality: **{result['predicted_label'].capitalize()}**")
            st.write("Prediction Confidence (if available):", result.get("predicted_proba"))
        else:
            st.error(f"Prediction failed with status code {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")