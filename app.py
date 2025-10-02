import streamlit as st
import pickle
import pandas as pd
import os

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "ipl_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.title("🏏 IPL Match Winner Predictor")

# Teams list
teams = [
    "Sunrisers Hyderabad", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals"
]

# Cities list
cities = [
    "Hyderabad", "Mumbai", "Bangalore", "Kolkata", "Delhi", "Chennai",
    "Jaipur", "Punjab", "Dubai", "Sharjah", "Abu Dhabi"
]

# User inputs
batting_team = st.selectbox("🏏 Select Batting Team", teams)
bowling_team = st.selectbox("🎯 Select Bowling Team", [t for t in teams if t != batting_team])
city = st.selectbox("📍 Match City", cities)

runs_left = st.number_input("Runs Left", min_value=0, max_value=300, value=50)
balls_left = st.number_input("Balls Left", min_value=1, max_value=120, value=60)
wickets = st.number_input("Wickets Left", min_value=0, max_value=10, value=5)
target = st.number_input("Target Runs", min_value=0, max_value=300, value=150)

# Calculate CRR and RRR
current_score = target - runs_left
crr = round((current_score * 6) / (120 - balls_left + 1e-5), 2) if balls_left != 120 else 0
rrr = round((runs_left * 6) / balls_left, 2)

st.write(f"📊 Current Run Rate (CRR): {crr}")
st.write(f"⚡ Required Run Rate (RRR): {rrr}")

# Prediction button
if st.button("Predict Winner"):
    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [wickets],
        "total_runs_x": [target],
        "crr": [crr],
        "rrr": [rrr]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(f"✅ {batting_team} is likely to win!")
    else:
        st.error(f"❌ {bowling_team} is likely to win!")
