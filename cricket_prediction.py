import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
match = pd.read_csv("matches.csv")
delivery = pd.read_csv("deliveries.csv")

# First innings total runs
total_score_df = (
    delivery.groupby(["match_id", "inning"])
    .sum()["total_runs"]
    .reset_index()
)
total_score_df = total_score_df[total_score_df["inning"] == 1]

match_df = match.merge(
    total_score_df[["match_id", "total_runs"]],
    left_on="id",
    right_on="match_id"
)

# Standardize team names
teams = [
    "Sunrisers Hyderabad", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals"
]

match_df["team1"] = match_df["team1"].replace({
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad"
})
match_df["team2"] = match_df["team2"].replace({
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad"
})

match_df = match_df[
    match_df["team1"].isin(teams) &
    match_df["team2"].isin(teams) &
    (match_df["dl_applied"] == 0)
]

match_df = match_df[["match_id", "city", "winner", "total_runs"]]
delivery_df = match_df.merge(delivery, on="match_id")
delivery_df = delivery_df[delivery_df["inning"] == 2]

# Calculate runs left, balls left, wickets, CRR, RRR
delivery_df["total_runs_y"] = pd.to_numeric(delivery_df["total_runs_y"], errors="coerce")
delivery_df["current_score"] = delivery_df.groupby("match_id")["total_runs_y"].cumsum()
delivery_df["runs_left"] = delivery_df["total_runs_x"] - delivery_df["current_score"]
delivery_df["balls_left"] = 126 - (delivery_df["over"] * 6) + delivery_df["ball"]

delivery_df["player_dismissed"] = delivery_df["player_dismissed"].fillna("0")
delivery_df["player_dismissed"] = delivery_df["player_dismissed"].apply(lambda x: 0 if x == "0" else 1).astype(int)
wickets = delivery_df.groupby("match_id")["player_dismissed"].cumsum().values
delivery_df["wickets"] = 10 - wickets

delivery_df["crr"] = (delivery_df["current_score"] * 6) / (120 - delivery_df["balls_left"])
delivery_df["rrr"] = (delivery_df["runs_left"] * 6) / delivery_df["balls_left"]

# Result column
delivery_df["result"] = (delivery_df["batting_team"] == delivery_df["winner"]).astype(int)

# Final dataset
final_df = delivery_df[[
    "batting_team", "bowling_team", "city",
    "runs_left", "balls_left", "wickets",
    "total_runs_x", "crr", "rrr", "result"
]]

final_df = final_df.sample(frac=1).dropna()
final_df = final_df[final_df["balls_left"] != 0]

x = final_df.drop("result", axis=1)
y = final_df["result"]

x.replace([np.inf, -np.inf], np.nan, inplace=True)
x.dropna(inplace=True)
y = y[x.index]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Pipeline
trf = ColumnTransformer([
    ("trf", OneHotEncoder(sparse_output=False, drop="first"),
     ["batting_team", "bowling_team", "city"])
], remainder="passthrough")

pipe = Pipeline([
    ("step1", trf),
    ("step2", RandomForestClassifier())
])

# Train model
pipe.fit(x_train, y_train)

# Evaluate
y_pred = pipe.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model
pickle.dump(pipe, open("ipl_model.pkl", "wb"))
