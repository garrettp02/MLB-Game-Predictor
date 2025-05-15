import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Load and prepare dataset ===
file_path = "/Users/garrettparr/Com Sci C++ Home/MLB Decision Tree/games.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=["home", "away", "home-score", "away-score"])

# Create winner column
def determine_winner(row):
    if row["home-score"] > row["away-score"]:
        return row["home"]
    elif row["home-score"] < row["away-score"]:
        return row["away"]
    else:
        return "TIE"

df["winner"] = df.apply(determine_winner, axis=1)
df = df[df["winner"] != "TIE"]

# Encode team names
teams = pd.unique(df[["home", "away"]].values.ravel())
team_map = {team: i for i, team in enumerate(teams)}
reverse_map = {v: k for k, v in team_map.items()}

df["home_id"] = df["home"].map(team_map)
df["away_id"] = df["away"].map(team_map)
df["winner_id"] = df["winner"].map(team_map)

# Train the model
X = df[["home_id", "away_id"]]
y = df["winner_id"]
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
clf.fit(X, y)

# === Streamlit UI ===
st.title("⚾ MLB Game Winner Predictor")
st.write("Select two teams and get a predicted winner based on historical data.")

home_team = st.selectbox("Home Team", sorted(team_map.keys()))
away_team = st.selectbox("Away Team", sorted(team_map.keys()))

if st.button("Predict Winner"):
    home_id = team_map[home_team]
    away_id = team_map[away_team]
    input_df = pd.DataFrame([[home_id, away_id]], columns=["home_id", "away_id"])
    probs = clf.predict_proba(input_df)[0]
    class_ids = clf.classes_.tolist()

    selected = {}
    for team_id in [home_id, away_id]:
        if team_id in class_ids:
            selected[team_id] = probs[class_ids.index(team_id)]

    if not selected:
        st.error("Neither team is in the training data.")
    elif len(selected) == 1:
        winner = reverse_map[list(selected.keys())[0]]
        st.warning(f"Only one team was seen in training data. Default winner: {winner}")
    else:
        winner_id = max(selected, key=selected.get)
        winner = reverse_map[winner_id]
        prob_diff = abs(selected[home_id] - selected[away_id])
        st.success(f"🏆 Predicted Winner: {winner}")
        st.caption(f"Confidence margin: {prob_diff:.2f}")
