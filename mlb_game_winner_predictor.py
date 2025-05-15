import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Load the dataset ===
file_path = "games.csv"
df = pd.read_csv(file_path)

# === Clean and prepare data ===
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

# Train/test split
X = df[["home_id", "away_id"]]
y = df["winner_id"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# === Train both models ===
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)

xgb_clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
xgb_clf.fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)

# === Print performance in console ===
print("📊 Random Forest:")
print(classification_report(y_test, rf_preds, zero_division=0))
print("Accuracy:", accuracy_score(y_test, rf_preds))

print("\n📊 XGBoost:")
print(classification_report(y_test, xgb_preds, zero_division=0))
print("Accuracy:", accuracy_score(y_test, xgb_preds))

# === Use XGBoost for live predictions ===
clf = xgb_clf

# === Streamlit UI ===
st.title("⚾ MLB Game Winner Predictor (XGBoost Model)")
st.write("Select two teams to predict the winner based on historical data.")

home_team = st.selectbox("Home Team", sorted(team_map.keys()))
away_team = st.selectbox("Away Team", sorted(team_map.keys()))

if st.button("Predict Winner"):
    if home_team not in team_map or away_team not in team_map:
        st.error("One or both teams not found in training data.")
    else:
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
            st.error("Neither team is in training data.")
        elif len(selected) == 1:
            only_team = reverse_map[list(selected.keys())[0]]
            st.warning(f"Only one team was in training data. Default winner: {only_team}")
        else:
            winner_id = max(selected, key=selected.get)
            predicted_winner = reverse_map[winner_id]
            prob_margin = abs(selected[home_id] - selected[away_id])
            st.success(f"🏆 Predicted Winner: {predicted_winner}")
             st.caption(f"📊 Confidence margin: {prob_margin:.2f}")

import datetime
version = "v1.1 - XGBoost upgrade"
last_updated = "2025-05-14"

st.markdown("---")
st.caption(f"🔢 App Version: **{version}**  |  🕒 Last Updated: {last_updated}")
