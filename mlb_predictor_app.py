import pandas as pd
import streamlit as st
import joblib

# === Load saved model and mappings ===
clf = joblib.load("xgb_model.pkl")
team_map = joblib.load("team_map.pkl")
reverse_map = joblib.load("reverse_map.pkl")

# === Streamlit UI ===
st.title("⚾ MLB Game Winner Predictor (XGBoost Model - Pretrained)")
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




# === App version tag ===
version = "v1.1 - XGBoost upgrade (cached)"
last_updated = "2025-05-14"
st.markdown("---")
st.caption(f"🔢 App Version: **{version}**  |  🕒 Last Updated: {last_updated}")

