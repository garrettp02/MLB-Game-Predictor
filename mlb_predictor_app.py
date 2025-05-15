import pandas as pd
import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup

# === Load model and mappings ===
clf = joblib.load("xgb_model.pkl")
team_map = joblib.load("team_map.pkl")
reverse_map = joblib.load("reverse_map.pkl")

# === Sidebar Navigation ===
st.sidebar.title("MLB Predictor Navigation")
page = st.sidebar.radio("Go to", ["Single Game Prediction", "Batch Predictions", "Team News Feeds"])

# === Single Game Prediction ===
if page == "Single Game Prediction":
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

# === Batch Predictions ===
elif page == "Batch Predictions":
    st.title("📂 Batch Game Predictions")
    uploaded = st.file_uploader("Upload a CSV with matchups (columns: home_team, away_team)", type="csv")

    if uploaded:
        matchups = pd.read_csv(uploaded)
        results = []
        for _, row in matchups.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            if home in team_map and away in team_map:
                input_df = pd.DataFrame([[team_map[home], team_map[away]]], columns=["home_id", "away_id"])
                probs = clf.predict_proba(input_df)[0]
                class_ids = clf.classes_.tolist()
                selected = {tid: probs[class_ids.index(tid)] for tid in [team_map[home], team_map[away]]}
                winner_id = max(selected, key=selected.get)
                results.append({
                    "Home": home,
                    "Away": away,
                    "Predicted Winner": reverse_map[winner_id],
                    "Confidence": round(abs(selected[team_map[home]] - selected[team_map[away]]), 4)
                })
            else:
                results.append({"Home": home, "Away": away, "Predicted Winner": "Invalid Team", "Confidence": 0})

        result_df = pd.DataFrame(results)
        st.dataframe(result_df)

# === Live News Feeds ===
elif page == "Team News Feeds":
    st.title("📰 MLB Team News Feeds")
    selected_team = st.selectbox("Choose a team to fetch recent news:", sorted(team_map.keys()))

    query = f"{selected_team} MLB news site:espn.com"
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        headlines = soup.select("h3")[:5]

        st.subheader(f"Top News for {selected_team}")
        for h in headlines:
            st.markdown(f"- {h.text}")
    except Exception as e:
        st.error(f"Error fetching news: {e}")

# === Footer ===
st.markdown("---")
version = "v2.0 - Full Dashboard Upgrade"
last_updated = "2025-05-14"
st.caption(f"🔢 App Version: **{version}**  |  🕒 Last Updated: {last_updated}")

