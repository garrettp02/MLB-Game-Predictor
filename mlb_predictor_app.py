import pandas as pd
import streamlit as st
import joblib
import feedparser

# === Load model and mappings ===
clf = joblib.load("xgb_model.pkl")
team_map = joblib.load("team_map.pkl")
reverse_map = joblib.load("reverse_map.pkl")

# === Team logos map ===
team_logos = {
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "CHC": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "CWS": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "KC":  "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "LAD": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "NYM": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "NYY": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "SD":  "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "SF":  "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "TB":  "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "WSH": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png"
}

coldwire_slugs = {
    "NYY": "new-york-yankees", "BOS": "boston-red-sox", "LAD": "los-angeles-dodgers", "SF": "san-francisco-giants",
    "HOU": "houston-astros", "CHC": "chicago-cubs", "ATL": "atlanta-braves", "BAL": "baltimore-orioles",
    "TEX": "texas-rangers", "PHI": "philadelphia-phillies", "ARI": "arizona-diamondbacks", "SEA": "seattle-mariners",
    "TOR": "toronto-blue-jays", "MIN": "minnesota-twins", "MIA": "miami-marlins", "CIN": "cincinnati-reds",
    "MIL": "milwaukee-brewers", "DET": "detroit-tigers", "CLE": "cleveland-guardians", "OAK": "oakland-athletics",
    "SD": "san-diego-padres", "PIT": "pittsburgh-pirates", "NYM": "new-york-mets", "STL": "st-louis-cardinals",
    "WSH": "washington-nationals", "CWS": "chicago-white-sox", "LAA": "los-angeles-angels", "COL": "colorado-rockies",
    "KC": "kansas-city-royals", "TB": "tampa-bay-rays"
}

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
    st.title("📰 MLB Team News Feed")
    selected_team = st.selectbox("Choose a team:", sorted(team_map.keys()))

    if selected_team in team_logos:
        st.image(team_logos[selected_team], width=150)

    st.subheader(f"Latest news about {selected_team}")

    # Try ColdWire first
    feed_url = None
    slug = coldwire_slugs.get(selected_team)
    if slug:
        feed_url = f"https://mlbnewsnow.com/tag/{slug}/feed/"
    else:
        st.info("ColdWire feed not found. Using ESPN backup.")
        feed_url = "https://www.espn.com/espn/rss/mlb/news"

    feed = feedparser.parse(feed_url)
    found = False
    for entry in feed.entries[:10]:
        if selected_team.lower() in entry.title.lower() or selected_team.lower() in entry.summary.lower():
            st.markdown(f"**[{entry.title}]({entry.link})**")
            st.caption(entry.published)
            found = True

    if not found:
        st.info(f"No recent news found for {selected_team}")

# === Footer ===
st.markdown("---")
version = "v2.2 - ColdWire News + Logos"
last_updated = "2025-05-14"
st.caption(f"🔢 App Version: **{version}**  |  🕒 Last Updated: {last_updated}")
