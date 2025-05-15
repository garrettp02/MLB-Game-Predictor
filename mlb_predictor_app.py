import pandas as pd
import streamlit as st
import joblib
import feedparser
import requests
import datetime

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
    "CHW": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",
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

mlb_team_ids = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CIN": 113, "CLE": 114, "COL": 115, "CHW": 145, "DET": 116,
    "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120
}

filtered_team_keys = [key for key in sorted(team_map.keys()) if key not in ("AL", "NL")]

st.sidebar.title("MLB Predictor Navigation")
page = st.sidebar.radio("Go to", ["Single Game Prediction", "Batch Predictions", "Team News Feeds", "Daily Matchups"])

# === Single Game Prediction ===
if page == "Single Game Prediction":
    st.title("⚾ MLB Game Winner Predictor (XGBoost Model - Pretrained)")
    st.write("Select two teams to predict the winner based on historical data.")

    home_team = st.selectbox("Home Team", filtered_team_keys)
    away_team = st.selectbox("Away Team", filtered_team_keys)


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

# === Daily Matchups ===
if page == "Daily Matchups":
    st.title("📅 Today's MLB Matchups & Predictions")
    today = datetime.date.today()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    response = requests.get(url)
    data = response.json()

    games = data.get("dates", [])
    if not games:
        st.info("No games scheduled for today.")
    else:
        matchups = []
        for game in games[0]["games"]:
            home_team = game["teams"]["home"]["team"]["abbreviation"]
            away_team = game["teams"]["away"]["team"]["abbreviation"]

            if home_team in team_map and away_team in team_map:
                home_id = team_map[home_team]
                away_id = team_map[away_team]
                input_df = pd.DataFrame([[home_id, away_id]], columns=["home_id", "away_id"])
                probs = clf.predict_proba(input_df)[0]
                class_ids = clf.classes_.tolist()

                selected = {tid: probs[class_ids.index(tid)] for tid in [home_id, away_id] if tid in class_ids}
                if selected:
                    winner_id = max(selected, key=selected.get)
                    predicted = reverse_map[winner_id]
                    margin = abs(selected[home_id] - selected[away_id])
                else:
                    predicted = "Unavailable"
                    margin = 0

                matchups.append({
                    "Away": away_team,
                    "Home": home_team,
                    "Predicted Winner": predicted,
                    "Confidence": round(margin, 3)
                })

        st.dataframe(pd.DataFrame(matchups))

# === Live News Feeds ===
elif page == "Team News Feeds":
    st.title("📰 MLB Team News Feed")

    team_display_names = {
        **{k: f"{k} - {v}" for k, v in {
            "ARI": "D-backs", "ATL": "Braves", "BAL": "Orioles", "BOS": "Red Sox",
            "CHC": "Cubs", "CIN": "Reds", "CLE": "Guardians", "COL": "Rockies",
            "CHW": "White Sox", "DET": "Tigers", "HOU": "Astros", "KC": "Royals",
            "LAA": "Angels", "LAD": "Dodgers", "MIA": "Marlins", "MIL": "Brewers",
            "MIN": "Twins", "NYM": "Mets", "NYY": "Yankees", "OAK": "Athletics",
            "PHI": "Phillies", "PIT": "Pirates", "SD": "Padres", "SEA": "Mariners",
            "SF": "Giants", "STL": "Cardinals", "TB": "Rays", "TEX": "Rangers",
            "TOR": "Blue Jays", "WSH": "Nationals"
        }.items()},
        "AL": "American League", "NL": "National League"
    }

    selected_label = st.selectbox("Choose a team or league:", list(team_display_names.values()))
    selected_team = next(k for k, v in team_display_names.items() if v == selected_label)

    if selected_team in team_logos:
        st.image(team_logos[selected_team], width=150)

    st.subheader(f"Latest news about {team_display_names[selected_team]}")

    team_name_map = {
        "ARI": "dbacks", "ATL": "braves", "BAL": "orioles", "BOS": "redsox",
        "CHC": "cubs", "CIN": "reds", "CLE": "guardians", "COL": "rockies",
        "CHW": "whitesox", "DET": "tigers", "HOU": "astros", "KC": "royals",
        "LAA": "angels", "LAD": "dodgers", "MIA": "marlins", "MIL": "brewers",
        "MIN": "twins", "NYM": "mets", "NYY": "yankees", "OAK": "athletics",
        "PHI": "phillies", "PIT": "pirates", "SD": "padres", "SEA": "mariners",
        "SF": "giants", "STL": "cardinals", "TB": "rays", "TEX": "rangers",
        "TOR": "bluejays", "WSH": "nationals"
    }

    if selected_team in ("AL", "NL"):
        feed_url = "https://www.espn.com/espn/rss/mlb/news"
    else:
        team_name = team_name_map.get(selected_team, selected_team.lower())
        feed_url = f"https://www.mlb.com/{team_name}/feeds/news/rss.xml"

    feed = feedparser.parse(feed_url)
    if not feed.entries:
        st.warning("No recent news found or feed unavailable.")
    else:
        for entry in feed.entries[:3]:
            st.markdown(f"**[{entry.title}]({entry.link})**")
            if hasattr(entry, "summary"):
                st.write(entry.summary)
            if hasattr(entry, "media_content"):
                for media in entry.media_content:
                    if media.get("medium") == "image" and "url" in media:
                        st.image(media["url"], width=250)
            st.caption(entry.published)
            st.markdown("---")

    # === Show upcoming games ===
    if selected_team in mlb_team_ids:
        st.subheader("📅 Upcoming Schedule")
        team_id = mlb_team_ids.get(selected_team)
        today = datetime.date.today()
        end = today + datetime.timedelta(days=14)
        url = f"https://statsapi.mlb.com/api/v1/schedule?teamId={team_id}&sportId=1&startDate={today}&endDate={end}"
        response = requests.get(url)
        data = response.json()

        games = data.get("dates", [])
        if not games:
            st.info("No upcoming games found.")
        else:
            for day in games[:5]:
                for game in day["games"]:
                    opponent = game["teams"]["away"]["team"] if game["teams"]["home"]["team"]["id"] == team_id else game["teams"]["home"]["team"]
                    game_date = game["gameDate"]
                    venue = game["venue"]["name"]
                    home_team = game["teams"]["home"]["team"]["name"]
                    away_team = game["teams"]["away"]["team"]["name"]
                    st.markdown(f"**{away_team} @ {home_team}** — {game_date[:10]} at {venue}")
    else:
        st.info("Schedule not available for league-wide selections.")

# === Footer ===
st.markdown("---")
version = "v2.5 - News & Schedule Integration"
last_updated = "2025-05-15"
st.caption(f"🔢 App Version: **{version}**  |  🕒 Last Updated: {last_updated}")

