#Newest With Sliders

import pandas as pd
import streamlit as st
import joblib
import feedparser
import requests
import datetime
import matplotlib.pyplot as plt

# === Load model and mappings ===
clf = joblib.load("xgb_model_updated.pkl")
team_map = joblib.load("team_map_updated.pkl")
reverse_map = joblib.load("reverse_map_updated.pkl")

# === Load API data ===
@st.cache_data(ttl=3600)
def get_last_10_game_stats(team_id):
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?teamId={team_id}&season=2025&sportId=1&gameType=R"
    schedule_data = requests.get(schedule_url).json()

    games = []
    for date in schedule_data.get("dates", []):
        for game in date.get("games", []):
            if game["status"]["abstractGameState"] == "Final":
                games.append(game["gamePk"])

    games = sorted(games, reverse=True)[:10]

    total_bases_list = []
    walks_issued_list = []
    strikeouts_thrown_list = []

    for game_pk in games:
        boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        box_data = requests.get(boxscore_url).json()

        teams = box_data.get("teams", {})
        for side in ["home", "away"]:
            team = teams[side].get("team", {})
            if team["id"] == team_id:
                stats = teams[side].get("teamStats", {})
                batting_stats = stats.get("batting", {})
                pitching_stats = stats.get("pitching", {})

                total_bases = batting_stats.get("totalBases", 0)
                walks_issued = pitching_stats.get("baseOnBalls", 0)
                strikeouts_thrown = pitching_stats.get("strikeOuts", 0)

                total_bases_list.append(total_bases)
                walks_issued_list.append(walks_issued)
                strikeouts_thrown_list.append(strikeouts_thrown)
                break

    if len(total_bases_list) == 0:
        return None

    return {
        "total_bases": round(sum(total_bases_list) / len(total_bases_list), 2),
        "walks_issued": round(sum(walks_issued_list) / len(walks_issued_list), 2),
        "strikeouts_thrown": round(sum(strikeouts_thrown_list) / len(strikeouts_thrown_list), 2)
    }

@st.cache_data(ttl=3600)
def get_team_win_pct(team_abbr):
    team_id = mlb_team_ids[team_abbr]
    standings_url = "https://statsapi.mlb.com/api/v1/standings?season=2025&leagueId=103,104&standingsTypes=regularSeason"
    standings_data = requests.get(standings_url).json()

    for record in standings_data.get("records", []):
        for tr in record.get("teamRecords", []):
            if tr["team"]["id"] == team_id:
                w, l = tr["wins"], tr["losses"]
                return round(w / (w + l), 3) if (w + l) > 0 else 0.5
    return 0.5


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
id_to_abbr = {v: k for k, v in mlb_team_ids.items()}

filtered_team_keys = [key for key in sorted(team_map.keys()) if key not in ("AL", "NL")]

st.sidebar.title("MLB Predictor Navigation")
page = st.sidebar.radio("Go to", ["Daily Matchups", "Single Game Prediction", "Team News Feeds", "10-Game Averages"])

# === Single Game Prediction ===
if page == "Single Game Prediction":
    st.title("⚾ MLB Game Winner Predictor (XGBoost Model - Pretrained)")
    st.write("Select two teams to predict the winner based on historical matchup patterns.")

    home_team = st.selectbox("Home Team", filtered_team_keys)
    away_team = st.selectbox("Away Team", filtered_team_keys)

    customize = st.checkbox("🔧 Customize game stats with sliders?")

    if customize:
        st.markdown("#### 📈 Adjust Team Performance Inputs")
        col1, col2 = st.columns(2)

        with col1:
            home_win_slider = st.slider("Home Win %", 0.0, 1.0, 0.55, step=0.01, key="home_win_slider")
            home_win_pct = st.number_input("Or type value for Home Win %", min_value=0.0, max_value=1.0, value=home_win_slider, step=0.01, key="home_win_input")

            walks_home_slider = st.slider("Walks Issued (Home)", 0.0, 10.0, 3.1, step=0.1, key="walks_home_slider")
            walks_home = st.number_input("Or type value for Walks Issued (Home)", min_value=0.0, max_value=10.0, value=walks_home_slider, step=0.1, key="walks_home_input")

            k_home_slider = st.slider("Strikeouts Thrown (Home)", 0.0, 15.0, 8.9, step=0.1, key="k_home_slider")
            k_home = st.number_input("Or type value for Strikeouts Thrown (Home)", min_value=0.0, max_value=15.0, value=k_home_slider, step=0.1, key="k_home_input")

            tb_home_slider = st.slider("Total Bases (Home)", 0.0, 20.0, 12.3, step=0.1, key="tb_home_slider")
            tb_home = st.number_input("Or type value for Total Bases (Home)", min_value=0.0, max_value=20.0, value=tb_home_slider, step=0.1, key="tb_home_input")

        with col2:
            away_win_slider = st.slider("Away Win %", 0.0, 1.0, 0.48, step=0.01, key="away_win_slider")
            away_win_pct = st.number_input("Or type value for Away Win %", min_value=0.0, max_value=1.0, value=away_win_slider, step=0.01, key="away_win_input")

            walks_away_slider = st.slider("Walks Issued (Away)", 0.0, 10.0, 2.8, step=0.1, key="walks_away_slider")
            walks_away = st.number_input("Or type value for Walks Issued (Away)", min_value=0.0, max_value=10.0, value=walks_away_slider, step=0.1, key="walks_away_input")

            k_away_slider = st.slider("Strikeouts Thrown (Away)", 0.0, 15.0, 9.1, step=0.1, key="k_away_slider")
            k_away = st.number_input("Or type value for Strikeouts Thrown (Away)", min_value=0.0, max_value=15.0, value=k_away_slider, step=0.1, key="k_away_input")

            tb_away_slider = st.slider("Total Bases (Away)", 0.0, 20.0, 11.5, step=0.1, key="tb_away_slider")
            tb_away = st.number_input("Or type value for Total Bases (Away)", min_value=0.0, max_value=20.0, value=tb_away_slider, step=0.1, key="tb_away_input")

    else:
        # Default average stats
        home_win_pct = 0.55
        away_win_pct = 0.48
        walks_home = 3.1
        walks_away = 2.8
        k_home = 8.9
        k_away = 9.1
        tb_home = 12.3
        tb_away = 11.5

    # === Display Charts ===

    st.markdown("### 📊 Live 10-Game Stats for Selected Teams")
    with st.spinner("Fetching stats from MLB API..."):
        home_stats = get_last_10_game_stats(mlb_team_ids[home_team])
        away_stats = get_last_10_game_stats(mlb_team_ids[away_team])
        home_win = get_team_win_pct(home_team)
        away_win = get_team_win_pct(away_team)

        if home_stats and away_stats:
            display_df = pd.DataFrame({
                "Stat": ["Total Bases", "Walks Issued", "Strikeouts Thrown", "Win %"],
                home_team: [
                    home_stats["total_bases"],
                    home_stats["walks_issued"],
                    home_stats["strikeouts_thrown"],
                    home_win
                ],
                away_team: [
                    away_stats["total_bases"],
                    away_stats["walks_issued"],
                    away_stats["strikeouts_thrown"],
                    away_win
                ]
            }).set_index("Stat")

            st.dataframe(display_df.style.format("{:.3f}"), use_container_width=True)
        else:
            st.warning("Could not retrieve stats for one or both teams.")


    if st.button("Predict Winner"):
        if home_team not in team_map or away_team not in team_map:
            st.error("One or both teams not found in training data.")
        else:
            home_id = team_map[home_team]
            away_id = team_map[away_team]

            input_df = pd.DataFrame([[
                home_id, away_id,
                home_win_pct, away_win_pct,
                walks_home, walks_away,
                k_home, k_away,
                tb_home, tb_away
            ]], columns=[
                "home_id", "away_id",
                "home_win_pct", "away_win_pct",
                "Walks Issued - Home", "Walks Issued - Away",
                "Strikeouts Thrown - Home", "Strikeouts Thrown - Away",
                "Total Bases - Home", "Total Bases - Away"
            ])

            probs = clf.predict_proba(input_df)[0]
            class_ids = clf.classes_.tolist()

            selected = {tid: probs[class_ids.index(tid)] for tid in [home_id, away_id] if tid in class_ids}
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
                st.caption(f"📊 Confidence margin: {prob_margin:.2%}")



# === Daily Matchups ===

if page == "Daily Matchups":
    st.title("📅 Today's MLB Matchups & Predictions")
    today = datetime.date.today()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    response = requests.get(url)
    data = response.json()

    id_to_abbr = {v: k for k, v in mlb_team_ids.items()}
    games = data.get("dates", [])

    team_subreddits = {
        "NYY": "NYYankees", "BOS": "RedSox", "LAD": "Dodgers", "CHC": "CHICubs",
        "SF": "SFGiants", "ATL": "Braves", "NYM": "NewYorkMets", "PHI": "Phillies",
        "SD": "Padres", "HOU": "Astros", "BAL": "Orioles", "SEA": "Mariners",
        "STL": "Cardinals", "TOR": "Torontobluejays", "CIN": "Reds", "TEX": "TexasRangers",
        "OAK": "OaklandAthletics", "WSH": "Nationals", "MIL": "Brewers",
        "MIN": "MinnesotaTwins", "PIT": "Pirates", "CLE": "ClevelandGuardians",
        "DET": "MotorCityKitties", "LAA": "AngelsBaseball", "MIA": "MiamiMarlins",
        "KC": "Royals", "ARI": "AZDiamondbacks", "COL": "ColoradoRockies",
        "TB": "TampaBayRays", "CWS": "whitesox"
    }

    def display_top_reddit_post(team_abbr):
        if team_abbr in team_subreddits:
            subreddit = team_subreddits[team_abbr]
            feed_url = f"https://www.reddit.com/r/{subreddit}/.rss"
            feed = feedparser.parse(feed_url)
            st.subheader(f"📣 Reddit - Top Post from r/{subreddit}")

            for entry in feed.entries:
                if 'Game Thread' in entry.title or 'Post Game Thread' in entry.title or 'Pre Game Thread' in entry.title:
                    st.markdown(f"**[{entry.title}]({entry.link})**")
                    st.caption(entry.published)
                    return

            if feed.entries:
                entry = feed.entries[0]
                st.markdown(f"**[{entry.title}]({entry.link})**")
                st.caption(entry.published)
            else:
                st.info("No recent Reddit posts found.")
        else:
            st.info(f"No subreddit found for {team_abbr}.")

    if not games:
        st.info("No games scheduled for today.")
    else:
        matchups = []
        for game in games[0]["games"]:
            home_id_raw = game["teams"]["home"]["team"]["id"]
            away_id_raw = game["teams"]["away"]["team"]["id"]
            home_team = id_to_abbr.get(home_id_raw)
            away_team = id_to_abbr.get(away_id_raw)

            if home_team in team_map and away_team in team_map:
                home_id = team_map[home_team]
                away_id = team_map[away_team]
            
            # Pull live API stats
            if home_team is not None and away_team is not None and home_team in mlb_team_ids and away_team in mlb_team_ids:
                home_stats = get_last_10_game_stats(mlb_team_ids[home_team])
                away_stats = get_last_10_game_stats(mlb_team_ids[away_team])
                home_win_pct = get_team_win_pct(home_team)
                away_win_pct = get_team_win_pct(away_team)
            else:
                home_stats = None
                away_stats = None
                home_win_pct = 0.5
                away_win_pct = 0.5

            # Use fallback if API fails
            if home_stats and away_stats:
                walks_home = home_stats["walks_issued"]
                walks_away = away_stats["walks_issued"]
                k_home = home_stats["strikeouts_thrown"]
                k_away = away_stats["strikeouts_thrown"]
                tb_home = home_stats["total_bases"]
                tb_away = away_stats["total_bases"]
            else:
                home_win_pct = 0.50
                away_win_pct = 0.50
                walks_home = 3.0
                walks_away = 3.0
                k_home = 8.0
                k_away = 8.0
                tb_home = 12.0
                tb_away = 12.0

            # Create input dataframe with real or fallback values
            input_df = pd.DataFrame([[
                home_id, away_id,
                home_win_pct, away_win_pct,
                walks_home, walks_away,
                k_home, k_away,
                tb_home, tb_away
            ]], columns=[
                "home_id", "away_id",
                "home_win_pct", "away_win_pct",
                "Walks Issued - Home", "Walks Issued - Away",
                "Strikeouts Thrown - Home", "Strikeouts Thrown - Away",
                "Total Bases - Home", "Total Bases - Away"
            ])

                # Average placeholder values used for new model
                #input_df = pd.DataFrame([[
                 #   home_id, away_id,
                  #  0.50, 0.50,
                   # 3.0, 3.0,
                    #8.0, 8.0,
                    #12.0, 12.0
                #]], columns=[
               #     "home_id", "away_id",
                #    "home_win_pct", "away_win_pct",
                 #   "Walks Issued - Home", "Walks Issued - Away",
                  #  "Strikeouts Thrown - Home", "Strikeouts Thrown - Away",
                   # "Total Bases - Home", "Total Bases - Away"
                #])

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
                "Confidence": round(margin, 3),
                "Home Win %": round(selected.get(home_id, 0) * 100, 1),
                "Away Win %": round(selected.get(away_id, 0) * 100, 1)
            })

        view_mode = st.radio("View Mode", ["View All Matchups", "Detailed Matchup View"], horizontal=True)

        if view_mode == "View All Matchups":
            st.dataframe(pd.DataFrame(matchups))
            df = pd.DataFrame(matchups)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(df["Home"] + " vs " + df["Away"], df["Confidence"], color="skyblue")
            ax.set_ylabel("Confidence")
            ax.set_title("Prediction Confidence for Today's Matchups")
            ax.set_xticklabels(df["Home"] + " vs " + df["Away"], rotation=45, ha='right')
            st.pyplot(fig)

        elif view_mode == "Detailed Matchup View":
            selected_matchup = st.selectbox("Select a Matchup", matchups, format_func=lambda x: f"{x['Away']} @ {x['Home']}")
            home_team = selected_matchup["Home"]
            away_team = selected_matchup["Away"]

            # Display matchup logos and teams
            if home_team in team_logos and away_team in team_logos:
                colA, colAt, colH = st.columns([1.5, 1, 1.5])
                with colA:
                    st.image(team_logos[away_team], width=100)
                    st.caption(f"{away_team}")
                with colAt:
                    st.markdown("<h3 style='text-align: center;'>@</h3>", unsafe_allow_html=True)
                with colH:
                    st.image(team_logos[home_team], width=100)
                    st.caption(f"{home_team}")



            st.markdown(f"### Predicted Winner: **{selected_matchup['Predicted Winner']}**")
            st.write(f"**Home Win Probability:** {selected_matchup['Home Win %']}%")
            st.write(f"**Away Win Probability:** {selected_matchup['Away Win %']}%")
            st.write(f"**Confidence Margin:** {selected_matchup['Confidence']}")

            def display_team_news(team_abbr):
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
                team_name = team_name_map.get(team_abbr, team_abbr.lower())
                feed_url = f"https://www.mlb.com/{team_name}/feeds/news/rss.xml"
                feed = feedparser.parse(feed_url)
                st.subheader(f"🗞️ News for {team_abbr}")
                if not feed.entries:
                    st.warning("No recent news found or feed unavailable.")
                else:
                    for entry in feed.entries[:3]:
                        st.markdown(f"**[{entry.title}]({entry.link})**")
                        if hasattr(entry, "summary"):
                            st.write(entry.summary)
                        st.caption(entry.published)
                        st.markdown("---")

            display_team_news(selected_matchup["Home"])
            display_top_reddit_post(selected_matchup["Home"])

            display_team_news(selected_matchup["Away"])
            display_top_reddit_post(selected_matchup["Away"])

# === 10-Game Averages ===
elif page == "10-Game Averages":
    st.title("📊 10-Game Simple Moving Averages (Live)")

    mlb_team_ids = {
        "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
        "CIN": 113, "CLE": 114, "COL": 115, "CHW": 145, "DET": 116,
        "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
        "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
        "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136, "SF": 137,
        "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120
    }

    @st.cache_data(ttl=3600)
    def get_last_10_game_stats(team_id):
        schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?teamId={team_id}&season=2025&sportId=1&gameType=R"
        schedule_data = requests.get(schedule_url).json()

        games = []
        for date in schedule_data.get("dates", []):
            for game in date.get("games", []):
                if game["status"]["abstractGameState"] == "Final":
                    games.append(game["gamePk"])

        games = sorted(games, reverse=True)[:10]

        total_bases_list = []
        walks_issued_list = []
        strikeouts_thrown_list = []

        for game_pk in games:
            boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
            box_data = requests.get(boxscore_url).json()

            teams = box_data.get("teams", {})
            for side in ["home", "away"]:
                team = teams[side].get("team", {})
                if team["id"] == team_id:
                    stats = teams[side].get("teamStats", {})
                    batting_stats = stats.get("batting", {})
                    pitching_stats = stats.get("pitching", {})

                    total_bases = batting_stats.get("totalBases", 0)
                    walks_issued = pitching_stats.get("baseOnBalls", 0)
                    strikeouts_thrown = pitching_stats.get("strikeOuts", 0)

                    total_bases_list.append(total_bases)
                    walks_issued_list.append(walks_issued)
                    strikeouts_thrown_list.append(strikeouts_thrown)
                    break

        if len(total_bases_list) == 0:
            return None

        return {
            "total_bases": round(sum(total_bases_list) / len(total_bases_list), 2),
            "walks_issued": round(sum(walks_issued_list) / len(walks_issued_list), 2),
            "strikeouts_thrown": round(sum(strikeouts_thrown_list) / len(strikeouts_thrown_list), 2)
        }

    def get_win_percentages():
        standings_url = "https://statsapi.mlb.com/api/v1/standings?season=2025&leagueId=103,104&standingsTypes=regularSeason"
        standings_data = requests.get(standings_url).json()
        win_pct_map = {}
        for record in standings_data.get("records", []):
            for team_record in record.get("teamRecords", []):
                team_id = team_record.get("team", {}).get("id")
                wins = team_record.get("wins", 0)
                losses = team_record.get("losses", 0)
                total = wins + losses
                if total > 0:
                    win_pct_map[team_id] = round(wins / total, 3)
        return win_pct_map

    with st.spinner("Fetching 10-game averages from MLB API..."):
        stats_data = {}
        win_pct_data = get_win_percentages()
        for abbr, team_id in mlb_team_ids.items():
            stats = get_last_10_game_stats(team_id)
            if stats:
                stats["win_pct"] = win_pct_data.get(team_id, 0.5)
                stats_data[abbr] = stats

        if stats_data:
            df_live = pd.DataFrame.from_dict(stats_data, orient="index").sort_index()
            st.success("Live stats successfully pulled!")
            st.dataframe(df_live.style.format({
                "total_bases": "{:.2f}",
                "walks_issued": "{:.2f}",
                "strikeouts_thrown": "{:.2f}",
                "win_pct": "{:.3f}"
            }), use_container_width=True)
        else:
            st.warning("Could not load stats from the API.")


# === Live News Feeds ===
elif page == "Team News Feeds":
    st.title("📰 MLB Team News Feed")

    custom_team_list = ["American League", "National League"] + sorted(team_map.keys())
    league_abbr_map = {
        "American League": "AL",
        "National League": "NL"
    }

    selected_label = st.selectbox("Choose a team or league:", custom_team_list)
    selected_team = league_abbr_map.get(selected_label, selected_label)

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

    if selected_team == "AL":
        st.subheader("📰 American League News (via ESPN)")
        rss_url = "https://www.espn.com/espn/rss/mlb/news"
    elif selected_team == "NL":
        st.subheader("📰 National League News (via ESPN)")
        rss_url = "https://www.espn.com/espn/rss/mlb/news"
    else:
        team_name = team_name_map.get(selected_team, selected_team.lower())
        rss_url = f"https://www.mlb.com/{team_name}/feeds/news/rss.xml"
        if selected_team in team_logos:
            st.image(team_logos[selected_team], width=150)
        st.subheader(f"Latest news about {selected_team}")

    feed = feedparser.parse(rss_url)
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
    if selected_team not in ["AL", "NL"]:
        st.subheader("📅 Upcoming Schedule")
        team_id = mlb_team_ids.get(selected_team)
        if team_id:
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
            st.error("Team ID not found for schedule lookup.")
    else:
        st.info("Schedule not available for league-wide selections.")
# === Footer ===
st.markdown("---")
version = "v5.0 - Live Data"
last_updated = "2025-05-15"
st.caption(f"🔢 App Version: **{version}**  |  🕒 Last Updated: {last_updated}")

