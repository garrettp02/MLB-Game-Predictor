import pandas as pd
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("games.csv")
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

teams = pd.unique(df[["home", "away"]].values.ravel())
team_map = {team: i for i, team in enumerate(teams)}
reverse_map = {v: k for k, v in team_map.items()}

df["home_id"] = df["home"].map(team_map)
df["away_id"] = df["away"].map(team_map)
df["winner_id"] = df["winner"].map(team_map)

X = df[["home_id", "away_id"]]
y = df["winner_id"]

model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X, y)

# Save model and mappings
joblib.dump(model, "xgb_model.pkl")
joblib.dump(team_map, "team_map.pkl")
joblib.dump(reverse_map, "reverse_map.pkl")
