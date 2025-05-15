import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Remove ties
df = df[df["winner"] != "TIE"]

# === Encode team names ===
teams = pd.unique(df[["home", "away"]].values.ravel())
team_map = {team: i for i, team in enumerate(teams)}
reverse_map = {v: k for k, v in team_map.items()}

df["home_id"] = df["home"].map(team_map)
df["away_id"] = df["away"].map(team_map)
df["winner_id"] = df["winner"].map(team_map)

# === Train the Random Forest ===
X = df[["home_id", "away_id"]]
y = df["winner_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    verbosity=0
)

clf.fit(X_train, y_train)

# === Report performance ===
print("Model Evaluation:")
print(classification_report(
    y_test,
    clf.predict(X_test),
    zero_division=0,
    target_names=[reverse_map[i] for i in sorted(y.unique())]
))

# === Prediction function ===
def predict_winner(home_team, away_team):
    if home_team not in team_map or away_team not in team_map:
        print("❌ One or both teams not found in training data.")
        return

    home_id = team_map[home_team]
    away_id = team_map[away_team]
    input_df = pd.DataFrame([[home_id, away_id]], columns=["home_id", "away_id"])

    # Get probabilities
    probs = clf.predict_proba(input_df)[0]
    class_ids = clf.classes_.tolist()

    # Safely retrieve probabilities for both teams
    selected = {}
    for team_id in [home_id, away_id]:
        if team_id in class_ids:
            selected[team_id] = probs[class_ids.index(team_id)]

    if not selected:
        print("❌ No prediction could be made for these teams.")
        return
    elif len(selected) == 1:
        only_team = reverse_map[list(selected.keys())[0]]
        print(f"\n⚠️ Only one team was in training data. Defaulting to: {only_team}")
        return

    winner_id = max(selected, key=selected.get)
    predicted_winner = reverse_map[winner_id]
    prob_margin = abs(selected[home_id] - selected[away_id]) if home_id in selected and away_id in selected else None
    print(f"\n🏆 Predicted winner: {predicted_winner} (Home: {home_team} vs. Away: {away_team})")
    if prob_margin is not None:
        print(f"📊 Confidence margin: {prob_margin:.2f}")

# === Prompt for prediction ===
while True:
    print("\n--- Predict a Game Outcome ---")
    home_team = input("Enter HOME team abbreviation (e.g. NYY): ").strip().upper()
    away_team = input("Enter AWAY team abbreviation (e.g. BOS): ").strip().upper()
    predict_winner(home_team, away_team)

    again = input("Try another? (y/n): ").strip().lower()
    if again != "y":
        break
