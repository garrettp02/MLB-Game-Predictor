import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("/Users/garrettparr/Com Sci C++ Home/MLB Decision Tree/plays.csv")


# Clean and prepare data
df = df.dropna(subset=["Event", "Team", "Batter Id"])

# Extract features and target
vectorizer = CountVectorizer(max_features=1000)  # Convert event text to vector
X = vectorizer.fit_transform(df["Event"])

# Example target: predicting which team made the play (you can change this)
y = df["Team"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example query (predicting team from a new event description)
example_event = ["Carpenter grounded into double play"]
query_vec = vectorizer.transform(example_event)
prediction = clf.predict(query_vec)
print(f"Prediction for query '{example_event[0]}': {prediction[0]}")
