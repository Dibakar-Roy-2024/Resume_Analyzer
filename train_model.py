import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# ✅ Load dataset
df = pd.read_csv("resume.csv")

# ✅ Convert text data into numerical vectors
vectorizer = TfidfVectorizer(max_features=500)
x = vectorizer.fit_transform(df["Resume"]).toarray()

# ✅ Convert job categories into numerical labels
y = pd.factorize(df["Category"])[0]

# ✅ Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ✅ Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# ✅ Evaluate the model
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ✅ Save the trained model and vectorizer
pickle.dump(model, open("resume_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
