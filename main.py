# AI Resume Analyzer - Edunet Internship Project

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')

# ----------------- Load and preprocess dataset -----------------
df = pd.read_csv("Dataset/ResumeDataSet.csv")  # Must contain 'Resume' and 'Category'


def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned)


df['cleaned'] = df['Resume'].apply(preprocess)

# ----------------- Train the model -----------------
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


# ----------------- Prediction loop -----------------
def predict_role(text):
    processed = preprocess(text)
    vector = vectorizer.transform([processed]).toarray()
    return model.predict(vector)[0]


print("\nModel trained. You can now input resumes to predict roles.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Paste resume text (or type 'exit'): ").strip()
    if user_input.lower() == "exit":
        print("Exiting. Thank you!")
        break
    elif user_input == "":
        continue
    else:
        role = predict_role(user_input)
        print("Predicted Job Role:", role, "\n")
