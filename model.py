import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
train_df = pd.read_csv('data/train_data_preprocessed.csv')

# Combine TITLE and DESCRIPTION into a single text feature
train_df['text'] = train_df['TITLE'].astype(str) + ' ' + train_df['DESCRIPTION'].astype(str)

# Features and labels
X = train_df['text']
y = train_df['GENRE']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Train a Logistic Regression classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = clf.predict(X_val_vec)
print('Validation Accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Function to predict genre for new input
def predict_genre(title, description):
    text = title + ' ' + description
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

# Example usage:
# genre = predict_genre('The Matrix', 'A computer hacker learns about the true nature of reality and his role in the war against its controllers.')
# print('Predicted genre:', genre)