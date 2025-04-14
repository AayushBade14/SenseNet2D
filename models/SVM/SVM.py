

import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load and reduce dataset
df = pd.read_csv("/content/train.csv", encoding="ISO-8859-1")

df = df[['text', 'sentiment']].dropna()
df = df.sample(n=25000, random_state=42)  # 

# Step 2: Preprocessing
X_raw = df['text']
y_raw = df['sentiment']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Step 3: TF-IDF vectorization with limited features
vectorizer = TfidfVectorizer(max_features=100)  # 
X = vectorizer.fit_transform(X_raw).toarray()

# Step 4: Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Try a few C/gamma combinations (RBF or linear kernel)
C_values = [0.1, 1, 10]
gamma_values = [0.01, 0.1,1]
total = len(C_values) * len(gamma_values)
count = 0
best_accuracy = 0
best_model = None
best_params = {}

print(" Starting fast training loop...\n")

# Step 7: Training loop
for C in C_values:
    for gamma in gamma_values:
        count += 1
        print(f" [{count}/{total}] Training C={C}, gamma={gamma}")
        start = time.time()

        model = SVC(kernel='rbf', C=C, gamma=gamma)  # change to 'linear' to go faster
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        end = time.time()

        print(f" Done! Accuracy: {acc * 100:.2f}% | Time: {end - start:.2f}s\n")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_params = {'C': C, 'gamma': gamma}

# Step 8: Save best model
print(" Best Accuracy: {:.2f}%".format(best_accuracy * 100))
print(" Best Parameters: C = {}, gamma = {}".format(best_params['C'], best_params['gamma']))

with open("svm_model.pkl", "wb") as f:
    pickle.dump({
        'model': best_model,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'label_encoder': label_encoder
    }, f)

print(" Model saved as 'svm_model.pkl'")

import pickle

# Load model
with open("svm_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved['model']
vectorizer = saved['vectorizer']
scaler = saved['scaler']
label_encoder = saved['label_encoder']

# Example usage on new text
texts = ["Tushar is happy", "my name is dhruv"]
X_new = vectorizer.transform(texts).toarray()
X_new_scaled = scaler.transform(X_new)
preds = model.predict(X_new_scaled)
labels = label_encoder.inverse_transform(preds)

print(" Predictions:", labels)

