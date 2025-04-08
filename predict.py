import numpy as np
from models.logistic_regression import LogisticRegression
from libs.sense_proc import TextPreprocessor

# Step 1: Load saved model and vocab
model = LogisticRegression()
model.load_model("model.pkl")

vocab = np.load("vocab.npy", allow_pickle=True).tolist()
idf_vector = np.load("idf.npy")

# Step 2: Prepare text preprocessor
preprocessor = TextPreprocessor()
preprocessor.vocab = vocab
preprocessor.idf_vector = idf_vector

# Step 3: Get input
input_text = input("Enter text to analyze sentiment: ")

# Step 4: Preprocess and convert to vector
cleaned = preprocessor.preprocess(input_text)
tfidf_vector = preprocessor.tf(cleaned) * preprocessor.idf_vector
tfidf_vector = tfidf_vector.reshape(1, -1)  # Reshape to (1, D)

# Step 5: Predict
prob = model.predict_prob(tfidf_vector)[0]
label = model.predict(tfidf_vector)[0]

print(f"[INFO] Sentiment Probability (Positive): {prob:.4f}")
print(f"[INFO] Predicted Sentiment: {'Positive' if label == 1 else 'Negative'}")

