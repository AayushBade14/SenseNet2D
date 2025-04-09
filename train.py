import numpy as np

from models.logistic_regression import LogisticRegression
from libs.dataloader import Dataloader
from libs.sense_proc import TextPreprocessor

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Step 1: Load and preprocess data
loader = Dataloader("./dataset/train.csv", drop_columns=["id", "user"])  # change as needed
df = loader.load_data()
df = loader.preprocess_text()

# Step 2: Extract X (text) and y (labels)
X_raw, y_raw = loader.split_data()

# Step 3: Convert y to binary labels (edit label logic if needed)
y = np.array([1 if label.lower() in ["positive", "pos", "1", "yes"] else 0 for label in y_raw])

# Step 4: Vectorize using TF-IDF
preprocessor = TextPreprocessor()
preprocessor.build_vocab(X_raw)
preprocessor.idf(X_raw)
X = preprocessor.tfidf(X_raw)

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Train the model
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
acc = model.accuracy(y_test, y_pred)
print(f"[RESULT] Test Accuracy: {acc * 100:.2f}%")

# Step 8: Save model and preprocessing objects
model.save_model("model.pkl")
np.save("vocab.npy", preprocessor.vocab)
np.save("idf.npy", preprocessor.idf_vector)
print("[INFO] Model and vocab saved.")

