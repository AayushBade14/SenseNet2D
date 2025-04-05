import numpy as np
import argparse
from libs.sense_proc import TextPreprocessor
from libs.text_encoder import TextEncoder
from libs.dataloader import Dataloader
from models.logistic_regression import LogisticRegression

def main():
    parser = argparse.ArgumentParser(description="Train or predict with Logistic Regression")
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help="train or predict using saved model")
    parser.add_argument('--model-path', default='logistic_model.pkl', help="Path to save/load the model")
    args = parser.parse_args()

    # 1. Load data
    loader = Dataloader("./dataset/train.csv", drop_columns=["id", "user", "location"])
    df = loader.load_data()

    # 2. Preprocess text
    df = loader.preprocess_text()

    # 3. Filter only +ve and -ve sentiments (binary classification)
    df = df[df["sentiment"].isin(["positive", "negative"])]

    # 4. Get X and y
    X_texts, y_labels = loader.split_data()
    y = np.array([1 if label == "positive" else 0 for label in y_labels])

    # 5. Encode text
    processor = TextPreprocessor()
    processor.build_vocab(X_texts)
    processor.idf(X_texts)
    X_encoded = processor.tfidf(X_texts)

    # 6. Train or Predict
    model = LogisticRegression(learning_rate=0.01, epochs=1000)

    if args.mode == "train":
        print("[INFO] Training mode selected.")
        model.fit(X_encoded, y)
        model.save_model(args.model_path)

        y_pred = model.predict(X_encoded)
        acc = model.accuracy(y, y_pred)
        print(f"Training Accuracy: {acc:.4f}")

    else:
        print("[INFO] Prediction mode selected.")
        model.load_model(args.model_path)
        y_pred = model.predict(X_encoded)
        acc = model.accuracy(y, y_pred)
        print(f"Prediction Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

