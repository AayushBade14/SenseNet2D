import numpy as np
import argparse
from libs.sense_proc import TextPreprocessor
from libs.text_encoder import TextEncoder
from libs.dataloader import Dataloader
from models.logistic_regression import LogisticRegression
import os
import pickle

def main():
    parser = argparse.ArgumentParser(description="Train or predict with Logistic Regression")
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help="train or predict using saved model")
    parser.add_argument('--model-path', default='logistic_model.pkl', help="Path to save/load the model")
    parser.add_argument('--processor-path', default='processor.pkl', help="Path to save/load the processor (vocab/idf)")
    parser.add_argument('--data-path', default='./dataset/train.csv', help="Path to data (train or test)")
    args = parser.parse_args()

    # 1. Load data
    loader = Dataloader(args.data_path, drop_columns=["id", "user", "location"])
    df = loader.load_data()
    df = loader.preprocess_text()
    df = df[df["sentiment"].isin(["positive", "negative"])]
    X_texts, y_labels = loader.split_data()
    y = np.array([1 if label == "positive" else 0 for label in y_labels])

    # 2. Encode text
    processor = TextPreprocessor()
    if args.mode == "train":
        processor.build_vocab(X_texts)
        processor.idf(X_texts)
        with open(args.processor_path, "wb") as f:
            pickle.dump({'vocab': processor.vocab, 'idf_values': processor.idf_values}, f)
    else:
        if not os.path.exists(args.processor_path):
            raise FileNotFoundError("Processor file not found. You must train first or provide processor.pkl")
        with open(args.processor_path, "rb") as f:
            saved = pickle.load(f)
            processor.vocab = saved['vocab']
            processor.idf_values = saved['idf_values']

    X_encoded = processor.tfidf(X_texts)

    # 3. Train or Predict
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

