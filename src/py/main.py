import numpy as np
from ../../libs/sense_proc.py import TextPreprocessor
from ../../libs/text_encoder.py import TextEncoder
from ./dataloader.py import Dataloader
from ../models/regression/logistic_regression.py import LogisticRegression

def main():
    # 1. Load data
    loader = Dataloader("",drop_columns=["id","user","location"])
    df = loader.load_data()

    # 2. preprocess text
    df = loader.preprocess_text()
    
    # 3. Filter only +ve and -ve sentiments (for binary classification)
    df = df[df["sentiment"].isin(["positive","negative"])]
    
    # 4. Get X (text) and y (labels)
    X_texts,y_labels = loader.split_data()
    y = np.array([1 if label == "positive" else 0 for label in y_labels])

    # 5. Build vocab and encode
    processor = TextPreprocessor()
    processor.build_vocab(X_texts)
    processor.idf(X_texts)
    X_encoded = processor.tfidf(X_texts)


    # 6. Train the model
    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_encoded, y)

    # 7. Evaluate
    y_pred = model.predict(X_encoded)
    acc = model.accuracy(y, y_pred)
    print(f"Training Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

