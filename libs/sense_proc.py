import numpy as np
import pickle

class TextPreprocessor:

    def __init__(self):
        self.vocab = []
        self.idf_vector = None

    def lowercase(self, text):
        return text.lower()

    def rm_punc_num(self, text):
        reqd_chars = "abcdefghijklmnopqrstuvwxyz"
        return "".join([char if char in reqd_chars else " " for char in text])

    def rm_stopwords(self, text):
        stopwords = {
            "the", "is", "in", "and", "to", "a", "on", "this",
            "that", "it", "of"
        }
        words = text.split()
        return " ".join([word for word in words if word not in stopwords])

    def tokenize(self, text):
        return text.split()

    def build_vocab(self, texts):
        unique_words = set()
        for text in texts:
            words = self.tokenize(text)
            unique_words.update(words)
        self.vocab = list(unique_words)

    def tf(self, text):
        words = self.tokenize(text)
        word_freq = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            word_freq[i] = words.count(word)
        return word_freq / max(1, len(words))

    def idf(self, texts):
        num_docs = len(texts)
        word_doc_counts = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            word_doc_counts[i] = sum(1 for text in texts if word in self.tokenize(text))
        self.idf_vector = np.log(num_docs / (1 + word_doc_counts))

    def tfidf(self, texts):
        return np.array([self.tf(text) * self.idf_vector for text in texts])

    def preprocess(self, text):
        text = self.lowercase(text)
        text = self.rm_punc_num(text)
        text = self.rm_stopwords(text)
        return text

    def save(self, path="preprocessor.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"vocab": self.vocab, "idf": self.idf_vector}, f)

    def load(self, path="preprocessor.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vocab = data["vocab"]
            self.idf_vector = data["idf"]

