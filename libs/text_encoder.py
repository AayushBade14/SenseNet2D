class TextEncoder:
    
    def __init__(self, vocab):
        self.word_to_index = {word: i + 1 for i, word in enumerate(vocab)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.unknown_token = "<UNK>"

    def encode_text(self, text):
        return [self.word_to_index.get(word, 0) for word in text.split()]

    def decode_text(self, indices):
        return " ".join([self.index_to_word.get(i, self.unknown_token) for i in indices])

