import numpy as np

class TextEncoder:
    
    def __init__(self,vocab):
        """Initialize encoder with a vocabulary"""
        self.word_to_index = {word: i+1 for i, word in enumerate(vocab)} # starting from 1 and leaving 0 for padding
        self.index_to_word = {i: word for word, i in self.word_to_index.items()} # Reserving mapping for decoding
        self.unknown_token = "<UNK>" # Token for unknown words

    def encode_text(self, text):
        """Encodes text into a list of numbers based on the vocabulary"""
        return [self.word_to_index.get(word,0) for word in text.split()] # 0 for unknown words
    
    def decode_text(self, indices):
        """Decodes a list of numbers into a text string"""
        return " ".join([self.index_to_word.get(i,self.unknown_token) for i in indices])

