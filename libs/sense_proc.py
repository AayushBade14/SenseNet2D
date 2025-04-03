import numpy as np
import pandas as pd

class TextPreprocessor:
    
    def __init__(self):
        self.vocab = [] # stores unique words
        self.idf_vector = None # stores IDF values

    def lowercase(self,text):
        """Convert the given text to lowercase
           as sentiments are case insensitive
        """
        return text.lower()

    def rm_punc_num(self,text):
        """Removes punctuations and numbers
           as these don't contribute much to sentiments
        """
        reqd_chars = "abcdefghijklmnopqrstuvwxyz"
        return "".join([char if char in reqd_chars else " " for char in text])

    def rm_stopwords(self,text):
        """Removes common stopwords
           These are common words like 
           'in', 'the', 'and'... etc
           that don't contribute to any sentiment
        """

        stopwords = {
            "the",
            "is",
            "in",
            "and",
            "to",
            "a",
            "on",
            "this",
            "that",
            "it",
            "of"
        }

        words = text.split()

        return "".join([word for word in words if word not in stopwords ])

    def tokenize(self,text):
        """Splits text into tokens (individual words)"""
        return text.split()

    def build_vocab(self,texts):
        """Creates vocabulary from the text datasets"""
        
        unique_words = set()
        
        for text in texts:
            words = self.tokenize(text)
            unique_words.update(words)

        self.vocab = list(unique_words)

    def tf(self,text):
        """Computes Term-Frequency(TF)
           It is a measure of how often a term
           appears in a document/text 

           TF(t) = (# times term t appears in a doc/text)/(total # terms in doc/text)
        """
        words = self.tokenize(text)
        word_freq = np.zeros(len(self.vocab))

        for i,word in enumerate(self.vocab):
            word_freq[i] = words.count(word)

        return word_freq/max(1,len(words))
    
    def idf(self,texts):
        """Computes Inverse-Document-Frequency(IDF)
           It measures how important a word is by checking how rare
           it is across multiple docs/texts

           IDF(t) = log((total # docs/texts)/(# docs/texts containing term t))
        """
        num_docs = len(texts)
        word_doc_counts = np.zeros(len(self.vocab))

        for i,word in enumerate(self.vocab):
            word_doc_counts[i] = sum(1 for text in texts if word in self.tokenize(text))

        self.idf_vector = np.log(num_docs/(1+word_doc_counts))

    def tfidf(self,texts):
        """Computes TF-IDF for all texts
           TF-IDF = TF x IDF
        """
        return np.array([self.tf(text) * self.idf_vector for text in texts])
