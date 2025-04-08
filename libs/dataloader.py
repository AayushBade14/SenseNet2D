import numpy as np
import pandas as pd
from libs.sense_proc import TextPreprocessor

class Dataloader:

    def __init__(self, file_path, drop_columns=None):
        """
        Initialize the dataloader with a CSV file

        Parameters:
        - file_path: str, path to the CSV_file
        - drop_columns: list, columns to drop from the dataset
        """
        self.file_path = file_path
        self.drop_columns = drop_columns if drop_columns else []
        self.df = None

    def load_data(self):
        """Loads the dataset and drops the unnecessary columns"""
        self.df = pd.read_csv(self.file_path, encoding="ISO-8859-1")
        self.df.drop(columns=self.drop_columns, inplace=True, errors="ignore")
        self.df.dropna(subset=["text", "sentiment"], inplace=True)
        return self.df

    def preprocess_text(self):
        """Applies basic text preprocessing to the text"""
        preprocessor = TextPreprocessor()
        self.df["text"] = self.df["text"].apply(preprocessor.preprocess)
        return self.df

    def split_data(self):
        """Splits the data into features (X) and labels (Y)"""
        x = self.df["text"].values
        y = self.df["sentiment"].values
        return x, y

