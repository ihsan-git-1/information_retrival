import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utility.text_processing_helper import  preprocess_text
import numpy as np


class VectorSpaceModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            max_df=0.7,
            stop_words='english',
            norm="l2",
            use_idf=True,
        )
        self.documents = []

    def add_document(self, doc_id, text):
        self.documents.append((doc_id, text))

    def build_vsm_tfidf(self):
        document_texts = [doc[1]
                          for doc in self.documents if doc[1] is not None]
        tfidf_vectors = self.tfidf_vectorizer.fit_transform(document_texts)
        # Set Feature Names
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.tfidf_vectors = tfidf_vectors
      
    def save(self, filename="state.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename="state.pkl"):
        with open(filename, 'rb') as f:
            return pickle.load(f)