"""
Course: CSC 466
Quarter: Fall 2023
Assigment: Lab 6

Name(s):
    Brendan Callender // bscallen@calpoly.edu
    Andrew Kerr // adkerr@calpoly.edu

Description:
    Creates a Vector class to store sparse tf-idf vector representations of text documents. 
    It also implements two similarity score computations for a pair of vectors: cosine similarity and okapi
"""
from __future__ import annotations
import pandas as pd
import numpy as np


class Vector:
    def __init__(self, doc, author, tf, tfidf_series, size):
        self.doc = doc
        self.author = author
        self.tf = tf
        self.tfidf_series = tfidf_series
        self.size = size

    def __lt__(self, other: Vector) -> bool:
        return self.num < other.num

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        else:
            return self.tfidf_series == other.tfidf_series and self.author == other.author

    def __repr__(self) -> str:
        return f'Document: {self.doc}\nAuthor: {self.author}\nSize: {self.size}'
    
    def __str__(self) -> str:
        return f'Document: {self.doc}\nAuthor: {self.author}\nSize: {self.size}'

    def get_doc(self):
        return self.doc

    def get_author(self):
        return self.author
        
    def get_tf(self):
        return self.tf

    def get_tfidf_series(self):
        return self.tfidf_series
    
    def get_size(self):
        return self.size

    def cosine_similarity(self, other):
        shared_words = self.tfidf_series.index.intersection(other.tfidf_series.index)

        dot_prod = np.sum(self.tfidf_series[shared_words] * other.tfidf_series[shared_words])
        mag_self = np.linalg.norm(self.tfidf_series.values)
        mag_other = np.linalg.norm(other.tfidf_series.values)
        
        return dot_prod / (mag_self * mag_other + 1e-10)

    def okapi_similarity(self, other, df, avdl, k1=1.5, b=.75, k2=500):
        """
        Compensates for the disparity in the size between two comapred documents
        # Inputs
            - self, other: Vector class objects
            - df: document frequency of all words in all documents in D
            - avdl: average length (in bytes) of a document in D
            - k1: normalization parameter for self (1.0 - 2.0)
            - b: normalization parameter for document length (usually 0.75)
            - k2 normalization parameter for other (1 - 1000)
        """
        shared_words = self.tfidf_series.index.intersection(other.tfidf_series.index)

        sim = np.sum(
            np.log(
                (self.tfidf_series.shape[0] - df[df['term'].isin(shared_words)]['idf'].values + 0.5) / (df[df['term'].isin(shared_words)]['idf'] + 0.5).values
            ) 
            * (((k1 + 1) * self.tf[shared_words]) / (k1 * (1 - b + b * (self.size / avdl)) + self.tf[shared_words] + 1e-10)).values
            * (((k2 + 1) * other.tf[shared_words]) / (k2 + other.tf[shared_words])).values
        )

        return sim
