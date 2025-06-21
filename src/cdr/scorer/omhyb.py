import fasttext
import numpy as np
import pandas as pd
from math import log10, floor, sqrt
from abc import ABC, abstractmethod
from typing import Tuple, Callable, List, Any
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from .base_scorer import Scorer

class OMHyb(Scorer):
    def __init__(self, verbose=False, **kwargs):
        if 'tokenizer' not in kwargs:
            raise ValueError("tokenizer is required for TfIdfScorer")
        self.model_path = kwargs.get('model_path', None)
        if not self.model_path:
            raise ValueError("model_path is required for omhyb")
        self.model = fasttext.load_model(self.model_path)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.verbose = verbose

    def _compute_tf(self, word_dict, doc_len):
        tf_dict = {}
        for word, count in word_dict.items():
            tf_dict[word] = count / float(doc_len)
        return tf_dict

    def _compute_idf(self, doc_list):
        idf_dict = {}
        N = len(doc_list)

        all_words = set(word for doc in doc_list for word in doc)
        idf_dict = dict.fromkeys(all_words, 0)
        for doc in doc_list:
            for word in doc.keys():
                idf_dict[word] += 1

        for word in idf_dict:
            idf_dict[word] = log10(N / float(idf_dict[word]))

        return idf_dict

    def _compute_tfidf(self, table):
        c2t = {}
        tfidf_dict = {}
        doc_list = []
        for column in table.columns:
            tfidf_dict[column] = {}
            col_vals = [str(entity) for entity in table[column]]
            word_set = set(token for cell in col_vals for token in self.tokenizer.encode(cell))
            word_dict = dict.fromkeys(word_set, 0)
            doc_len = 0
            for cell in col_vals:
                cell_tokens = self.tokenizer.encode(cell)
                doc_len += len(cell_tokens)
                for token in cell_tokens:
                    word_dict[token] += 1
            tf_dict = self._compute_tf(word_dict, doc_len)
            doc_list.append(tf_dict)
            c2t[column] = doc_len

        idf_dict = self._compute_idf(doc_list)

        for col, doc in zip(table.columns, doc_list):
            for word, tf_val in doc.items():
                tfidf_score = tf_val * idf_dict.get(word, 0)
                tfidf_dict[col][word] = tfidf_score

        return tfidf_dict, c2t

    def _guess_k(self, c2c, c2t, c2b):
        c2k = {}
        for col in c2c.keys():
            total_cells = len(list(set(c2c[col])))
            K_base = int(1 * sqrt(total_cells))
            c2k[col] = max(1, min(total_cells, K_base))
        return c2k

    def _compute_medoids(self, k, unique_cells, gt, this_tfidf_scores):
        embeddings = np.array([self.model.get_sentence_vector(cell) for cell in unique_cells])
        this_km = KMeans(n_clusters=k, random_state=rs).fit(embeddings)
        centroids = this_km.cluster_centers_
        sel_cel = [0] * len(unique_cells)
        for c in centroids:
            dist = np.linalg.norm(embeddings - c, axis=1)
            medoid_idx = np.argmin(dist)
            sel_cel[medoid_idx] = 1

        return sel_cel

    def gen_score(self, df: pd.DataFrame, c2b: dict = None, gt=None) :
        c2c, c2s, c2w = {}, {}, {}
        tfidf_dict, c2t = self._compute_tfidf(df)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                raise ValueError("Hybrid does not support numerical columns, make sure numerical ones are pre-processed")
            unique_cells = df[col].astype(str).dropna().unique()
            unique_cells = [cell for cell in unique_cells if cell.lower() not in ['nan', 'none']]
            if len(unique_cells) == 0:
                unique_cells = ['N/A', 'N/A', 'N/A']
            c2c[col] = unique_cells

        c2k = self._guess_k(c2c, c2t, c2b)
        for col in df.columns:
            this_gt = None
            this_tfidf_dict = tfidf_dict[col]
            unique_cells = c2c[col]
            this_scores = []
            this_weights = []
            for cell in unique_cells:
                cell_tokens = self.tokenizer.encode(cell)
                cell_token_count = len(cell_tokens)
                cell_score = sum(this_tfidf_dict.get(token, 0) for token in cell_tokens) / (cell_token_count if cell_token_count > 0 else 1)
                this_scores.append(cell_score)
                this_weights.append(cell_token_count)
            tfidf_scores = np.array(this_scores)
            medois_mask = self._compute_medoids(floor(c2k[col]), unique_cells, this_gt, tfidf_scores)
            assert len(unique_cells) == len(medois_mask) == len(tfidf_scores) == len(this_weights), 'length mismatch'
            ## c=2
            c2s[col] = [2 * tfidf_scores[i] if medois_mask[i] == 1 else 1 * tfidf_scores[i] for i in range(len(unique_cells))]
            c2w[col] = this_weights

        return c2c, c2s, c2w


    def __str__(self):
        return f"OMHybScorer()"
