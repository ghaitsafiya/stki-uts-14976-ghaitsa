"""
======================================================
File: vsm_ir.py
Deskripsi:
Implementasi Vector Space Model (VSM) dengan pembobotan TF-IDF.
Tahapan:
1. Hitung TF, DF, IDF
2. Bentuk vektor dokumen
3. Normalisasi
4. Hitung cosine similarity buat ranking hasil pencarian
======================================================
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


class VSM_IR:
    def __init__(self, scheme='tfidf', sublinear_tf=False):
        """
        scheme: jenis pembobotan ('tfidf' default)
        sublinear_tf: kalau True, pakai log-scaling (1 + log(tf))
        """
        self.scheme = scheme
        self.sublinear_tf = sublinear_tf
        self.doc_freq = {}
        self.idf = {}
        self.doc_vectors = {}
        self.N = 0

    def fit(self, docs: Dict[str, str]):
        """Bangun model vektor dari semua dokumen"""
        self.N = len(docs)
        df = defaultdict(int)
        docs_terms = {}

        # Hitung DF (berapa dokumen yang mengandung term)
        for doc_id, text in docs.items():
            terms = text.split()
            docs_terms[doc_id] = Counter(terms)
            for t in set(terms):
                df[t] += 1

        self.doc_freq = dict(df)
        # Hitung IDF dengan smoothing (biar ga NaN)
        self.idf = {t: math.log((self.N) / df_t) + 1.0 for t, df_t in df.items()}

        # Bangun vektor dokumen
        for doc_id, ctr in docs_terms.items():
            vec = {}
            for t, tf in ctr.items():
                tf_w = 1 + math.log(tf) if self.sublinear_tf and tf > 0 else tf
                if self.scheme == 'tfidf':
                    vec[t] = tf_w * self.idf.get(t, 0.0)
                else:
                    vec[t] = tf_w
            # Normalisasi panjang vektor (biar fair)
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for t in vec:
                vec[t] /= norm
            self.doc_vectors[doc_id] = vec

    def _vectorize_query(self, query_terms: List[str]):
        """Ubah query user jadi vektor TF-IDF"""
        ctr = Counter(query_terms)
        vec = {}
        for t, tf in ctr.items():
            tf_w = 1 + math.log(tf) if self.sublinear_tf and tf > 0 else tf
            if self.scheme == 'tfidf':
                idf_t = self.idf.get(t, math.log((self.N) / 1) + 1.0)
                vec[t] = tf_w * idf_t
            else:
                vec[t] = tf_w
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for t in vec:
            vec[t] /= norm
        return vec

    def query(self, query: str, k=5) -> List[Tuple[str, float]]:
        """
        Hitung cosine similarity antara query dan semua dokumen,
        terus urutin dari skor tertinggi.
        """
        terms = query.split()
        qvec = self._vectorize_query(terms)
        scores = []
        for doc_id, dvec in self.doc_vectors.items():
            s = sum(qvec.get(t, 0.0) * dvec.get(t, 0.0) for t in qvec)
            scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
