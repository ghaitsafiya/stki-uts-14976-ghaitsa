import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VSMRetrieval:
    """
    Vector Space Model (VSM) untuk Sistem Temu Kembali Informasi.
    Termasuk:
    - Load dokumen hasil preprocessing
    - TF, DF, IDF → TF-IDF Matrix (sparse)
    - Vectorisasi query
    - Ranking cosine similarity
    - Evaluasi: Precision@k, MAP@k, nDCG@k (UJI WAJIB)
    """

    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir  # Menyimpan folder tempat file teks hasil preprocessing disimpan
        self.docs = []                      # List untuk menyimpan isi dokumen
        self.doc_ids = []                   # List untuk menyimpan nama file dokumen
        self.vectorizer = None              # Akan diisi dengan TfidfVectorizer()
        self.tfidf_matrix = None            # Akan diisi dengan matrix TF-IDF berbentuk sparse (CSR matrix)

    # LOAD DOKUMEN HASIL PREPROCESSING
    def load_processed_docs(self):
        """
        Membaca semua dokumen .txt di folder processed.
        Isi file sudah berupa token hasil preprocessing.
        """
        files = sorted(os.listdir(self.processed_dir))

        for fname in files:
            if fname.endswith(".txt"):
                with open(os.path.join(self.processed_dir, fname), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                self.docs.append(text)
                self.doc_ids.append(fname)

        print(f"[INFO] Loaded {len(self.docs)} documents.")


    # MEMBUAT TF-IDF MATRIX
    def build_tfidf(self):
        """
        TF, DF, IDF dihitung otomatis oleh TfidfVectorizer.
        Matrix yang dihasilkan = sparse CSR.
        """
        if not self.docs:
            raise ValueError("Dokumen belum dimuat.")

        self.vectorizer = TfidfVectorizer()                             # Inisialisasi vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)    # Hitung TF-IDF untuk semua dokumen

        print(f"[INFO] TF-IDF shape: {self.tfidf_matrix.shape} (docs x terms)")

    
    # QUERY → TF-IDF VECTOR
    def vectorize_query(self, query):
        """
        Mengubah query (string) menjadi TF-IDF vector.
        Menggunakan vocabulary yang sama dengan dokumen.
        """
        query = query.lower().strip()
        return self.vectorizer.transform([query])


    # COSINE SIMILARITY RANKING
    def rank(self, query, k=5):
        """
        Output:
        - doc_id
        - cosine score
        - snippet 120 char
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF belum dibuat.")

        q_vec = self.vectorize_query(query)                            # Ubah query menjadi TF-IDF vektor
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten() # Hitung cosine similarity antara query dan semua dokumen

        top_idx = np.argsort(scores)[::-1][:k]  # Ambil indeks dokumen dengan skor tertinggi

        results = []
        for idx in top_idx:
            snippet = self.docs[idx][:120].replace("\n", " ")
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(scores[idx]),
                "snippet": snippet
            })

        return results
   
    # -------------------------------------------------------------

    # PRECISION @ k
    def precision_at_k(self, retrieved, relevant, k):
        """
        Precision@k = proporsi dokumen TOP-K yang relevan.
        retrieved = daftar doc_id hasil ranking
        relevant  = gold set (task C)
        """
        retrieved_k = retrieved[:k]
        rel = set(relevant)
        hit = sum(1 for d in retrieved_k if d in rel)
        return hit / k

    # AVERAGE PRECISION (untuk MAP)
    def average_precision(self, retrieved, relevant, k):
        """
        Average Precision digunakan untuk MAP@k.
        Menghitung AP untuk satu query.
        """
        rel = set(relevant)
        score = 0.0
        hit = 0

        for i in range(min(k, len(retrieved))):
            if retrieved[i] in rel:
                hit += 1
                score += hit / (i + 1)

        if len(relevant) == 0:
            return 0.0

        return score / len(relevant)

    # nDCG @ k
    def ndcg_at_k(self, retrieved, relevant, k):
        """
        nDCG@k = Normalized Discounted Cumulative Gain.
        Semakin tinggi posisi dokumen relevan, semakin besar skor.
        """
        rel = set(relevant)
        dcg = 0.0

        # Hitung DCG: diskon logaritmik berdasarkan rank
        for i in range(min(k, len(retrieved))):
            if retrieved[i] in rel:
                dcg += 1.0 / np.log2(i + 2)

        # Hitung IDCG (skor ideal)
        ideal_hits = min(k, len(relevant))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0

        return dcg / idcg


# DEMO KALAU FILE DI-RUN LANGSUNG
if __name__ == "__main__":
    vsm = VSMRetrieval("data/processed")
    vsm.load_processed_docs()
    vsm.build_tfidf()

    query = "sistem informasi kampus"
    results = vsm.rank(query, k=5)

    print("\n=== TOP-5 RANKING ===")
    for r in results:
        print(f"{r['doc_id']} | {r['score']:.4f} | {r['snippet']}")

    # contoh gold set
    gold = ["doc1.txt", "doc3.txt"]

    retrieved = [r["doc_id"] for r in results]

    print("\n=== UJI WAJIB ===")
    print("Precision@5:", vsm.precision_at_k(retrieved, gold, 5))
    print("MAP@5:", vsm.average_precision(retrieved, gold, 5))
    print("nDCG@5:", vsm.ndcg_at_k(retrieved, gold, 5))
