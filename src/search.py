"""
======================================================
File: search.py
Deskripsi:
Main orchestrator buat menjalankan sistem IR.
- Bisa pilih model (boolean / vsm)
- Bisa atur jumlah hasil (top-k)
- Bisa rebuild preprocessing
======================================================
"""

import argparse, os
from src.preprocess import preprocess_text, process_all_raw
from src.boolean_ir import build_inverted_index, parse_boolean_query
from src.vsm_ir import VSM_IR


def load_processed_docs(processed_dir='data/processed'):
    """Baca semua file hasil preprocessing"""
    docs = {}
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(processed_dir, fname), 'r', encoding='utf-8') as f:
            docs[fname] = f.read().strip()
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['boolean', 'vsm'], default='vsm', help='Pilih model retrieval')
    p.add_argument('--query', type=str, required=True, help='Masukkan query pencarian')
    p.add_argument('--k', type=int, default=5, help='Ambil berapa hasil teratas')
    p.add_argument('--rebuild', action='store_true', help='Jalankan ulang preprocessing')
    p.add_argument('--sublinear', action='store_true', help='Gunakan pembobotan sublinear (1+log(tf))')
    args = p.parse_args()

    # Jalankan preprocessing kalau diminta
    if args.rebuild:
        process_all_raw()

    docs = load_processed_docs()

    if args.model == 'boolean':
        # === MODEL BOOLEAN RETRIEVAL ===
        inv = build_inverted_index(docs)
        all_docs = set(docs.keys())
        res, explanation = parse_boolean_query(args.query, inv, all_docs)
        print("=== BOOLEAN RETRIEVAL ===")
        print("Query :", args.query)
        print("Penjelasan :", explanation)
        print("Hasil dokumen relevan :", sorted(res))

    else:
        # === MODEL VECTOR SPACE (VSM) ===
        vsm = VSM_IR(scheme='tfidf', sublinear_tf=args.sublinear)
        vsm.fit(docs)
        q_tokens = preprocess_text(args.query)
        q_str = " ".join(q_tokens)
        results = vsm.query(q_str, k=args.k)
        print("=== VECTOR SPACE MODEL ===")
        print("Query :", args.query)
        for doc, score in results:
            print(f"{doc:25s} | skor = {score:.4f}")


if __name__ == "__main__":
    main()
