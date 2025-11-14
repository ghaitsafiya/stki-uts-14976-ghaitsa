import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import os
import matplotlib.pyplot as plt
from collections import Counter
from src.preprocess import preprocess_text as preprocess_pipeline

# === Konfigurasi path ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# === Judul Halaman ===
st.set_page_config(page_title="UTS STKI 14976 - Preprocessing", layout="wide")
st.title("🧠 UTS STKI — Document Preprocessing")
st.markdown("**A11.2023.14976 — Ghaitsa Qiyala Shafiya**")

# === Pilih Dokumen ===
files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
selected_file = st.selectbox("📂 Pilih Dokumen:", files)

if selected_file:
    file_path = os.path.join(RAW_DIR, selected_file)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Jalankan preprocessing
    tokens = preprocess_pipeline(text)
    processed_text = " ".join(tokens)

    # Simpan hasil
    with open(os.path.join(PROC_DIR, selected_file), "w", encoding="utf-8") as fo:
        fo.write(processed_text)

    # Tampilkan hasil Before–After
    st.subheader("📜 Sebelum Preprocessing")
    st.text_area("Teks Asli", text[:1000], height=200)

    st.subheader("✨ Sesudah Preprocessing (50 token pertama)")
    st.text_area("Token Hasil", " ".join(tokens[:50]) + " ...", height=200)

    # Tampilkan top token
    st.subheader("🔝 Top 10 Token Paling Sering")
    freq = Counter(tokens)
    top10 = freq.most_common(10)
    st.table(top10)

# === Distribusi Panjang Dokumen ===
if st.button("📊 Lihat Grafik Distribusi Panjang Dokumen"):
    lengths = []
    for fname in os.listdir(PROC_DIR):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(PROC_DIR, fname), "r", encoding="utf-8") as f:
            toks = f.read().split()
        lengths.append((fname, len(toks)))

    if lengths:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar([x[0] for x in lengths], [x[1] for x in lengths], color="#00bfff", edgecolor="navy")
        ax.set_title("Distribusi Panjang Dokumen (Jumlah Token)")
        ax.set_xlabel("Dokumen")
        ax.set_ylabel("Jumlah Token")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
