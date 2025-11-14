"""
======================================================
File: preprocess.py
Deskripsi:
Tahapan awal (preprocessing) di sistem temu kembali informasi (STKI).
Langkah-langkahnya:
1. Case folding (semua huruf jadi kecil)
2. Normalisasi angka dan tanda baca
3. Tokenisasi (pisah jadi kata-kata)
4. Stopword removal (hapus kata umum kayak 'dan', 'yang', dll)
5. Stemming (ubah kata ke bentuk dasar, misal 'berlari' -> 'lari')
6. Simpan hasilnya ke folder data/processed/
======================================================
"""

import os, re
from collections import Counter
from typing import List
import nltk

# --- Cek dan download resource NLTK (biar ga error pas tokenisasi / stopword) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Coba pakai stemmer Bahasa Indonesia (library Sastrawi) ---
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
except Exception:
    _stemmer = None  # fallback kalau Sastrawi belum keinstall

# --- Ambil stopword list Bahasa Indonesia ---
DEFAULT_STOPWORDS = set(stopwords.words('indonesian')) if 'indonesian' in stopwords.fileids() else set(stopwords.words('english'))


def normalize_text(text: str) -> str:
    """Langkah 1-2: ubah huruf jadi kecil + bersihin angka dan tanda baca"""
    text = text.lower()
    # ganti angka jadi token <num>
    text = re.sub(r'\d+(\.\d+)?', ' <num> ', text)
    # hapus simbol dan tanda baca
    text = re.sub(r'[^\w\s<>-]', ' ', text)
    # rapihin spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Langkah 3: ubah teks jadi list token/kata"""
    try:
        tokens = word_tokenize(text)
    except Exception:
        # kalau nltk ga bisa, fallback ke regex
        tokens = re.findall(r"\w+|<num>", text)
    return tokens


def remove_stopwords(tokens: List[str], extra_stopwords: set = None) -> List[str]:
    """Langkah 4: hapus kata umum (stopwords)"""
    sw = set(DEFAULT_STOPWORDS)
    if extra_stopwords:
        sw |= set(extra_stopwords)
    return [t for t in tokens if t not in sw and t.strip() != '']


def stem_tokens(tokens: List[str]) -> List[str]:
    """Langkah 5: stemming pake Sastrawi (kalau ada)"""
    if _stemmer:
        return [_stemmer.stem(t) for t in tokens]
    else:
        return tokens  # kalau gak ada stemmer, balikin token aslinya


def preprocess_text(text: str, do_stem=True, extra_stopwords=None) -> List[str]:
    """Fungsi utama buat menjalankan semua tahapan di atas"""
    norm = normalize_text(text)
    toks = tokenize(norm)
    toks = remove_stopwords(toks, extra_stopwords)
    if do_stem:
        toks = stem_tokens(toks)
    return toks


def process_all_raw(input_dir='data/raw', output_dir='data/processed'):
    """Fungsi otomatis: baca semua file di data/raw dan simpan hasil preprocessing ke data/processed"""
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = preprocess_text(text)
        out_fname = fname if fname.endswith('.txt') else fname + '.txt'
        out_path = os.path.join(output_dir, out_fname)
        with open(out_path, 'w', encoding='utf-8') as fo:
            fo.write(" ".join(tokens))
        print(f"[OK] {fname} -> {len(tokens)} tokens -> {out_path}")


# Kalau file ini dijalankan langsung: proses semua dokumen
if __name__ == "__main__":
    process_all_raw()
    