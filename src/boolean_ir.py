"""
======================================================
File: boolean_ir.py
Deskripsi:
Model Boolean Retrieval.
Konsepnya: dokumen dianggap relevan kalau memenuhi kondisi logika
berdasarkan query pengguna (pakai operator AND, OR, NOT).
======================================================
"""

from typing import Dict, Set, List
import re


def build_inverted_index(docs: Dict[str, str]) -> Dict[str, Set[str]]:
    """
    Bikin struktur inverted index:
    - Input: dict berisi doc_id -> isi dokumen (hasil preprocessing)
    - Output: dict berisi term -> set dokumen yang mengandung term tsb
    """
    inv = {}
    for doc_id, text in docs.items():
        tokens = set(text.split())
        for t in tokens:
            inv.setdefault(t, set()).add(doc_id)
    return inv


def _tokenize_query(query: str) -> List[str]:
    """Pisahin query user jadi token (bisa AND, OR, NOT, kurung)"""
    q = query.lower()
    q = q.replace('(', ' ( ').replace(')', ' ) ')
    toks = q.split()
    normalized = []
    for t in toks:
        if t in ('and', '&&'):
            normalized.append('AND')
        elif t in ('or', '||'):
            normalized.append('OR')
        elif t in ('not', '!'):
            normalized.append('NOT')
        else:
            normalized.append(re.sub(r'[^\w<>-]', '', t))
    return normalized


def parse_boolean_query(query: str, inverted_index: Dict[str, Set[str]], all_docs: Set[str]):
    """
    Parse query Boolean sederhana + evaluasi hasilnya.
    Output:
      - result: dokumen relevan (set)
      - explanation: penjelasan langkah operasi (buat laporan / explain)
    """
    tokens = _tokenize_query(query)
    i = 0

    # Rekursif parsing query
    def parse_factor():
        nonlocal i
        if i < len(tokens) and tokens[i] == 'NOT':
            i += 1
            operand, expl = parse_factor()
            res = all_docs - operand
            return res, f"NOT({expl})"
        if i < len(tokens) and tokens[i] == '(':
            i += 1
            res, expl = parse_expr()
            if i < len(tokens) and tokens[i] == ')':
                i += 1
            return res, f"({expl})"
        if i < len(tokens):
            term = tokens[i]
            i += 1
            docs = set(inverted_index.get(term, set()))
            return docs, term
        return set(), "EMPTY"

    def parse_term():
        nonlocal i
        left, expl_left = parse_factor()
        while i < len(tokens) and tokens[i] == 'AND':
            i += 1
            right, expl_right = parse_factor()
            left = left & right
            expl_left = f"({expl_left} AND {expl_right})"
        return left, expl_left

    def parse_expr():
        nonlocal i
        left, expl_left = parse_term()
        while i < len(tokens) and tokens[i] == 'OR':
            i += 1
            right, expl_right = parse_term()
            left = left | right
            expl_left = f"({expl_left} OR {expl_right})"
        return left, expl_left

    result, explanation = parse_expr()
    return result, explanation
