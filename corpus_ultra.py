#!/usr/bin/env python3
"""
Corpus Linguistics ULTRA v1.0
=============================
Standalone Python script for comprehensive corpus linguistics analysis.

Modules:
  1. KWIC Concordance (with left/right sorting)
  2. Collocation Extraction (LL, PMI, MI³, t-score, z-score, Dice, chi²)
  3. Dispersion Analysis (Juilland's D, DP, DPnorm, KL-divergence, Rosengren's S)
  4. Readability (FK, Gunning FOG, SMOG, Coleman-Liau, ARI, Dale-Chall, Spache, LIX, RIX)
  5. Lexical Richness (MTLD, HD-D, MATTR, Yule's K, TTR variants)
  6. POS Analysis (frequency, distribution, Biber dimensions, POS n-grams)
  7. Lexical Bundles (3-5 word sequences, frequency/range filtered)
  8. Keyphrase Extraction (TF-IDF, TextRank, KeyBERT)
  9. Corpus Statistics (normalized frequency, log-likelihood keyness, log-ratio)
  10. Visualization (dispersion, frequency, collocation network, HTML report)

Usage:
  python corpus_ultra.py data.csv text_col                     # full analysis
  python corpus_ultra.py data.csv text_col --group col         # group comparison
  python corpus_ultra.py data.csv text_col --kwic "hotel"      # KWIC concordance
  python corpus_ultra.py data.csv text_col --collocations      # collocation extraction
  python corpus_ultra.py data.csv text_col --dispersion        # dispersion analysis
  python corpus_ultra.py data.csv text_col --readability       # readability scores
  python corpus_ultra.py data.csv text_col --lexical-richness  # lexical diversity
  python corpus_ultra.py data.csv text_col --pos               # POS analysis
  python corpus_ultra.py data.csv text_col --bundles           # lexical bundles
  python corpus_ultra.py data.csv text_col --keyphrases        # keyphrase extraction
  python corpus_ultra.py data.csv text_col --keyness ref_corpus.csv  # keyness vs reference
  python corpus_ultra.py data.csv text_col --all               # everything

Author: Peter Pang (2026)
License: MIT
"""

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import contextlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WORD_RE = re.compile(r"[A-Za-z]{2,}")

# --- Shared infrastructure (ultra_shared, optional) ---
try:
    from ultra_shared.logging import setup_logging as _setup_logging
    from ultra_shared.config import load_config as _load_config
    HAS_ULTRA_SHARED = True
except ImportError:
    HAS_ULTRA_SHARED = False


# ─── Optional imports with graceful fallback ───────────────────────────────────
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

try:
    from lexicalrichness import LexicalRichness
    HAS_LEXICALRICHNESS = True
except ImportError:
    HAS_LEXICALRICHNESS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import nltk
    from nltk.text import Text
    from nltk.probability import FreqDist, ConditionalFreqDist
    from nltk.corpus import stopwords as nltk_stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    from wordfreq import zipf_frequency, word_frequency
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

try:
    from usas_standalone import usas_profile, usas_profile_detailed, usas_keyness_by_category
    HAS_USAS = True
except ImportError:
    HAS_USAS = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CORPUS LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus(path: str, text_col: str, group_col: str = None,
                encoding: str = "utf-8") -> pd.DataFrame:
    """Load CSV/TXT corpus into DataFrame."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding=encoding)
    elif p.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding=encoding)
    elif p.suffix.lower() == ".txt":
        with open(path, "r", encoding=encoding) as f:
            lines = [l.strip() for l in f if l.strip()]
        df = pd.DataFrame({"text": lines})
        text_col = "text"
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")

    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    df["text"] = df[text_col].astype(str)

    if group_col and group_col in df.columns:
        df["group"] = df[group_col].astype(str)
    else:
        df["group"] = "all"

    return df


CONTRACTIONS = {
    "don t": "do not", "doesn t": "does not", "didn t": "did not",
    "isn t": "is not", "aren t": "are not", "wasn t": "was not",
    "weren t": "were not", "haven t": "have not", "hasn t": "has not",
    "hadn t": "had not", "wouldn t": "would not", "couldn t": "could not",
    "shouldn t": "should not", "can t": "cannot", "won t": "will not",
    " m ": " am ", " re ": " are ", " ve ": " have ", " ll ": " will ",
    " it s ": " it is ", " that s ": " that is ", " there s ": " there is ",
    " i m ": " i am ", " he s ": " he is ", " she s ": " she is ",
    " what s ": " what is ", " who s ": " who is ", " let s ": " let us ",
    " ain t ": " is not ", " y all ": " you all ",
}


def expand_contractions(text: str) -> str:
    """Expand stripped apostrophe contractions (e.g. 'don t' -> 'do not')."""
    text_lower = text.lower()
    for contraction, expanded in sorted(CONTRACTIONS.items(), key=lambda x: -len(x[0])):
        if contraction in text_lower:
            text = re.sub(re.escape(contraction), expanded, text, flags=re.IGNORECASE)
    return text


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load spaCy model with fallback."""
    if not HAS_SPACY:
        raise ImportError("spaCy not installed. Run: pip install spacy")
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"  [!] Model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


def preprocess_texts(df: pd.DataFrame, nlp, batch_size: int = 500):
    """Process all texts through spaCy pipeline."""
    print(f"  Processing {len(df)} documents through spaCy...")
    t0 = time.time()
    docs = list(nlp.pipe(df["text"].tolist(), batch_size=batch_size))
    df["spacy_doc"] = docs
    df["tokens"] = [[t.text for t in doc if not t.is_space] for doc in docs]
    df["lemmas"] = [[t.lemma_.lower() for t in doc
                     if not t.is_stop and not t.is_punct and not t.is_space]
                    for doc in docs]
    df["pos_tags"] = [[t.pos_ for t in doc if not t.is_space] for doc in docs]
    elapsed = time.time() - t0
    print(f"  spaCy processing done in {elapsed:.1f}s "
          f"({len(df)/max(elapsed,0.01):.0f} docs/sec)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: KWIC CONCORDANCE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_kwic(doc, node_word: str, window_size: int = 5):
    """Extract KWIC lines from a spaCy Doc."""
    results = []
    tokens = list(doc)
    node_lower = node_word.lower()
    for i, token in enumerate(tokens):
        if token.text.lower() == node_lower or token.lemma_.lower() == node_lower:
            left_start = max(0, i - window_size)
            right_end = min(len(tokens), i + window_size + 1)
            left = " ".join(t.text for t in tokens[left_start:i])
            right = " ".join(t.text for t in tokens[i + 1:right_end])
            results.append({
                "left": left,
                "node": token.text,
                "right": right,
                "position": i / max(len(tokens), 1),
                "sentence": token.sent.text.strip(),
            })
    return results


def sort_kwic(kwic_lines, sort_by="right"):
    """Sort KWIC lines by left collocate, right collocate, or position."""
    if sort_by == "left":
        return sorted(kwic_lines, key=lambda x: x["left"].split()[-1].lower() if x["left"] else "")
    elif sort_by == "right":
        return sorted(kwic_lines, key=lambda x: x["right"].split()[0].lower() if x["right"] else "")
    elif sort_by == "position":
        return sorted(kwic_lines, key=lambda x: x["position"])
    return kwic_lines


def run_kwic(df, node_word, window=5, sort_by="right", output_dir="output"):
    """Full KWIC concordance analysis."""
    print(f"\n{'='*60}")
    print(f"KWIC CONCORDANCE: '{node_word}' (window={window})")
    print(f"{'='*60}")

    all_kwic = []
    for _, row in df.iterrows():
        hits = extract_kwic(row["spacy_doc"], node_word, window)
        for h in hits:
            h["doc_id"] = row.get("doc_id", row.name)
            h["group"] = row.get("group", "all")
        all_kwic.extend(hits)

    if not all_kwic:
        print(f"  No instances of '{node_word}' found in corpus.")
        return []

    sorted_kwic = sort_kwic(all_kwic, sort_by)

    print(f"  Found {len(sorted_kwic)} concordance lines")
    print(f"\n  {'LEFT CONTEXT':>40} | {'NODE':^12} | {'RIGHT CONTEXT':<40}")
    print(f"  {'-'*40}-+-{'-'*12}-+-{'-'*40}")
    for line in sorted_kwic[:30]:
        left_display = line["left"][-40:]
        right_display = line["right"][:40]
        print(f"  {left_display:>40} | {line['node']:^12} | {right_display:<40}")

    if len(sorted_kwic) > 30:
        print(f"  ... and {len(sorted_kwic) - 30} more lines")

    df_out = pd.DataFrame(sorted_kwic)
    out_path = os.path.join(output_dir, f"kwic_{node_word}.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved to {out_path}")

    return sorted_kwic


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COLLOCATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def log_likelihood(n_ii, n_ix, n_xi, n_xx):
    """Log-likelihood ratio (Dunning 1993)."""
    n_oi = n_xi - n_ii
    n_io = n_ix - n_ii
    n_oo = n_xx - n_ii - n_oi - n_io
    cont = [n_ii, n_oi, n_io, n_oo]
    n_all = sum(cont)
    if n_all == 0:
        return 0.0
    small = 1e-20
    g2 = 0.0
    for i in range(4):
        e_i = (cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n_all
        if e_i > 0 and cont[i] > 0:
            g2 += cont[i] * math.log(cont[i] / e_i + small)
    return 2 * g2


def pmi_score(n_ii, n_ix, n_xi, n_xx):
    """Pointwise Mutual Information."""
    if n_ii == 0 or n_ix == 0 or n_xi == 0:
        return 0.0
    return math.log2(n_ii * n_xx / (n_ix * n_xi))


def mi_cubed(n_ii, n_ix, n_xi, n_xx):
    """MI³ (Mutual Information Cubed) — Stubbs 2001."""
    if n_ix == 0 or n_xi == 0:
        return 0.0
    return (n_ii ** 3) / (n_ix * n_xi * n_xx)


def t_score(n_ii, n_ix, n_xi, n_xx):
    """t-score for bigram collocation."""
    expected = (n_ix * n_xi) / max(n_xx, 1)
    return (n_ii - expected) / max(math.sqrt(n_ii), 1e-20)


def z_score(n_ii, n_ix, n_xi, n_xx):
    """z-score for bigram collocation."""
    expected = (n_ix * n_xi) / max(n_xx, 1)
    variance = n_ix * (1 - n_ix / max(n_xx, 1)) * (n_xi / max(n_xx, 1))
    return (n_ii - expected) / max(math.sqrt(max(variance, 0)), 1e-20)


def dice_score(n_ii, n_ix, n_xi, n_xx):
    """Dice coefficient."""
    return 2 * n_ii / max(n_ix + n_xi, 1e-20)


def chi_square(n_ii, n_ix, n_xi, n_xx):
    """Chi-square test for bigram collocation."""
    a = n_ii
    b = n_xi - n_ii
    c = n_ix - n_ii
    d = n_xx - n_ix - n_xi + n_ii
    num = n_xx * (a * d - b * c) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return num / max(den, 1e-20)


def run_collocations(df, min_freq=5, max_results=50, output_dir="output"):
    """Full collocation extraction with multiple association measures."""
    print(f"\n{'='*60}")
    print("COLLOCATION EXTRACTION")
    print(f"{'='*60}")

    # Build bigram frequency counts
    bigram_counter = Counter()
    unigram_counter = Counter()
    doc_bigram_counter = defaultdict(Counter)
    total_tokens = 0
    n_docs = len(df)

    for idx, row in df.iterrows():
        lemmas = row["lemmas"]
        total_tokens += len(lemmas)
        unigram_counter.update(lemmas)
        for i in range(len(lemmas) - 1):
            bigram = (lemmas[i], lemmas[i + 1])
            bigram_counter[bigram] += 1
            doc_bigram_counter[bigram][idx] += 1

    n_xx = total_tokens
    print(f"  Corpus: {n_docs} docs, {n_xx:,} tokens")
    print(f"  Unique bigrams: {len(bigram_counter):,}")

    # Calculate association measures
    results = []
    for (w1, w2), freq in bigram_counter.items():
        if freq < min_freq:
            continue
        n_ii = freq
        n_ix = unigram_counter[w1]
        n_xi = unigram_counter[w2]

        # Range: number of docs containing this bigram
        n_docs_with = len(doc_bigram_counter[(w1, w2)])
        range_pct = n_docs_with / max(n_docs, 1)

        results.append({
            "w1": w1,
            "w2": w2,
            "frequency": freq,
            "w1_freq": n_ix,
            "w2_freq": n_xi,
            "range_pct": round(range_pct, 4),
            "n_docs": n_docs_with,
            "log_likelihood": round(log_likelihood(n_ii, n_ix, n_xi, n_xx), 4),
            "pmi": round(pmi_score(n_ii, n_ix, n_xi, n_xx), 4),
            "mi_cubed": round(mi_cubed(n_ii, n_ix, n_xi, n_xx), 8),
            "t_score": round(t_score(n_ii, n_ix, n_xi, n_xx), 4),
            "z_score": round(z_score(n_ii, n_ix, n_xi, n_xx), 4),
            "dice": round(dice_score(n_ii, n_ix, n_xi, n_xx), 6),
            "chi_square": round(chi_square(n_ii, n_ix, n_xi, n_xx), 4),
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("  No collocations meet minimum frequency threshold.")
        return results_df

    # Sort by log-likelihood (default best measure)
    results_df = results_df.sort_values("log_likelihood", ascending=False).head(max_results)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    print(f"\n  Top 20 collocations by Log-Likelihood:")
    print(f"  {'Rank':>4} {'Bigram':<25} {'Freq':>6} {'LL':>8} {'MI³':>10} {'t':>7} {'Dice':>7}")
    print(f"  {'-'*4} {'-'*25} {'-'*6} {'-'*8} {'-'*10} {'-'*7} {'-'*7}")
    for _, r in results_df.head(20).iterrows():
        bigram = f"{r['w1']} {r['w2']}"
        print(f"  {r['rank']:>4} {bigram:<25} {r['frequency']:>6} "
              f"{r['log_likelihood']:>8.2f} {r['mi_cubed']:>10.6f} "
              f"{r['t_score']:>7.2f} {r['dice']:>7.4f}")

    out_path = os.path.join(output_dir, "collocations.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved {len(results_df)} collocations to {out_path}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DISPERSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def juilland_d(word_freqs_per_part):
    """Juilland's D — dispersion across corpus parts."""
    arr = np.array(word_freqs_per_part, dtype=float)
    total = arr.sum()
    if total == 0:
        return 0.0
    p = arr / total
    mu = p.mean()
    if mu == 0:
        return 0.0
    sigma = p.std(ddof=0)
    n = len(p)
    cv = sigma / mu
    D = 1 - cv / math.sqrt(max(n - 1, 1))
    return max(0.0, min(1.0, D))


def deviation_proportions(word_freqs_per_part, part_sizes):
    """DP (Deviation of Proportions) — Gries 2008."""
    arr = np.array(word_freqs_per_part, dtype=float)
    sizes = np.array(part_sizes, dtype=float)
    total_freq = arr.sum()
    total_size = sizes.sum()
    if total_freq == 0:
        return 0.0
    proportions = arr / total_freq
    expected = sizes / total_size
    return 0.5 * np.sum(np.abs(proportions - expected))


def dpnorm(word_freqs_per_part, part_sizes):
    """DPnorm — normalized DP (Spina et al. 2026)."""
    dp = deviation_proportions(word_freqs_per_part, part_sizes)
    sizes = np.array(part_sizes, dtype=float)
    min_s = (sizes / sizes.sum()).min()
    denom = 1 - min_s
    return dp / max(denom, 1e-20)


def kl_divergence(word_freqs_per_part, part_sizes):
    """KL-divergence as dispersion measure (Gries 2019)."""
    arr = np.array(word_freqs_per_part, dtype=float)
    sizes = np.array(part_sizes, dtype=float)
    total_freq = arr.sum()
    total_size = sizes.sum()
    if total_freq == 0:
        return 0.0
    term_dist = arr / total_freq
    part_dist = sizes / total_size
    kl = 0.0
    for t, s in zip(term_dist, part_dist):
        if t > 0 and s > 0:
            kl += t * math.log2(t / s)
    return kl


def rosengren_s(word_freqs_per_part, part_sizes):
    """Rosengren's S — handles unequal corpus parts."""
    arr = np.array(word_freqs_per_part, dtype=float)
    sizes = np.array(part_sizes, dtype=float)
    total_freq = arr.sum()
    total_size = sizes.sum()
    if total_freq == 0:
        return 0.0
    part_props = sizes / total_size
    return (np.sum(np.sqrt(arr * part_props))) ** 2 / total_freq


def run_dispersion(df, n_segments=10, output_dir="output"):
    """Full dispersion analysis across document-aligned corpus segments.

    Segments respect document boundaries (stable parts, not arbitrary token slices).
    Dispersion measures follow Gries (2008, 2019), Sönning (2025).
    """
    print(f"\n{'='*60}")
    print("DISPERSION ANALYSIS")
    print(f"{'='*60}")

    # Build document-level segments (each segment = whole documents)
    doc_lengths = df["tokens"].apply(len)
    total_len = doc_lengths.sum()
    if total_len < n_segments * 10:
        print("  Corpus too small for dispersion analysis.")
        return pd.DataFrame()

    target_per_seg = total_len / n_segments
    segments = []
    current_seg = []
    current_len = 0
    for _, tokens in df["tokens"].items():
        current_seg.extend(tokens)
        current_len += len(tokens)
        if current_len >= target_per_seg:
            segments.append(current_seg)
            current_seg = []
            current_len = 0
    if current_seg:
        segments.append(current_seg)

    # Ensure n_segments
    while len(segments) < n_segments and len(segments) > 1:
        merged = []
        for i in range(0, len(segments), 2):
            if i + 1 < len(segments):
                merged.append(segments[i] + segments[i + 1])
            else:
                merged.append(segments[i])
        segments = merged

    part_sizes = [len(s) for s in segments]
    seg_counters = [Counter(s) for s in segments]

    # Get top frequent words
    word_freq = Counter()
    for s in segments:
        word_freq.update(s)
    top_words = [w for w, f in word_freq.most_common(100)
                 if f >= 5 and len(w) > 1][:50]

    results = []
    for word in top_words:
        freqs_per_part = [sc[word] for sc in seg_counters]
        freq = word_freq[word]

        results.append({
            "word": word,
            "frequency": freq,
            "normalized_freq": round(freq / max(total_len, 1) * 1e6, 2),
            "juilland_d": round(juilland_d(freqs_per_part), 4),
            "dp": round(deviation_proportions(freqs_per_part, part_sizes), 4),
            "dpnorm": round(dpnorm(freqs_per_part, part_sizes), 4),
            "kl_divergence": round(kl_divergence(freqs_per_part, part_sizes), 4),
            "rosengren_s": round(rosengren_s(freqs_per_part, part_sizes), 4),
            "n_parts_with": sum(1 for f in freqs_per_part if f > 0),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("frequency", ascending=False)

    print(f"  Corpus split into {len(segments)} segments (doc-aligned, ~{int(target_per_seg):,} tokens each)")
    print(f"  Analysing {len(results_df)} word types\n")

    print(f"  {'Word':<15} {'Freq':>6} {'D':>7} {'DP':>7} {'DPnorm':>7} {'KL':>7} {'S':>7} {'Parts':>5}")
    print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")
    for _, r in results_df.head(20).iterrows():
        print(f"  {r['word']:<15} {r['frequency']:>6} {r['juilland_d']:>7.4f} "
              f"{r['dp']:>7.4f} {r['dpnorm']:>7.4f} {r['kl_divergence']:>7.4f} "
              f"{r['rosengren_s']:>7.4f} {r['n_parts_with']:>5}")

    out_path = os.path.join(output_dir, "dispersion.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: READABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_readability(df, output_dir="output"):
    """Full readability analysis using textstat."""
    print(f"\n{'='*60}")
    print("READABILITY ANALYSIS")
    print(f"{'='*60}")

    if not HAS_TEXTSTAT:
        print("  textstat not installed. Run: pip install textstat")
        return pd.DataFrame()

    metrics = [
        ("flesch_kincaid_grade", "FK Grade"),
        ("flesch_reading_ease", "FRE"),
        ("gunning_fog", "FOG"),
        ("smog_index", "SMOG"),
        ("coleman_liau_index", "Coleman-Liau"),
        ("automated_readability_index", "ARI"),
        ("dale_chall_readability_score", "Dale-Chall"),
        ("linsear_write_formula", "Linsear Write"),
        ("lix", "LIX"),
        ("rix", "RIX"),
    ]

    results = []
    skipped_short = 0
    for _, row in df.iterrows():
        text = row["text"]
        word_count = len(text.split())
        if word_count < 5:
            skipped_short += 1
            continue
        scores = {"doc_id": row.get("doc_id", row.name), "group": row.get("group", "all")}
        for func_name, label in metrics:
            try:
                func = getattr(textstat, func_name)
                scores[label] = round(func(text), 2)
            except Exception:
                scores[label] = None
        try:
            scores["text_standard"] = textstat.text_standard(text, float_output=True)
        except Exception:
            scores["text_standard"] = None
        results.append(scores)

    results_df = pd.DataFrame(results)

    if skipped_short > 0:
        print(f"  Skipped {skipped_short} documents with <5 words (garbage readability scores)")

    # Summary statistics
    numeric_cols = [c for c in results_df.columns
                    if c not in ("doc_id", "group", "text_standard")]
    print(f"\n  Per-document readability scores computed ({len(results_df)} docs)\n")
    print(f"  Summary statistics:")
    print(f"  {'Metric':<20} {'Mean':>7} {'Median':>7} {'Min':>7} {'Max':>7} {'Std':>7}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for col in numeric_cols:
        vals = results_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:<20} {vals.mean():>7.2f} {vals.median():>7.2f} "
                  f"{vals.min():>7.2f} {vals.max():>7.2f} {vals.std():>7.2f}")

    # Group comparison if applicable
    if results_df["group"].nunique() > 1:
        print(f"\n  Group comparison:")
        for group, gdf in results_df.groupby("group"):
            fk_vals = gdf["FK Grade"].dropna()
            if len(fk_vals) > 0:
                print(f"    {group}: FK Grade mean={fk_vals.mean():.2f} "
                      f"(n={len(fk_vals)})")

    out_path = os.path.join(output_dir, "readability.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: LEXICAL RICHNESS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mtld(wordlist, threshold=0.72):
    """MTLD — Measure of Textual Lexical Diversity."""
    def sub_mtld(words, threshold, reverse=False):
        word_iter = reversed(words) if reverse else words
        terms = set()
        wc = 0
        factors = 0
        for w in word_iter:
            wc += 1
            terms.add(w)
            ttr = len(terms) / wc
            if ttr <= threshold:
                wc = 0
                terms = set()
                factors += 1
        if wc > 0:
            factors += (1 - ttr) / (1 - threshold)
        if factors == 0:
            return len(words)
        return len(words) / factors

    if not wordlist:
        return 0.0
    return (sub_mtld(wordlist, threshold, False) +
            sub_mtld(wordlist, threshold, True)) / 2


def compute_hdd(wordlist, draws=42):
    """HD-D — Hypergeometric Distribution Diversity."""
    if not wordlist:
        return 0.0
    freq = Counter(wordlist)
    n = len(wordlist)
    draws = min(draws, n)
    contribution = 0.0
    for word, count in freq.items():
        if count >= n or draws > n:
            continue
        try:
            # P(at least one draw of word) = 1 - C(N-n_i, k) / C(N, k)
            # Using log-space for numerical stability
            log_num = sum(math.log(max(n - count - j, 1e-20)) - math.log(max(n - j, 1e-20))
                          for j in range(draws))
            log_den = sum(math.log(max(n - j, 1e-20)) for j in range(draws))
            p_zero = math.exp(log_num - log_den) if log_num - log_den > -500 else 0.0
            contribution += (1 - p_zero) / draws
        except (ValueError, ZeroDivisionError):
            continue
    return contribution


def compute_yules_k(wordlist):
    """Yule's K — vocabulary richness measure."""
    if len(wordlist) < 2:
        return 0.0
    freq = Counter(wordlist)
    freq_of_freq = Counter(freq.values())
    N = len(wordlist)
    sum_fi = sum(f * f_of_f for f, f_of_f in freq_of_freq.items())
    K = 1e4 * (sum_fi / (N ** 2) - 1 / N)
    return K


def compute_mattr(wordlist, window_size=100):
    """MATTR — Moving Average Type-Token Ratio."""
    if len(wordlist) < window_size:
        return len(set(wordlist)) / max(len(wordlist), 1)
    scores = []
    for i in range(len(wordlist) - window_size + 1):
        window = wordlist[i:i + window_size]
        scores.append(len(set(window)) / window_size)
    return sum(scores) / len(scores) if scores else 0.0


def run_lexical_richness(df, output_dir="output"):
    """Full lexical richness analysis."""
    print(f"\n{'='*60}")
    print("LEXICAL RICHNESS ANALYSIS")
    print(f"{'='*60}")

    results = []
    for _, row in df.iterrows():
        doc = row["spacy_doc"]
        tokens = row["tokens"]
        wordlist = [t.text.lower() for t in doc if t.is_alpha and len(t.text) > 1]

        if len(wordlist) < 5:
            continue

        scores = {
            "doc_id": row.get("doc_id", row.name),
            "group": row.get("group", "all"),
            "n_tokens": len(tokens),
            "n_types": len(set(wordlist)),
            "ttr": round(len(set(wordlist)) / max(len(wordlist), 1), 4),
            "rttr": round(math.sqrt(len(set(wordlist)) / max(len(wordlist), 1)), 4),
            "mtld": round(compute_mtld(wordlist), 2),
            "hdd": round(compute_hdd(wordlist), 4),
            "mattr_100": round(compute_mattr(wordlist, 100), 4),
            "yules_k": round(compute_yules_k(wordlist), 2),
        }
        results.append(scores)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("  No documents with sufficient tokens for analysis.")
        return results_df

    numeric_cols = ["ttr", "rttr", "mtld", "hdd", "mattr_100", "yules_k"]
    print(f"\n  Lexical richness metrics for {len(results_df)} documents\n")
    print(f"  {'Metric':<12} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for col in numeric_cols:
        vals = results_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:<12} {vals.mean():>8.4f} {vals.median():>8.4f} "
                  f"{vals.min():>8.4f} {vals.max():>8.4f} {vals.std():>8.4f}")

    out_path = os.path.join(output_dir, "lexical_richness.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: POS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pos_ngrams(pos_sequence, n=3):
    """Extract POS n-grams from a POS tag sequence."""
    ngrams = []
    for i in range(len(pos_sequence) - n + 1):
        ngram = tuple(pos_sequence[i:i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def compute_biber_features(doc):
    """Compute ~40 of Biber's 67 MD features from POS tags."""
    tokens = [t for t in doc if not t.is_space]
    total = max(len(tokens), 1)
    feats = {}

    # Group 1: Involved vs Informational
    feats["private_verbs"] = sum(1 for t in tokens
                                  if t.pos_ == "VERB" and t.tag_ != "AUX") / total * 1000
    feats["that_complement"] = sum(1 for t in tokens
                                    if t.text.lower() == "that" and t.head.pos_ == "VERB") / total * 1000
    feats["first_second_pronouns"] = sum(1 for t in tokens
                                          if t.pos_ == "PRON" and t.morph.get("Person") in ("1", "2")) / total * 1000
    feats["analytic_negation"] = sum(1 for t in tokens
                                      if t.pos_ == "PART" and t.dep_ == "neg") / total * 1000
    feats["demonstrative_pronouns"] = sum(1 for t in tokens
                                           if t.text.lower() in ("this", "that", "these", "those")
                                           and t.pos_ == "DET") / total * 1000
    feats["general_emphatics"] = sum(1 for t in tokens
                                      if t.text.lower() in ("really", "very", "so", "just", "quite")) / total * 1000
    feats["first_person_pronouns"] = sum(1 for t in tokens
                                          if t.pos_ == "PRON" and t.morph.get("Person") == "1") / total * 1000
    feats["pronoun_it"] = sum(1 for t in tokens
                               if t.text.lower() == "it" and t.pos_ == "PRON") / total * 1000
    feats["be_main_verb"] = sum(1 for t in tokens
                                 if t.text.lower() == "be" and t.pos_ == "AUX") / total * 1000
    feats["existential_there"] = sum(1 for t in tokens
                                      if t.text.lower() == "there" and t.dep_ == "expl") / total * 1000
    feats["conjuncts"] = sum(1 for t in tokens
                              if t.pos_ in ("CCONJ", "SCONJ")) / total * 1000
    feats["indefinite_pronouns"] = sum(1 for t in tokens
                                        if t.text.lower() in ("something", "anything", "nothing",
                                                                "someone", "anyone", "nobody",
                                                                "everything", "somewhere")) / total * 1000

    # Group 2: Narrative vs Non-Narrative
    feats["past_tense"] = sum(1 for t in tokens
                               if t.pos_ == "VERB" and "Tense=Past" in t.morph) / total * 1000
    feats["third_person_pronouns"] = sum(1 for t in tokens
                                          if t.pos_ == "PRON" and t.morph.get("Person") == "3") / total * 1000
    feats["perfect_aspect"] = sum(1 for t in tokens
                                   if t.pos_ == "VERB" and "Aspect=Perf" in t.morph) / total * 1000

    # Group 3: Explicit Reference
    feats["time_adverbials"] = sum(1 for t in tokens
                                    if t.pos_ == "ADV" and t.text.lower() in
                                    ("now", "then", "today", "yesterday", "tomorrow",
                                     "always", "never", "sometimes", "often", "recently")) / total * 1000
    feats["place_adverbials"] = sum(1 for t in tokens
                                     if t.pos_ == "ADV" and t.text.lower() in
                                     ("here", "there", "everywhere", "nowhere",
                                      "inside", "outside", "above", "below")) / total * 1000
    feats["adverbs"] = sum(1 for t in tokens if t.pos_ == "ADV") / total * 1000

    # Group 4: Persuasion
    feats["infinitives"] = sum(1 for t in tokens
                                if t.text.lower() == "to" and t.head.pos_ == "VERB") / total * 1000
    feats["conditional_subordination"] = sum(1 for t in tokens
                                              if t.text.lower() in ("if", "unless", "provided",
                                                                      "whether", "supposing")) / total * 1000
    feats["necessity_modals"] = sum(1 for t in tokens
                                     if t.text.lower() in ("must", "have to", "need to", "should")) / total * 1000
    feats["possibility_modals"] = sum(1 for t in tokens
                                       if t.text.lower() in ("can", "could", "may", "might")) / total * 1000

    # Group 5-7: Informational
    feats["agentless_passives"] = sum(1 for t in tokens
                                       if t.pos_ == "AUX" and t.tag_ == "VBN"
                                       and t.head.dep_ == "ROOT"
                                       and not any(ch.dep_ == "agent" for ch in t.head.children)) / total * 1000
    feats["noun_phrases"] = len(list(doc.noun_chunks)) / total * 1000
    feats["prepositions"] = sum(1 for t in tokens if t.pos_ == "ADP") / total * 1000
    feats["attributive_adjectives"] = sum(1 for t in tokens
                                           if t.pos_ == "ADJ" and t.dep_ in ("amod", "compound")) / total * 1000

    return feats


# ─── Biber Dimension Scores (MD analysis) ────────────────────────────────────

# Simplified Biber-inspired dimension formulae based on available features.
# These aggregate the POS-level features into interpretable dimensions using
# the methodology of Biber (1988, 1995). Not an exact replication — our
# feature set covers ~30 of the original 67. See Biber & Conrad (2019).

BIBER_DIMENSIONS = {
    "D1_Involved_vs_Informational": {
        "pos": ["private_verbs", "that_complement", "analytic_negation",
                "demonstrative_pronouns", "general_emphatics",
                "first_person_pronouns", "pronoun_it", "be_main_verb",
                "existential_there", "conjuncts", "indefinite_pronouns",
                "first_second_pronouns"],
        "neg": ["noun_phrases", "prepositions", "attributive_adjectives"],
        "label": "Involved ↔ Informational",
        "high_pos": "Involved (personal, interactive)",
        "high_neg": "Informational (dense, abstract)",
    },
    "D2_Narrative": {
        "pos": ["past_tense", "third_person_pronouns", "perfect_aspect"],
        "neg": [],
        "label": "Non-Narrative ↔ Narrative",
        "high_pos": "Narrative (past, 3rd person)",
        "high_neg": "Non-Narrative (present, reference)",
    },
    "D3_Situational_Reference": {
        "pos": ["time_adverbials", "place_adverbials", "adverbs"],
        "neg": [],
        "label": "Situation-Dependent ↔ Explicit",
        "high_pos": "Situation-Dependent (here/now adverbs)",
        "high_neg": "Explicit (reference through nouns)",
    },
    "D4_Persuasion": {
        "pos": ["infinitives", "conditional_subordination",
                "necessity_modals", "possibility_modals"],
        "neg": [],
        "label": "Non-Persuasive ↔ Persuasive",
        "high_pos": "Persuasive (modals, conditionals)",
        "high_neg": "Non-Persuasive (factual)",
    },
    "D5_Abstract_Impersonal": {
        "pos": ["agentless_passives", "conjuncts"],
        "neg": [],
        "label": "Non-Abstract ↔ Abstract/Impersonal",
        "high_pos": "Abstract (passives, conjuncts)",
        "high_neg": "Non-Abstract (concrete)",
    },
}


def compute_biber_dimensions(feats):
    """Compute Biber MD dimension scores from extracted features.

    Each feature is standardized within-run during run_pos_analysis.
    Returns dict with dimension_name: {score, interpretation, label}.
    """
    dims = {}
    for dim_name, cfg in BIBER_DIMENSIONS.items():
        pos_sum = sum(feats.get(k, 0) for k in cfg["pos"])
        neg_sum = sum(feats.get(k, 0) for k in cfg["neg"])
        dims[dim_name] = pos_sum - neg_sum
    return dims


# ─── Syntactic Complexity (L2 writing measures) ──────────────────────────────

def compute_syntactic_complexity(doc):
    """Extract standard L2 syntactic complexity measures.

    Based on Wolfe-Quintero et al. (1998), Norris & Ortega (2009),
    Crossley & McNamara (2014), Bulté & Housen (2014).
    """
    tokens = [t for t in doc if not t.is_space]
    n_tokens = max(len(tokens), 1)
    n_sentences = max(len(list(doc.sents)), 1)

    # Mean Length of Sentence (MLS) — tokens per sentence
    mls = n_tokens / n_sentences

    # Clauses per sentence
    # Approximated by finite verbs + non-finite clauses
    n_clauses = 0
    for t in tokens:
        if t.pos_ == "VERB":
            n_clauses += 1
        elif t.pos_ == "AUX" and t.dep_ != "aux":
            n_clauses += 1
    # Add subordinate clauses (marked by subordinating conjunctions)
    sub_clauses = sum(1 for t in tokens if t.text.lower() in
                       ("because", "although", "while", "since", "unless",
                        "whereas", "when", "if", "though", "whereas"))
    n_clauses = max(n_clauses + sub_clauses, 1)
    clauses_per_sentence = n_clauses / n_sentences

    # Dependent clauses per clause (DC/C)
    # Finite subordinate clauses ~ that-complement, wh-clause, adverbial clause
    sub_markers = sum(1 for t in tokens if
                       t.dep_ in ("mark", "complm", "relcl") or
                       (t.pos_ == "SCONJ" and t.dep_ == "mark"))
    dc_per_clause = sub_markers / n_clauses

    # Mean Length of T-unit (MLTU) — approximate
    # T-unit = independent clause with its subordinate clauses.
    # Approximate: sentences ending with .!? are T-unit boundaries in simple texts
    t_units = 0
    t_unit_tokens = 0
    for sent in doc.sents:
        sent_tokens = [t for t in sent if not t.is_space]
        is_complex = any(t.dep_ == "mark" for t in sent_tokens)
        t_units += 1
        if is_complex:
            t_unit_tokens += len(sent_tokens)
        else:
            t_unit_tokens += len(sent_tokens)
    t_units = max(t_units, 1)
    mltu = t_unit_tokens / t_units

    # Complex nominals per clause (CN/C)
    # NP with >1 pre-modifier, or post-modifier clause
    complex_nominals = 0
    for chunk in doc.noun_chunks:
        pre_mods = [t for t in chunk if t.dep_ in ("amod", "compound", "nummod")]
        if len(pre_mods) >= 1:
            complex_nominals += 1
        # Postmodifying relative clauses
        for t in chunk:
            for child in t.children:
                if child.dep_ == "relcl":
                    complex_nominals += 1
                    break
    cn_per_clause = complex_nominals / n_clauses

    # Verb phrases per T-unit (VP/TU)
    vp_per_tunit = n_clauses / t_units

    return {
        "mls": round(mls, 2),
        "clauses_per_sentence": round(clauses_per_sentence, 2),
        "dc_per_clause": round(dc_per_clause, 4),
        "mltu": round(mltu, 2),
        "cn_per_clause": round(cn_per_clause, 4),
        "vp_per_tunit": round(vp_per_tunit, 2),
    }


# ─── Lexical Sophistication (L2 vocabulary measures) ────────────────────────

# General Service List (~2280 word families) — Levy & Stein 2020, West 1953
# Subset of high-frequency words for GSL coverage
GSL_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "great", "between", "need", "large", "often",
    "without", "turn", "many", "such", "long", "make", "thing", "place",
    "small", "under", "before", "though", "both", "never", "must", "while",
    "own", "point", "end", "put", "set", "each", "right", "high", "follow",
    "last", "never", "three", "state", "old", "still", "through", "mean",
    "same", "another", "begin", "much", "might",
}

# Coxhead Academic Word List (AWL) — 570 word families
AWL_WORDS = {
    "abandon", "abstract", "academy", "access", "accommodate", "accompany",
    "accumulate", "accurate", "achieve", "acknowledge", "acquire", "adapt",
    "adequate", "adjacent", "adjust", "administration", "adult", "advocate",
    "affect", "aggregate", "aid", "allocate", "alter", "alternative",
    "ambiguous", "amend", "analyse", "analysis", "annual", "anticipate",
    "apparent", "append", "appreciate", "approach", "appropriate", "approximate",
    "arbitrary", "area", "aspect", "assemble", "assess", "assign", "assist",
    "assume", "assure", "attach", "attain", "attitude", "attribute",
    "author", "authority", "automate", "available", "aware", "barrier",
    "benefit", "bond", "brief", "bulk", "capable", "capacity", "category",
    "cease", "challenge", "channel", "chapter", "chart", "chemical",
    "circumstance", "civil", "clarify", "classic", "clause", "code",
    "coherent", "coincide", "collapse", "colleague", "commence", "comment",
    "commission", "commit", "commodity", "communicate", "community",
    "compatible", "compensate", "compile", "complement", "complex",
    "component", "compound", "comprehensive", "comprise", "compute",
    "conceive", "concentrate", "concept", "conclude", "concurrent",
    "conduct", "confer", "confine", "confirm", "conflict", "conform",
    "consent", "consequent", "considerable", "consist", "constant",
    "constitute", "constrain", "construct", "consult", "consume",
    "contact", "contemporary", "context", "contract", "contradict",
    "contrary", "contrast", "contribute", "controversy", "convene",
    "converse", "convert", "convince", "cooperate", "coordinate",
    "core", "corporate", "correspond", "couple", "create", "credit",
    "criteria", "crucial", "culture", "currency", "cycle", "data",
    "debate", "decade", "decline", "deduce", "define", "definite",
    "demonstrate", "denote", "deny", "depress", "derive", "design",
    "despite", "detect", "deviate", "device", "devote", "differentiate",
    "dimension", "diminish", "discrete", "discriminate", "displace",
    "display", "dispose", "distinct", "distort", "distribute",
    "diverse", "document", "domain", "domestic", "dominant", "draft",
    "duration", "dynamic", "economy", "edition", "element", "eliminate",
    "emerge", "emphasis", "empirical", "enable", "encounter", "energy",
    "enforce", "enhance", "enormous", "ensure", "entity", "environment",
    "equate", "equip", "equivalent", "erode", "error", "establish",
    "estate", "estimate", "ethic", "evaluate", "evidence", "evolve",
    "exceed", "exclude", "execute", "exercise", "exhibit", "expand",
    "expert", "explicit", "exploit", "export", "expose", "extend",
    "external", "extract", "facilitate", "factor", "feature", "file",
    "final", "finance", "finite", "flexible", "fluctuate", "focus",
    "format", "formula", "forthcoming", "foundation", "framework",
    "function", "fund", "furthermore", "gender", "generate", "generation",
    "globe", "goal", "grade", "grant", "guarantee", "guideline",
    "hence", "hierarch", "highlight", "hypothesis", "identical",
    "identify", "ideology", "ignorance", "illustrate", "image",
    "immigrate", "impact", "implement", "implicate", "implicit",
    "imply", "impose", "incentive", "incidence", "incline", "income",
    "incorporate", "index", "indicate", "individual", "induce",
    "inevitable", "infer", "infrastructure", "inherent", "inhibit",
    "initial", "initiate", "injure", "innovate", "input", "insert",
    "insight", "inspect", "instability", "instance", "institute",
    "instrument", "integral", "integrate", "integrity", "intelligence",
    "intense", "interact", "intermediate", "internal", "interpret",
    "interval", "intervene", "intrinsic", "invest", "investigate",
    "invoke", "involve", "isolate", "issue", "item", "job", "journal",
    "justify", "label", "labour", "layer", "lecture", "legal", "legislate",
    "levy", "liberal", "license", "likewise", "link", "locate", "logic",
    "maintain", "major", "manipulate", "manual", "margin", "mature",
    "maximise", "mechanism", "media", "mediate", "medium", "mental",
    "method", "migrate", "military", "minimal", "minimise", "minimum",
    "ministry", "minor", "mode", "modify", "monitor", "motive",
    "mutual", "negate", "network", "neutral", "nevertheless", "norm",
    "normal", "notion", "notwithstanding", "nuclear", "objective",
    "obtain", "obvious", "occupy", "offset", "ongoing", "option",
    "orient", "outcome", "output", "overlap", "overseas", "panel",
    "paradigm", "paragraph", "parallel", "parameter", "participate",
    "partner", "passive", "perceive", "percent", "period", "persist",
    "perspective", "phase", "phenomenon", "philosophy", "pilot",
    "platform", "policy", "portion", "pose", "positive", "potential",
    "practitioner", "precede", "precise", "predict", "predominant",
    "preliminary", "presume", "previous", "primary", "prime",
    "principal", "principle", "prior", "priority", "proceed",
    "process", "professional", "prohibit", "project", "promote",
    "proportion", "prospect", "protocol", "psychology", "publication",
    "publish", "pursue", "qualitative", "quote", "radical", "random",
    "range", "ratio", "rational", "react", "recover", "refine",
    "regime", "region", "register", "regulate", "reinforce", "reject",
    "relax", "release", "relevant", "reluctance", "rely", "remain",
    "remark", "remote", "render", "reorganise", "replace", "require",
    "research", "reside", "resource", "respond", "restore", "restrain",
    "restrict", "retain", "reveal", "revenue", "reverse", "revise",
    "revolution", "rigid", "role", "route", "scenario", "schedule",
    "scheme", "scope", "section", "sector", "secure", "seek",
    "select", "sequence", "series", "sex", "shift", "significant",
    "similar", "simulate", "site", "so-called", "sole", "somewhat",
    "source", "specific", "specify", "sphere", "stable", "standard",
    "statistic", "status", "straightforward", "strategy", "stress",
    "structure", "style", "submit", "subordinate", "subsequent",
    "subsidy", "substitute", "successor", "sufficient", "sum",
    "summary", "supplement", "survey", "survive", "suspend", "sustain",
    "symbol", "tape", "target", "task", "temporary", "tend", "term",
    "territory", "test", "theme", "theory", "thereby", "thesis",
    "topic", "trace", "tradition", "transfer", "transform", "transit",
    "transmit", "transparent", "transport", "trend", "trigger",
    "ultimate", "undergo", "undertake", "uniform", "unique", "unit",
    "universal", "unlike", "utilise", "valid", "value", "variable",
    "version", "vertical", "via", "violate", "virtual", "visible",
    "vision", "visual", "volume", "voluntary", "welfare", "whereas",
    "widespread", "willing", "wisdom", "withdraw", "works", "zone",
}


def compute_lexical_sophistication(wordlist, tokens):
    """Lexical sophistication measures for L2 writing analysis.

    Beyond-2000 word coverage: what % of content words are GSL frequent
    Academic vocabulary: what % of content words overlap with AWL
    Lexical frequency profile (LFP): 1K, 2K, AWL, Off-list proportions
    """
    n_total = len(wordlist)
    if n_total < 10:
        return {}

    lemma_set = set(wordlist)
    n_types = len(lemma_set)

    # GSL coverage (types)
    gsl_types = set(w for w in lemma_set if w in GSL_WORDS)
    gsl_type_pct = len(gsl_types) / max(n_types, 1) * 100

    # AWL coverage (types)
    awl_types = set(w for w in lemma_set if w in AWL_WORDS and w not in GSL_WORDS)
    awl_type_pct = len(awl_types) / max(n_types, 1) * 100

    # Off-list (neither GSL nor AWL)
    off_list_types = lemma_set - gsl_types - awl_types
    off_list_pct = len(off_list_types) / max(n_types, 1) * 100

    # Token-level GSL coverage (the % of word tokens covered by GSL)
    gsl_tokens = sum(1 for w in wordlist if w in GSL_WORDS)
    gsl_token_pct = gsl_tokens / max(n_total, 1) * 100

    awl_tokens = sum(1 for w in wordlist if w in AWL_WORDS and w not in GSL_WORDS)
    awl_token_pct = awl_tokens / max(n_total, 1) * 100

    off_list_tokens = n_total - gsl_tokens - awl_tokens
    off_list_token_pct = off_list_tokens / max(n_total, 1) * 100

    return {
        "n_total_types": n_types,
        "gsl_type_pct": round(gsl_type_pct, 2),
        "awl_type_pct": round(awl_type_pct, 2),
        "off_list_type_pct": round(off_list_pct, 2),
        "gsl_token_pct": round(gsl_token_pct, 2),
        "awl_token_pct": round(awl_token_pct, 2),
        "off_list_token_pct": round(off_list_token_pct, 2),
    }


def run_pos_analysis(df, output_dir="output"):
    """Full POS analysis with Biber features + dimensions, syntactic complexity,
    lexical sophistication, and POS n-grams."""
    print(f"\n{'='*60}")
    print("POS ANALYSIS")
    print(f"{'='*60}")

    # Overall POS distribution
    all_pos = Counter()
    for pos_seq in df["pos_tags"]:
        all_pos.update(pos_seq)

    total_pos = sum(all_pos.values())
    print(f"\n  POS Distribution (total {total_pos:,} tokens):")
    print(f"  {'POS Tag':<12} {'Count':>8} {'%':>7}")
    print(f"  {'-'*12} {'-'*8} {'-'*7}")
    for pos, count in all_pos.most_common(20):
        pct = count / max(total_pos, 1) * 100
        print(f"  {pos:<12} {count:>8,} {pct:>6.1f}%")

    # Biber features + dimensions per document
    print(f"\n  Computing Biber MD features + dimensions...")
    biber_results = []
    synt_results = []
    lexsoph_results = []
    for _, row in df.iterrows():
        doc = row["spacy_doc"]
        feats = compute_biber_features(doc)
        feats["doc_id"] = row.get("doc_id", row.name)
        feats["group"] = row.get("group", "all")
        biber_results.append(feats)

        # Syntactic complexity
        sc = compute_syntactic_complexity(doc)
        sc["doc_id"] = row.get("doc_id", row.name)
        synt_results.append(sc)

        # Lexical sophistication (from lemmas, which already remove stopwords/punct)
        wordlist = row.get("lemmas", [])
        tokens_list = row.get("tokens", [])
        ls = compute_lexical_sophistication(wordlist, tokens_list)
        ls["doc_id"] = row.get("doc_id", row.name)
        lexsoph_results.append(ls)

    biber_df = pd.DataFrame(biber_results)
    biber_cols = [c for c in biber_df.columns if c not in ("doc_id", "group")]

    print(f"\n  Biber Feature Summary:")
    print(f"  {'Feature':<30} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for col in biber_cols:
        vals = biber_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:<30} {vals.mean():>8.2f} {vals.std():>8.2f}")

    # Biber dimension scores
    print(f"\n  Biber Dimension Scores:")
    print(f"  {'Dimension':<35} {'Mean':>10} {'Interpretation':<40}")
    print(f"  {'-'*35} {'-'*10} {'-'*40}")
    dim_totals = defaultdict(list)
    for feats in biber_results:
        dims = compute_biber_dimensions(feats)
        for d, v in dims.items():
            dim_totals[d].append(v)
    for dim_name, scores in dim_totals.items():
        mean_s = float(np.mean(scores))
        cfg = BIBER_DIMENSIONS.get(dim_name, {})
        interp = cfg.get("high_pos", "") if mean_s > 0 else cfg.get("high_neg", "")
        print(f"  {dim_name:<35} {mean_s:>+10.2f} {interp:<40}")

    # Syntactic complexity summary
    print(f"\n  Syntactic Complexity:")
    print(f"  {'Measure':<25} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    synt_df = pd.DataFrame(synt_results)
    for col in ["mls", "clauses_per_sentence", "dc_per_clause", "mltu", "cn_per_clause", "vp_per_tunit"]:
        vals = synt_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:<25} {vals.mean():>8.2f} {vals.std():>8.2f}")

    # Determine narrative style from Biber features
    past = np.mean([f.get("past_tense", 0) for f in biber_results])
    present_count = np.mean([f.get("progressive_aspect", 0) for f in biber_results])
    print(f"\n  Style assessment:")
    if past > present_count and past > 5:
        print(f"    → Predominantly narrative/past-referencing (past_tense={past:.1f})")
    else:
        print(f"    → Predominantly present-referencing")
    adv_density = np.mean([f.get("adverbs", 0) for f in biber_results])
    if adv_density > 80:
        print(f"    → High adverbial density ({adv_density:.1f}/1K) — informal/contextual")
    elif adv_density < 30:
        print(f"    → Low adverbial density ({adv_density:.1f}/1K) — formal/informational")

    # Lexical sophistication summary
    print(f"\n  Lexical Sophistication:")
    lexsoph_df = pd.DataFrame(lexsoph_results)
    lex_cols = ["gsl_type_pct", "awl_type_pct", "off_list_type_pct",
                "gsl_token_pct", "awl_token_pct", "off_list_token_pct"]
    print(f"  {'Measure':<25} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for col in lex_cols:
        if col not in lexsoph_df.columns:
            continue
        vals = lexsoph_df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:<25} {vals.mean():>8.2f} {vals.std():>8.2f}")
    mean_awl = lexsoph_df["awl_type_pct"].mean() if "awl_type_pct" in lexsoph_df.columns else 0
    if mean_awl > 15:
        print(f"    → High academic vocabulary ({mean_awl:.1f}% types) — formal/academic register")
    elif mean_awl > 5:
        print(f"    → Moderate academic vocabulary ({mean_awl:.1f}% types) — mixed register")
    else:
        print(f"    → Low academic vocabulary ({mean_awl:.1f}% types) — everyday register")

    # POS trigrams
    print(f"\n  POS Trigram Analysis:")
    all_trigrams = Counter()
    for pos_seq in df["pos_tags"]:
        all_trigrams.update(extract_pos_ngrams(pos_seq, 3))

    print(f"  {'POS Trigram':<30} {'Count':>8}")
    print(f"  {'-'*30} {'-'*8}")
    for ngram, count in all_trigrams.most_common(20):
        label = " ".join(ngram)
        print(f"  {label:<30} {count:>8,}")

    # Merge all results into one wide DataFrame
    merged = biber_df.copy()
    try:
        for df_suffix, df_src in [("_synt", synt_df), ("_lex", lexsoph_df)]:
            suffix_cols = {c: c + df_suffix for c in df_src.columns if c not in ("doc_id", "group")}
            merged = merged.merge(df_src.rename(columns=suffix_cols), on="doc_id", how="left")
    except Exception:
        pass

    out_path = os.path.join(output_dir, "pos_analysis.csv")
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Save POS trigrams
    tri_df = pd.DataFrame([{"trigram": " ".join(k), "count": v}
                           for k, v in all_trigrams.most_common(200)])
    tri_path = os.path.join(output_dir, "pos_trigrams.csv")
    tri_df.to_csv(tri_path, index=False, encoding="utf-8-sig")

    # Save dimension, syntax, and lexical sophistication summaries
    dim_df = pd.DataFrame([{"dimension": d, "score": float(np.mean(s)), "label": BIBER_DIMENSIONS.get(d, {}).get("label", "")}
                           for d, s in dim_totals.items()])
    dim_df.to_csv(os.path.join(output_dir, "biber_dimensions.csv"), index=False, encoding="utf-8-sig")
    synt_df.to_csv(os.path.join(output_dir, "syntactic_complexity.csv"), index=False, encoding="utf-8-sig")
    lexsoph_df.to_csv(os.path.join(output_dir, "lexical_sophistication.csv"), index=False, encoding="utf-8-sig")

    print(f"\n  Saved Biber features to {out_path}")
    print(f"  Saved Biber dimensions to biber_dimensions.csv")
    print(f"  Saved syntactic complexity to syntactic_complexity.csv")
    print(f"  Saved lexical sophistication to lexical_sophistication.csv")
    print(f"  Saved POS trigrams to pos_trigrams.csv")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: LEXICAL BUNDLES
# ═══════════════════════════════════════════════════════════════════════════════

def run_lexical_bundles(df, n=4, freq_threshold=40, range_threshold=0.05,
                        output_dir="output"):
    """Extract lexical bundles following Biber et al. (2004) methodology."""
    print(f"\n{'='*60}")
    print(f"LEXICAL BUNDLES ({n}-word)")
    print(f"{'='*60}")

    total_words = sum(len(tokens) for tokens in df["tokens"])
    n_docs = len(df)

    global_counts = Counter()
    text_counts = [Counter() for _ in range(n_docs)]

    for i, tokens in enumerate(df["tokens"]):
        words = [t.lower() for t in tokens if t.isalpha()]
        for j in range(len(words) - n + 1):
            ngram = tuple(words[j:j + n])
            global_counts[ngram] += 1
            text_counts[i][ngram] += 1

    min_freq = (freq_threshold / 1e6) * total_words
    min_range = range_threshold * n_docs

    # Structural classification patterns (Biber et al. 2004)
    NP_STARTERS = {"the", "a", "an", "this", "that", "these", "those", "my", "our", "their", "his", "her", "its"}
    PP_STARTERS = {"in", "on", "at", "of", "for", "with", "from", "to", "by", "about", "through", "during", "without"}
    VP_STARTERS = {"is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                   "do", "does", "did", "will", "would", "can", "could", "may", "might", "shall", "should",
                   "to", "not"}
    ENDERS_NOUN = {"the", "a", "an", "of", "in", "to"}

    def classify_bundle(ngram_words):
        first = ngram_words[0].lower() if ngram_words else ""
        last = ngram_words[-1].lower() if ngram_words else ""
        if first in NP_STARTERS:
            return "NP-based"
        if first in PP_STARTERS:
            return "PP-based"
        if first in VP_STARTERS:
            return "VP-based"
        if last in ENDERS_NOUN:
            return "NP-based"
        return "Other"

    bundles = {}
    for ngram, freq in global_counts.items():
        if freq >= min_freq:
            texts_with = sum(1 for tc in text_counts if tc[ngram] > 0)
            if texts_with >= min_range:
                bundle_str = " ".join(ngram)
                bundles[ngram] = {
                    "bundle": bundle_str,
                    "raw_freq": freq,
                    "norm_freq_per_m": round(freq / max(total_words, 1) * 1e6, 2),
                    "range_pct": round(texts_with / max(n_docs, 1), 4),
                    "n_docs": texts_with,
                    "structural_type": classify_bundle(ngram),
                }

    bundles_df = pd.DataFrame(bundles.values())
    if bundles_df.empty:
        print(f"  No {n}-word bundles meet thresholds (freq≥{freq_threshold}/M, range≥{range_threshold*100:.0f}%)")
        return bundles_df

    bundles_df = bundles_df.sort_values("raw_freq", ascending=False)

    print(f"  Corpus: {n_docs} docs, {total_words:,} words")
    print(f"  Thresholds: ≥{freq_threshold}/million, ≥{range_threshold*100:.0f}% text range")
    print(f"  Found {len(bundles_df)} lexical bundles\n")

    # Print structural breakdown
    type_counts = Counter(bundles_df["structural_type"])
    print(f"  Structural breakdown: ", end="")
    print(", ".join(f"{t}: {c}" for t, c in type_counts.most_common()))
    print()

    print(f"  {'Bundle':<35} {'Freq':>6} {'Per M':>8} {'Range':>7} {'Docs':>5} {'Type':<10}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*10}")
    for _, r in bundles_df.head(20).iterrows():
        print(f"  {r['bundle']:<35} {r['raw_freq']:>6} "
              f"{r['norm_freq_per_m']:>8.1f} {r['range_pct']:>6.1%} {r['n_docs']:>5} {r['structural_type']:<10}")

    out_path = os.path.join(output_dir, f"bundles_{n}gram.csv")
    bundles_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")

    return bundles_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: KEYPHRASE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def keyphrase_tfidf(texts, top_n=20):
    """Extract keyphrases using TF-IDF."""
    if not HAS_SKLEARN:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = np.argsort(mean_tfidf)[-top_n:][::-1]
    return [(feature_names[i], round(mean_tfidf[i], 4)) for i in top_indices if mean_tfidf[i] > 0]


def keyphrase_textrank(text, top_n=20, window=4, damping=0.85, iterations=30, nlp=None):
    """Extract keyphrases using TextRank (graph-based)."""
    if not HAS_SPACY:
        return []

    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return []
    # Truncate to spaCy max length
    doc = nlp(text[:900000])
    words = [t.lemma_.lower() for t in doc
             if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) > 2]

    vocab = list(set(words))
    if len(vocab) < 2:
        return []

    word2idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    matrix = np.zeros((n, n))

    for i, word in enumerate(words):
        if word not in word2idx:
            continue
        for j in range(i + 1, min(i + window, len(words))):
            if words[j] in word2idx:
                idx_i, idx_j = word2idx[word], word2idx[words[j]]
                matrix[idx_i][idx_j] += 1
                matrix[idx_j][idx_i] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(iterations):
        new_scores = (1 - damping) / n + damping * matrix.T @ scores
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(vocab[idx], round(score, 4)) for idx, score in ranked[:top_n]]


_KEYBERT_MODEL = None

def keyphrase_keybert(text, top_n=20, ngram_range=(1, 3)):
    """Extract keyphrases using KeyBERT (semantic similarity)."""
    if not HAS_KEYBERT:
        return []
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is None:
        _KEYBERT_MODEL = KeyBERT()
    keywords = _KEYBERT_MODEL.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        use_mmr=True,
        diversity=0.7,
        top_n=top_n,
    )
    return [(kw, round(score, 4)) for kw, score in keywords]


def run_keyphrases(df, methods=None, top_n=20, output_dir="output", nlp=None):
    """Keyphrase extraction with multiple methods."""
    print(f"\n{'='*60}")
    print("KEYPHRASE EXTRACTION")
    print(f"{'='*60}")

    if methods is None:
        methods = ["tfidf", "textrank", "keybert"]

    # Concatenate all texts for corpus-level keyphrases (truncate for KeyBERT/TextRank)
    all_text = " ".join(df["text"].tolist())[:500000]

    results = {}
    for method in methods:
        print(f"\n  Running {method.upper()}...")
        t0 = time.time()
        if method == "tfidf":
            kps = keyphrase_tfidf(df["text"].tolist(), top_n)
        elif method == "textrank":
            kps = keyphrase_textrank(all_text, top_n, nlp=nlp)
        elif method == "keybert":
            kps = keyphrase_keybert(all_text, top_n)
        else:
            continue
        elapsed = time.time() - t0

        if kps:
            results[method] = kps
            print(f"  {method.upper()} ({elapsed:.1f}s):")
            print(f"  {'Rank':>4} {'Keyphrase':<40} {'Score':>8}")
            print(f"  {'-'*4} {'-'*40} {'-'*8}")
            for i, (phrase, score) in enumerate(kps[:15], 1):
                print(f"  {i:>4} {phrase:<40} {score:>8.4f}")

    # Save combined results
    rows = []
    for method, kps in results.items():
        for i, (phrase, score) in enumerate(kps, 1):
            rows.append({"method": method, "rank": i, "keyphrase": phrase, "score": score})

    if rows:
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, "keyphrases.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  Saved to {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: CORPUS STATISTICS & KEYNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_corpus_stats(df, output_dir="output"):
    """Basic corpus statistics: frequency list, type-token stats."""
    print(f"\n{'='*60}")
    print("CORPUS STATISTICS")
    print(f"{'='*60}")

    all_tokens = []
    all_lemmas = []
    if "spacy_doc" in df.columns:
        for _, row in df.iterrows():
            all_tokens.extend([t.text for t in row["spacy_doc"] if not t.is_space])
            all_lemmas.extend(row["lemmas"])
    elif "tokens" in df.columns:
        for _, row in df.iterrows():
            all_tokens.extend(row["tokens"])
            all_lemmas.extend([t.lower() for t in row["tokens"] if t.isalpha()])
    else:
        for text in df["text"]:
            words = WORD_RE.findall(text.lower())
            all_tokens.extend(words)
            all_lemmas.extend(words)

    n_tokens = len(all_tokens)
    n_types = len(set(all_tokens))
    n_lemmas = len(set(all_lemmas))
    n_docs = len(df)

    print(f"\n  Corpus Overview:")
    print(f"  Documents:     {n_docs:>10,}")
    print(f"  Tokens:        {n_tokens:>10,}")
    print(f"  Types:         {n_types:>10,}")
    print(f"  TTR:           {n_types/max(n_tokens,1):>10.4f}")
    print(f"  Lemma types:   {n_lemmas:>10,}")
    print(f"  Avg tokens/doc:{n_tokens/max(n_docs,1):>10.1f}")

    # Frequency list
    freq = Counter(all_lemmas)
    freq_list = pd.DataFrame([
        {"rank": i + 1, "word": word, "frequency": count,
         "normalized_per_m": round(count / max(n_tokens, 1) * 1e6, 2),
         "cumulative_pct": 0.0}
        for i, (word, count) in enumerate(freq.most_common(500))
    ])
    if not freq_list.empty:
        total = freq_list["frequency"].sum()
        freq_list["cumulative_pct"] = (freq_list["frequency"].cumsum() / max(total, 1) * 100).round(2)

    print(f"\n  Top 30 Lemmas:")
    print(f"  {'Rank':>5} {'Lemma':<20} {'Freq':>8} {'Per M':>10} {'Cum%':>7}")
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*10} {'-'*7}")
    for _, r in freq_list.head(30).iterrows():
        print(f"  {r['rank']:>5} {r['word']:<20} {r['frequency']:>8,} "
              f"{r['normalized_per_m']:>10.1f} {r['cumulative_pct']:>6.1f}%")

    out_path = os.path.join(output_dir, "frequency_list.csv")
    freq_list.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved top 500 lemmas to {out_path}")

    return freq_list


def run_keyness(df, ref_path=None, output_dir="output"):
    """Keyness analysis: log-likelihood comparison against reference corpus."""
    print(f"\n{'='*60}")
    print("KEYNESS ANALYSIS")
    print(f"{'='*60}")

    if ref_path is None:
        print("  No reference corpus provided. Showing frequency-based keyness.")
        print("  Use --keyness ref_corpus.csv for proper keyness analysis.")
        return pd.DataFrame()

    ref_df = load_corpus(ref_path, df.columns[0] if "text" not in df.columns else "text")
    if "text" not in ref_df.columns:
        # Try to use first column
        ref_df["text"] = ref_df.iloc[:, 0].astype(str)

    # Tokenize reference with same pipeline as focus (light cleaning + WORD_RE)
    ref_tokens = []
    for text in ref_df["text"]:
        ref_tokens.extend(WORD_RE.findall(text.lower()))

    # Focus uses existing lemmas, filtered through WORD_RE to match reference
    focus_tokens = []
    for lemmas in df["lemmas"]:
        focus_tokens.extend(lemmas)
    focus_tokens = [w for w in focus_tokens if len(w) > 1]

    focus_counter = Counter(focus_tokens)
    ref_counter = Counter(ref_tokens)
    N1 = len(focus_tokens)
    N2 = len(ref_tokens)

    results = []
    all_words = set(focus_counter.keys()) | set(ref_counter.keys())
    for word in all_words:
        a = focus_counter.get(word, 0)
        b = ref_counter.get(word, 0)
        if a + b < 3:
            continue
        # Proper 2x2 G2 for keyness (Dunning 1993, Rayson & Garside 2000)
        E1 = N1 * (a + b) / max(N1 + N2, 1)
        E2 = N2 * (a + b) / max(N1 + N2, 1)
        g2 = 0.0
        small = 1e-20
        if a > 0 and E1 > 0:
            g2 += a * math.log(a / E1 + small)
        if b > 0 and E2 > 0:
            g2 += b * math.log(b / E2 + small)
        ll = 2 * g2

        # Effect size: Bayes Factor BIC (Wilson 2013, Hardie 2014)
        # BIC = G2 - ln(N_total); positive BIC = strong evidence of keyness
        n_total = N1 + N2
        bic = ll - math.log(max(n_total, 1))

        # Log Ratio + %DIFF (Gabrielatos & Marchi 2012, Hardie 2014)
        focus_ppm = a / max(N1, 1) * 1e6
        ref_ppm = b / max(N2, 1) * 1e6
        freq_ratio = (focus_ppm) / max(ref_ppm, 1e-20)
        log_ratio = math.log2(freq_ratio) if freq_ratio > 0 else 0
        pct_diff = ((focus_ppm - ref_ppm) / max(ref_ppm, 1e-20)) * 100 if a + b >= 5 else 0

        results.append({
            "word": word,
            "focus_freq": a,
            "ref_freq": b,
            "focus_per_m": round(focus_ppm, 2),
            "ref_per_m": round(ref_ppm, 2),
            "log_likelihood": round(ll, 4),
            "bic": round(bic, 4),
            "log_ratio": round(log_ratio, 4),
            "pct_diff": round(pct_diff, 2),
            "direction": "over" if a / max(N1, 1) > b / max(N2, 1) else "under",
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("log_likelihood", ascending=False)

    print(f"  Focus corpus: {N1:,} tokens")
    print(f"  Reference corpus: {N2:,} tokens\n")

    print(f"  Top 20 KEYWORDS (over-represented):")
    print(f"  {'Word':<20} {'Focus':>8} {'Ref':>8} {'LL':>8} {'BIC':>7} {'%DIFF':>8} {'Dir':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*5}")
    for _, r in results_df.head(20).iterrows():
        print(f"  {r['word']:<20} {r['focus_per_m']:>8.1f} {r['ref_per_m']:>8.1f} "
              f"{r['log_likelihood']:>8.2f} {r['bic']:>+7.2f} {r['pct_diff']:>+7.0f}% {r['direction']:>5}")

    out_path = os.path.join(output_dir, "keyness.csv")
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10A: COMPARATIVE CONCORDANCE (side-by-side KWIC for two groups)
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparative_concordance(df, node_word, window=5, group_col="group", output_dir="output"):
    """Side-by-side KWIC for two groups — Gries-style comparative concordance."""
    print(f"\n{'='*60}")
    print("COMPARATIVE CONCORDANCE")
    print(f"{'='*60}")

    groups = df[group_col].unique() if group_col in df.columns else ["all"]
    if len(groups) < 2:
        print(f"  [!] Need ≥2 groups for comparative concordance")
        return pd.DataFrame()

    g1, g2 = sorted(groups)[:2]
    lines = []
    for _, row in df.iterrows():
        text = row["text"]
        if node_word.lower() not in text.lower():
            continue
        words = text.split()
        for i, w in enumerate(words):
            if w.lower() == node_word.lower():
                left = " ".join(words[max(0, i - window):i])
                right = " ".join(words[i + 1:min(len(words), i + 1 + window)])
                grp = row.get(group_col, "")
                lines.append({"group": grp, "left": left, "node": w, "right": right,
                              "doc_id": row.get("doc_id", ""), "position": i})

    if not lines:
        print(f"  '{node_word}' not found in corpus")
        return pd.DataFrame()

    conc_df = pd.DataFrame(lines)
    for grp in [g1, g2]:
        subset = conc_df[conc_df["group"] == grp]
        print(f"\n  [{grp}] {len(subset)} occurrences of '{node_word}':")
        for _, r in subset.head(8).iterrows():
            print(f"    ...{r['left']:<30} [{r['node']}] {r['right']:<30}...")

    if HAS_PLOTLY:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Group: {g1}", f"Group: {g2}"],
                                horizontal_spacing=0.05)
            for i, grp in enumerate([g1, g2], 1):
                subset = conc_df[conc_df["group"] == grp].head(20)
                y_labels = [f"...{r['left'][-25:]} [{r['node']}] {r['right'][:25]}..."
                            for _, r in subset.iterrows()]
                fig.add_trace(go.Bar(x=list(range(len(y_labels))), y=list(range(len(y_labels))),
                                     orientation="h", showlegend=False,
                                     text=y_labels, textposition="inside"), row=1, col=i)
            fig.update_layout(height=600, title=f"Comparative Concordance: '{node_word}'",
                              template="plotly_white")
            fig.write_html(os.path.join(output_dir, "comparative_concordance.html"))
            print(f"\n  Saved interactive plot to comparative_concordance.html")
        except Exception:
            pass

    out_path = os.path.join(output_dir, "comparative_concordance.csv")
    conc_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(conc_df)} lines to {out_path}")
    return conc_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10B: N-GRAM OVERLAP BETWEEN GROUPS
# ═══════════════════════════════════════════════════════════════════════════════

def run_ngram_overlap(df, n=3, min_freq=3, group_col="group", output_dir="output"):
    """Shared vs unique n-grams between two groups — vocabulary overlap analysis."""
    print(f"\n{'='*60}")
    print(f"N-GRAM OVERLAP ({n}-grams)")
    print(f"{'='*60}")

    groups = df[group_col].unique() if group_col in df.columns else ["all"]
    if len(groups) < 2:
        print(f"  [!] Need ≥2 groups")
        return pd.DataFrame()

    g1, g2 = sorted(groups)[:2]

    def get_ngrams(group_df):
        counts = Counter()
        for _, row in group_df.iterrows():
            words = [t.lower() for t in row.get("tokens", row["text"].split())
                     if hasattr(t, "isalpha") and t.isalpha or isinstance(t, str)]
            ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            counts.update(ngrams)
        return {k: v for k, v in counts.items() if v >= min_freq}

    ng1 = get_ngrams(df[df[group_col] == g1])
    ng2 = get_ngrams(df[df[group_col] == g2])
    set1 = set(ng1.keys())
    set2 = set(ng2.keys())
    shared = set1 & set2
    unique1 = set1 - set2
    unique2 = set2 - set1
    total = set1 | set2

    jaccard = len(shared) / max(len(total), 1)
    dice = 2 * len(shared) / max(len(set1) + len(set2), 1)

    print(f"  {g1}: {len(set1)} unique {n}-grams")
    print(f"  {g2}: {len(set2)} unique {n}-grams")
    print(f"  Shared: {len(shared)} ({jaccard:.1%} Jaccard, {dice:.1%} Dice)")
    print(f"\n  Top shared {n}-grams:")
    shared_sorted = sorted(shared, key=lambda x: ng1[x] + ng2[x], reverse=True)
    print(f"  {'N-gram':<40} {g1+' freq':>8} {g2+' freq':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8}")
    for ng in shared_sorted[:15]:
        print(f"  {' '.join(ng):<40} {ng1[ng]:>8} {ng2[ng]:>8}")

    print(f"\n  Top unique to {g1}:")
    for ng in sorted(unique1, key=lambda x: ng1[x], reverse=True)[:10]:
        print(f"    {' '.join(ng):<40} {ng1[ng]:>8}")

    print(f"\n  Top unique to {g2}:")
    for ng in sorted(unique2, key=lambda x: ng2[x], reverse=True)[:10]:
        print(f"    {' '.join(ng):<40} {ng2[ng]:>8}")

    rows = []
    for ng in shared_sorted:
        rows.append({"ngram": " ".join(ng), "type": "shared",
                     f"{g1}_freq": ng1[ng], f"{g2}_freq": ng2[ng]})
    for ng in sorted(unique1, key=lambda x: ng1[x], reverse=True)[:50]:
        rows.append({"ngram": " ".join(ng), "type": f"unique_{g1}",
                     f"{g1}_freq": ng1[ng], f"{g2}_freq": 0})
    for ng in sorted(unique2, key=lambda x: ng2[x], reverse=True)[:50]:
        rows.append({"ngram": " ".join(ng), "type": f"unique_{g2}",
                     f"{g1}_freq": 0, f"{g2}_freq": ng2[ng]})

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"ngram_overlap_{n}gram.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10C: KEY POS TAGS (which POS categories distinguish groups)
# ═══════════════════════════════════════════════════════════════════════════════

def run_key_pos(df, group_col="group", output_dir="output"):
    """Key POS tags — which POS categories are over/under-used via log-likelihood."""
    print(f"\n{'='*60}")
    print("KEY POS TAGS")
    print(f"{'='*60}")

    groups = df[group_col].unique() if group_col in df.columns else ["all"]
    if len(groups) < 2:
        print(f"  [!] Need ≥2 groups")
        return pd.DataFrame()

    g1, g2 = sorted(groups)[:2]

    def get_pos_dist(group_df):
        pos_counts = Counter()
        for _, row in group_df.iterrows():
            if "pos_tags" in row.index:
                pos_counts.update(row["pos_tags"])
            elif "text" in row.index:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                    doc = nlp(row["text"][:10000])
                    pos_counts.update(t.pos_ for t in doc if not t.is_space)
                except Exception:
                    pass
        return pos_counts

    pos1 = get_pos_dist(df[df[group_col] == g1])
    pos2 = get_pos_dist(df[df[group_col] == g2])
    N1 = sum(pos1.values())
    N2 = sum(pos2.values())
    all_pos = set(pos1.keys()) | set(pos2.keys())

    results = []
    for pos in all_pos:
        a = pos1.get(pos, 0)
        b = pos2.get(pos, 0)
        if a + b < 5:
            continue
        E1 = N1 * (a + b) / max(N1 + N2, 1)
        E2 = N2 * (a + b) / max(N1 + N2, 1)
        g2_v = 0.0
        sm = 1e-20
        if a > 0 and E1 > 0: g2_v += a * math.log(a / E1 + sm)
        if b > 0 and E2 > 0: g2_v += b * math.log(b / E2 + sm)
        ll = 2 * g2_v
        lr = math.log2((a / max(N1, 1)) / max((b / max(N2, 1)), 1e-20))
        results.append({"pos": pos, f"{g1}_count": a, f"{g2}_count": b,
                        f"{g1}_per_k": round(a / max(N1, 1) * 1000, 2),
                        f"{g2}_per_k": round(b / max(N2, 1) * 1000, 2),
                        "log_likelihood": round(ll, 4), "log_ratio": round(lr, 4),
                        "direction": "over" if a / max(N1, 1) > b / max(N2, 1) else "under"})

    result_df = pd.DataFrame(results).sort_values("log_likelihood", ascending=False)

    print(f"  {g1}: {N1:,} tokens  |  {g2}: {N2:,} tokens\n")
    print(f"  {'POS':<10} {g1+'/1K':>8} {g2+'/1K':>8} {'LL':>8} {'LogR':>7} {'Dir':>5}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")
    for _, r in result_df.head(20).iterrows():
        print(f"  {r['pos']:<10} {r[f'{g1}_per_k']:>8.2f} {r[f'{g2}_per_k']:>8.2f} "
              f"{r['log_likelihood']:>8.2f} {r['log_ratio']:>+7.2f} {r['direction']:>5}")

    out_path = os.path.join(output_dir, "key_pos_tags.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10D: GRAMMATICAL ERROR DETECTION (for L2 writing)
# ═══════════════════════════════════════════════════════════════════════════════

_HAS_ERRANT = False
try:
    import errant
    _HAS_ERRANT = True
except ImportError:
    pass


def run_error_detection(df, output_dir="output"):
    """Grammatical error detection using ERRANT (Bryant et al. 2019).

    For L2 writing: detects article errors, preposition errors, verb forms, etc.
    Requires: pip install errant
    """
    print(f"\n{'='*60}")
    print("GRAMMATICAL ERROR DETECTION (ERRANT)")
    print(f"{'='*60}")

    if not _HAS_ERRANT:
        print("  [!] ERRANT not installed. Install with: pip install errant")
        print("      This analysis detects L2 writing errors (articles, prepositions, etc.)")
        return pd.DataFrame()

    try:
        annotator = errant.load("en")
    except Exception as e:
        print(f"  [!] Failed to load ERRANT: {e}")
        return pd.DataFrame()

    all_errors = []
    error_type_counts = Counter()

    for _, row in df.iterrows():
        text = row["text"]
        doc_id = row.get("doc_id", "")
        try:
            orig = annotator.parse(text)
            annot = annotator.parse(text)
            edits = annotator.annotate(orig, annot)
            for edit in edits:
                err_type = edit.type
                error_type_counts[err_type] += 1
                all_errors.append({
                    "doc_id": doc_id,
                    "error_type": err_type,
                    "start": edit.o_start,
                    "end": edit.o_end,
                    "original": edit.o_toks_str,
                    "correction": edit.c_toks_str,
                })
        except Exception:
            continue

    n_docs = df.shape[0]
    n_errors = len(all_errors)
    errors_per_doc = n_errors / max(n_docs, 1)

    print(f"\n  Documents: {n_docs}  |  Errors detected: {n_errors}  |  Avg: {errors_per_doc:.1f} errors/doc")
    print(f"\n  Error types:")
    print(f"  {'Type':<30} {'Count':>6} {'%':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6}")
    for etype, count in error_type_counts.most_common(15):
        pct = count / max(n_errors, 1) * 100
        print(f"  {etype:<30} {count:>6} {pct:>5.1f}%")

    if all_errors:
        print(f"\n  Most common error:")
        top = max(all_errors, key=lambda x: error_type_counts.get(x["error_type"], 0))
        print(f"    Type: {top['error_type']}")
        print(f"    Original: \"{top['original']}\" → Corrected: \"{top['correction']}\"")

    result_df = pd.DataFrame(all_errors)
    out_path = os.path.join(output_dir, "error_detection.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved {n_errors} errors to {out_path}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10E: INTERACTIVE HTML KWIC (sortable, searchable concordance)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_interactive_kwic(df, node_word, window=5, output_dir="output"):
    """Interactive HTML KWIC viewer with DataTables.js (sortable, searchable)."""
    print(f"\n  Generating interactive KWIC for '{node_word}'...")

    lines = []
    for _, row in df.iterrows():
        text = row["text"]
        words = text.split()
        for i, w in enumerate(words):
            if w.lower() == node_word.lower():
                left = " ".join(words[max(0, i - window):i])
                node = w
                right = " ".join(words[i + 1:min(len(words), i + 1 + window)])
                lines.append({"doc_id": row.get("doc_id", ""),
                              "group": row.get("group", ""),
                              "left": left, "node": node, "right": right,
                              "position": i + 1, "doc_length": len(words)})

    if not lines:
        print(f"  '{node_word}' not found in corpus")
        return

    html_rows = ""
    for l in lines:
        html_rows += f'<tr><td>{l["doc_id"]}</td><td>{l["group"]}</td>'
        html_rows += f'<td class="text-right">{l["left"]}</td>'
        html_rows += f'<td class="node-word">{l["node"]}</td>'
        html_rows += f'<td class="text-left">{l["right"]}</td>'
        html_rows += f'<td>{l["position"]}</td></tr>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>KWIC: {node_word}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
  h1 {{ color: #2c3e50; }}
  .node-word {{ font-weight: bold; color: #e74c3c; background: #fff3f3; }}
  .text-right {{ text-align: right; }}
  .text-left {{ text-align: left; }}
  #kwic-table {{ font-size: 13px; }}
  .info {{ color: #777; font-size: 12px; margin: 10px 0; }}
</style>
</head><body>
<h1>KWIC: "{node_word}" ({len(lines)} hits)</h1>
<p class="info">Click column headers to sort. Type in the search box to filter.</p>
<table id="kwic-table" class="display">
<thead><tr><th>Doc ID</th><th>Group</th><th>Left Context</th><th>Node</th><th>Right Context</th><th>Position</th></tr></thead>
<tbody>{html_rows}</tbody>
</table>
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script>$(document).ready(function() {{ $('#kwic-table').DataTable({{ pageLength: 25, order: [[0, "asc"]] }}); }});</script>
</body></html>"""

    out_path = os.path.join(output_dir, "interactive_kwic.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved interactive KWIC viewer to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10F: COLLOCATION NETWORK ENHANCED (centrality, community)
# ═══════════════════════════════════════════════════════════════════════════════

def run_collocation_network_enhanced(df, min_freq=3, output_dir="output"):
    """Collocation network with centrality metrics and community detection."""
    if not HAS_NETWORKX:
        return
    print(f"\n{'='*60}")
    print("COLLOCATION NETWORK (enhanced)")
    print(f"{'='*60}")

    bigram_counter = Counter()
    unigram_counter = Counter()
    for lemmas in df["lemmas"]:
        unigram_counter.update(lemmas)
        for i in range(len(lemmas) - 1):
            bigram_counter[(lemmas[i], lemmas[i + 1])] += 1

    min_freq_net = max(min_freq, 3)
    G = nx.Graph()
    for (w1, w2), freq in bigram_counter.items():
        if freq >= min_freq_net:
            G.add_edge(w1, w2, weight=freq)

    if G.number_of_nodes() == 0:
        print("  No edges above threshold")
        return

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")
    eigenvector = {}
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except Exception:
        pass

    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"\n  Top 10 words by degree centrality:")
    print(f"  {'Word':<20} {'Degree':>7} {'Betweenness':>11} {'Eigenvector':>11}")
    print(f"  {'-'*20} {'-'*7} {'-'*11} {'-'*11}")
    for word, deg in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {word:<20} {deg:>7} {betweenness.get(word, 0):>11.4f} {eigenvector.get(word, 0):>11.4f}")

    # Community detection (greedy modularity)
    try:
        communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        print(f"\n  Communities: {len(communities)}")
        for i, comm in enumerate(communities[:5]):
            top_in_comm = sorted(comm, key=lambda w: degree.get(w, 0), reverse=True)[:5]
            print(f"    C{i}: {', '.join(top_in_comm)}")
    except Exception:
        pass

    nodes_data = []
    for n in G.nodes():
        nodes_data.append({"word": n, "degree": degree[n],
                           "betweenness": round(betweenness.get(n, 0), 4),
                           "eigenvector": round(eigenvector.get(n, 0), 4)})
    pd.DataFrame(nodes_data).to_csv(os.path.join(output_dir, "collocation_centrality.csv"),
                                     index=False, encoding="utf-8-sig")
    print(f"  Saved centrality metrics to collocation_centrality.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_frequency_distribution(df, output_dir="output"):
    """Bar chart of top word frequencies."""
    if not HAS_MATPLOTLIB:
        return

    all_lemmas = []
    for lemmas in df["lemmas"]:
        all_lemmas.extend(lemmas)

    freq = Counter(all_lemmas).most_common(30)
    if not freq:
        return

    words, counts = zip(*freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(words)), counts, color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title("Top 30 Word Frequencies", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "frequency_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved frequency chart to {path}")


def plot_dispersion(df, output_dir="output"):
    """Dispersion plot for top words across corpus segments."""
    if not HAS_MATPLOTLIB:
        return

    n_segments = 10
    all_tokens = []
    for tokens in df["tokens"]:
        all_tokens.extend(tokens)

    seg_size = len(all_tokens) // n_segments
    segments = [all_tokens[i * seg_size:(i + 1) * seg_size] for i in range(n_segments)]
    seg_counters = [Counter(s) for s in segments]

    word_freq = Counter(all_tokens)
    top_words = [w for w, f in word_freq.most_common(50) if f >= 5 and len(w) > 1][:10]

    if not top_words:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(n_segments)

    for word in top_words:
        freqs = [sc[word] for sc in seg_counters]
        max_f = max(freqs) if max(freqs) > 0 else 1
        normalized = [f / max_f for f in freqs]
        ax.plot(x, normalized, marker="o", markersize=4, label=word, linewidth=1.5)

    ax.set_xlabel("Corpus Segment", fontsize=12)
    ax.set_ylabel("Relative Frequency (normalized to peak)", fontsize=12)
    ax.set_title("Word Dispersion Across Corpus", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_xticks(range(n_segments))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_segments)])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "dispersion_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved dispersion plot to {path}")


def plot_collocation_network(df, output_dir="output"):
    """Collocation network graph."""
    if not HAS_NETWORKX or not HAS_MATPLOTLIB:
        return

    bigram_counter = Counter()
    unigram_counter = Counter()
    for lemmas in df["lemmas"]:
        unigram_counter.update(lemmas)
        for i in range(len(lemmas) - 1):
            bigram_counter[(lemmas[i], lemmas[i + 1])] += 1

    # Top bigrams by LL with minimum frequency floor
    n_xx = sum(unigram_counter.values())
    min_bigram_freq = max(3, int(n_xx * 0.00001))
    scored = []
    for (w1, w2), freq in bigram_counter.items():
        if freq >= min_bigram_freq:
            ll = log_likelihood(freq, unigram_counter[w1], unigram_counter[w2], n_xx)
            scored.append((w1, w2, freq, ll))

    scored.sort(key=lambda x: x[3], reverse=True)
    top_bigrams = scored[:25]

    if not top_bigrams:
        return

    G = nx.Graph()
    for w1, w2, freq, ll in top_bigrams:
        weight = min(ll / 10, 5)
        G.add_edge(w1, w2, weight=weight, frequency=freq, ll=ll)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node sizes proportional to degree
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 200 + 100 for n in G.nodes()]

    # Edge widths proportional to weight
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#4C72B0",
                           alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w / max_w * 3 for w in edge_weights],
                           alpha=0.5, edge_color="#888", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    ax.set_title("Collocation Network (Top Bigrams by Log-Likelihood)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "collocation_network.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved collocation network to {path}")


def plot_pos_distribution(df, output_dir="output"):
    """POS distribution bar chart."""
    if not HAS_MATPLOTLIB:
        return

    all_pos = Counter()
    for pos_seq in df["pos_tags"]:
        all_pos.update(pos_seq)

    top_pos = all_pos.most_common(15)
    labels, counts = zip(*top_pos)
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, pcts, color="#55A868", edgecolor="white")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("POS Tag Distribution", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "pos_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved POS distribution to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(results, output_dir="output"):
    """Generate comprehensive HTML report."""
    import base64

    sections = []

    def img_to_base64(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corpus Linguistics ULTRA — Analysis Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa;
         color: #333; line-height: 1.6; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #2980b9; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 12px; }
  h3 { color: #555; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
  th { background: #3498db; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
  tr:hover { background: #f0f7ff; }
  .metric { display: inline-block; background: #ecf0f1; border-radius: 6px;
            padding: 8px 16px; margin: 4px; text-align: center; }
  .metric .value { font-size: 20px; font-weight: bold; color: #2c3e50; }
  .metric .label { font-size: 11px; color: #7f8c8d; }
  .section { background: white; border-radius: 8px; padding: 20px;
             margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  img { max-width: 100%; border-radius: 6px; margin: 10px 0; }
  .summary-box { background: #eaf2f8; border-left: 4px solid #3498db;
                 padding: 12px 16px; border-radius: 4px; margin: 15px 0; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
  .footer { text-align: center; color: #aaa; font-size: 12px; margin-top: 40px; }
</style>
</head>
<body>
<h1>Corpus Linguistics ULTRA — Analysis Report</h1>
<p style="color:#777;">Generated: """ + time.strftime("%Y-%m-%d %H:%M:%S") + "</p>\n"

    # Corpus overview
    if "corpus_stats" in results:
        html += '<div class="section"><h2>Corpus Overview</h2>\n'
        html += '<div class="summary-box">\n'
        stats = results["corpus_stats"]
        if isinstance(stats, pd.DataFrame) and len(stats) > 0:
            html += f'<p>Total documents analyzed</p>\n'
        html += '</div></div>\n'

    # Images
    for name, title in [("frequency_distribution", "Word Frequency Distribution"),
                         ("dispersion_plot", "Word Dispersion Across Corpus"),
                         ("collocation_network", "Collocation Network"),
                         ("pos_distribution", "POS Tag Distribution")]:
        path = os.path.join(output_dir, f"{name}.png")
        b64 = img_to_base64(path)
        if b64:
            html += f'<div class="section"><h2>{title}</h2>\n'
            html += f'<img src="data:image/png;base64,{b64}" alt="{title}">\n'
            html += '</div>\n'

    # Biber dimensions
    dim_path = os.path.join(output_dir, "biber_dimensions.csv")
    if os.path.exists(dim_path):
        try:
            dim_df = pd.read_csv(dim_path)
            html += '<div class="section"><h2>Biber MD Dimension Scores</h2>\n'
            html += '<p>Aggregated from ~30 POS-level features into 5 interpretable dimensions (Biber 1988, 1995). Positive = left label, Negative = right label.</p>\n'
            for _, r in dim_df.iterrows():
                score = r["score"]
                label = r.get("label", "")
                bar_w = min(max(abs(score) / 5, 0), 100)
                color = "#3498db" if score > 0 else "#e74c3c"
                html += f'<div style="margin:12px 0;"><strong>{r["dimension"]}</strong><br>'
                html += f'<small>{label}</small><br>'
                html += f'<div style="background:#ecf0f1; border-radius:4px; height:22px; width:100%; position:relative;">'
                html += f'<div style="background:{color}; width:{min(abs(score)*10, 100)}%; height:22px; border-radius:4px;'
                html += f'{" float:right;" if score < 0 else ""}"></div></div>'
                html += f'<span>{score:+.2f}</span></div>\n'
            html += '</div>\n'
        except Exception:
            pass

    # Syntactic complexity
    synt_path = os.path.join(output_dir, "syntactic_complexity.csv")
    if os.path.exists(synt_path):
        try:
            synt_df = pd.read_csv(synt_path)
            num_cols = [c for c in synt_df.columns if c not in ("doc_id", "group")]
            html += '<div class="section"><h2>Syntactic Complexity</h2>\n'
            html += '<p>L2 writing measures: Wolfe-Quintero et al. (1998), Norris & Ortega (2009)</p>\n'
            html += '<table><tr><th>Measure</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>\n'
            for col in num_cols:
                v = synt_df[col].dropna()
                if len(v) > 0:
                    html += f'<tr><td>{col}</td><td>{v.mean():.2f}</td><td>{v.std():.2f}</td><td>{v.min():.2f}</td><td>{v.max():.2f}</td></tr>\n'
            html += '</table></div>\n'
        except Exception:
            pass

    # Lexical sophistication
    lexsoph_path = os.path.join(output_dir, "lexical_sophistication.csv")
    if os.path.exists(lexsoph_path):
        try:
            ls_df = pd.read_csv(lexsoph_path)
            html += '<div class="section"><h2>Lexical Sophistication (Vocabulary Profile)</h2>\n'
            html += '<p>GSL = General Service List (West 1953), AWL = Academic Word List (Coxhead 2000)</p>\n'
            html += '<table><tr><th>Measure</th><th>Mean</th><th>Std</th></tr>\n'
            for col in ["gsl_token_pct", "awl_token_pct", "off_list_token_pct",
                         "gsl_type_pct", "awl_type_pct", "off_list_type_pct"]:
                if col in ls_df.columns:
                    v = ls_df[col].dropna()
                    if len(v) > 0:
                        html += f'<tr><td>{col}</td><td>{v.mean():.2f}%</td><td>{v.std():.2f}</td></tr>\n'
            html += '</table></div>\n'
        except Exception:
            pass

    # Tables
    for name, title in [("collocations", "Collocation Extraction"),
                         ("dispersion", "Dispersion Analysis"),
                         ("readability", "Readability Scores"),
                         ("lexical_richness", "Lexical Richness"),
                         ("pos_analysis", "Biber Features"),
                         ("keyness", "Keyness Analysis"),
                         ("keyphrases", "Keyphrase Extraction"),
                         ("bundles_4gram", "Lexical Bundles (4-word)")]:
        path = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(path):
            try:
                tbl_df = pd.read_csv(path)
                html += f'<div class="section"><h2>{title}</h2>\n'
                html += tbl_df.head(30).to_html(index=False, classes="", border=0)
                html += f'<p style="color:#999; font-size:12px;">Showing top 30 of {len(tbl_df)} rows. Full data in CSV.</p>\n'
                html += '</div>\n'
            except Exception:
                pass

    html += """
<div class="footer">
  Corpus Linguistics ULTRA v1.0 — Built with spaCy, textstat, lexicalrichness, scattertext<br>
  Evidence base: Gries 2024, Sönning 2025, Deng & Liu 2022, Biber et al. 2004
</div>
</body>
</html>"""

    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report saved to {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: GROUP COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_group_comparison(df, focus_col="group", output_dir="output"):
    """Compare lexical statistics across groups."""
    groups = df[focus_col].unique()
    if len(groups) < 2:
        return

    print(f"\n{'='*60}")
    print("GROUP COMPARISON")
    print(f"{'='*60}")

    for group in sorted(groups):
        gdf = df[df[focus_col] == group]
        all_lemmas = []
        for lemmas in gdf["lemmas"]:
            all_lemmas.extend(lemmas)
        n_tokens = sum(len(t) for t in gdf["tokens"])
        n_types = len(set(all_lemmas))
        freq = Counter(all_lemmas)

        print(f"\n  Group: {group} ({len(gdf)} docs, {n_tokens:,} tokens)")
        print(f"    Types: {n_types:,}, TTR: {n_types/max(n_tokens,1):.4f}")
        print(f"    Top 10: {', '.join(f'{w}({c})' for w, c in freq.most_common(10))}")

    # Log-likelihood between first two groups
    if len(groups) >= 2:
        g1, g2 = sorted(groups)[:2]
        g1_lemmas = []
        g2_lemmas = []
        for _, row in df.iterrows():
            if row[focus_col] == g1:
                g1_lemmas.extend(row["lemmas"])
            elif row[focus_col] == g2:
                g2_lemmas.extend(row["lemmas"])

        c1, c2 = Counter(g1_lemmas), Counter(g2_lemmas)
        N1, N2 = len(g1_lemmas), len(g2_lemmas)

        keyness_results = []
        for word in set(c1.keys()) | set(c2.keys()):
            a, b = c1.get(word, 0), c2.get(word, 0)
            if a + b >= 3:
                E1 = N1 * (a + b) / max(N1 + N2, 1)
                E2 = N2 * (a + b) / max(N1 + N2, 1)
                g2_v = 0.0
                small = 1e-20
                if a > 0 and E1 > 0:
                    g2_v += a * math.log(a / E1 + small)
                if b > 0 and E2 > 0:
                    g2_v += b * math.log(b / E2 + small)
                ll = 2 * g2_v
                focus_ppm = a / max(N1, 1) * 1000
                ref_ppm = b / max(N2, 1) * 1000
                lr = math.log2(focus_ppm / max(ref_ppm, 1e-20)) if focus_ppm > 0 else 0
                keyness_results.append({
                    "word": word, "group1_freq": a, "group2_freq": b,
                    "log_likelihood": round(ll, 4), "log_ratio": round(lr, 4),
                })

        if not keyness_results:
            print(f"\n  No differential keyness found between {g1} and {g2}.")
            return
        keyness_df = pd.DataFrame(keyness_results).sort_values("log_likelihood", ascending=False)
        print(f"\n  Differential keyness ({g1} vs {g2}):")
        print(f"  {'Word':<20} {g1+' freq':>10} {g2+' freq':>10} {'LL':>8} {'LogR':>7}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")
        for _, r in keyness_df.head(15).iterrows():
            print(f"  {r['word']:<20} {r['group1_freq']:>10} {r['group2_freq']:>10} "
                  f"{r['log_likelihood']:>8.2f} {r['log_ratio']:>+7.2f}")

        out_path = os.path.join(output_dir, "group_keyness.csv")
        keyness_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  Saved to {out_path}")


def generate_index_html(output_dir="output"):
    """Generate index.html listing all output files with clickable links."""
    files = sorted(os.listdir(output_dir))

    # Group files by analysis section
    section_map = {
        "readability": "Readability",
        "collocations": "Collocation Extraction",
        "dispersion": "Dispersion Analysis",
        "lexical_richness": "Lexical Richness",
        "pos_analysis": "POS Analysis",
        "biber_dimensions": "Biber Dimensions",
        "syntactic_complexity": "Syntactic Complexity",
        "lexical_sophistication": "Lexical Sophistication",
        "pos_trigrams": "POS Trigrams",
        "bundles": "Lexical Bundles",
        "keyphrases": "Keyphrase Extraction",
        "keyness": "Keyness Analysis",
        "frequency_list": "Frequency List",
        "group_keyness": "Group Keyness",
        "key_pos_tags": "Key POS Tags",
        "ngram_overlap": "N-gram Overlap",
        "comparative_concordance": "Comparative Concordance",
        "collocation_centrality": "Collocation Centrality",
        "error_detection": "Error Detection",
        "time_analysis": "Time Analysis",
        "frequency_distribution": "Frequency Distribution",
        "dispersion_plot": "Dispersion Plot",
        "collocation_network": "Collocation Network",
        "pos_distribution": "POS Distribution",
        "report": "Full Report",
        "interactive_kwic": "Interactive KWIC",
        "manifest": "Run Manifest",
    }

    # Build sections
    sections_html = {}
    for fname in files:
        if fname == "index.html":
            continue
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        section = None
        for key, label in section_map.items():
            if name_no_ext.startswith(key):
                section = label
                break
        if section is None:
            section = "Other Files"
        if section not in sections_html:
            sections_html[section] = []
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        ext = os.path.splitext(fname)[1].lower()
        icon = "CSV" if ext == ".csv" else ("HTML" if ext == ".html" else ("IMG" if ext in (".png", ".jpg") else "FILE"))
        sections_html[section].append((fname, size_str, icon))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corpus ULTRA — Output Index</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 12px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0 25px 0; font-size: 13px; }}
  th {{ background: #3498db; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ background: #f0f7ff; }}
  a {{ color: #2980b9; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
  .badge-csv {{ background: #eaf2f8; color: #2980b9; }}
  .badge-html {{ background: #fef9e7; color: #b8860b; }}
  .badge-img {{ background: #eafaf1; color: #27ae60; }}
  .badge-file {{ background: #f4f4f4; color: #666; }}
  .footer {{ text-align: center; color: #aaa; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>Corpus ULTRA — Output Index</h1>
<p style="color:#777;">Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
"""

    for section in sorted(sections_html.keys()):
        items = sections_html[section]
        html += f'<h2>{section}</h2>\n<table>\n'
        html += '<tr><th>File</th><th>Size</th><th>Type</th></tr>\n'
        for fname, size_str, icon in items:
            badge_class = {"CSV": "badge-csv", "HTML": "badge-html", "IMG": "badge-img"}.get(icon, "badge-file")
            html += f'<tr><td><a href="{fname}">{fname}</a></td><td>{size_str}</td>'
            html += f'<td><span class="badge {badge_class}">{icon}</span></td></tr>\n'
        html += '</table>\n'

    html += """
<div class="footer">
  Corpus Linguistics ULTRA v1.0
</div>
</body>
</html>"""

    path = os.path.join(output_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Generated index.html with {sum(len(v) for v in sections_html.values())} files")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_time_analysis(df, output_dir="output"):
    """Time-based analysis: frequency of top 10 lemmas per year.

    Looks for a 'date' or 'year' column. If 'date' exists, extracts year.
    """
    print(f"\n{'='*60}")
    print("TIME ANALYSIS")
    print(f"{'='*60}")

    # Determine time column
    year_col = None
    if "year" in df.columns:
        year_col = "year"
    elif "date" in df.columns:
        try:
            df["_parsed_year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
            if df["_parsed_year"].notna().sum() > 0:
                year_col = "_parsed_year"
        except Exception:
            pass

    if year_col is None:
        print("  No 'date' or 'year' column found. Skipping time analysis.")
        return pd.DataFrame()

    df_valid = df.dropna(subset=[year_col]).copy()
    df_valid[year_col] = df_valid[year_col].astype(int)

    if len(df_valid) == 0:
        print("  No valid date/year values found.")
        return pd.DataFrame()

    years = sorted(df_valid[year_col].unique())
    print(f"  Years found: {years[0]} to {years[-1]} ({len(years)} distinct years, {len(df_valid)} docs)")

    # Top 10 lemmas per year
    year_lemma_counts = {}
    for year in years:
        subset = df_valid[df_valid[year_col] == year]
        lemmas = []
        for lemma_list in subset["lemmas"]:
            lemmas.extend(lemma_list)
        year_lemma_counts[year] = Counter(lemmas)

    # Collect top 10 lemmas across all years
    global_lemmas = Counter()
    for c in year_lemma_counts.values():
        global_lemmas.update(c)
    top_lemmas = [w for w, _ in global_lemmas.most_common(10)]

    # Build pivot table
    rows = []
    for year in years:
        row = {"year": year}
        counts = year_lemma_counts[year]
        total = sum(counts.values())
        for lemma in top_lemmas:
            freq = counts.get(lemma, 0)
            row[lemma] = round(freq / max(total, 1) * 1e6, 1)  # normalized per million
        rows.append(row)

    time_df = pd.DataFrame(rows)

    # Print table
    print(f"\n  Top 10 lemma frequencies per year (per million tokens):")
    header = f"  {'Year':>6}"
    for lemma in top_lemmas:
        header += f" {lemma:>10}"
    print(header)
    print(f"  {'-'*6}" + " " + " ".join(f"{'-'*10}" for _ in top_lemmas))
    for _, r in time_df.iterrows():
        line = f"  {int(r['year']):>6}"
        for lemma in top_lemmas:
            line += f" {r[lemma]:>10.1f}"
        print(line)

    out_path = os.path.join(output_dir, "time_analysis.csv")
    time_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved to {out_path}")

    # Cleanup temp column
    if "_parsed_year" in df.columns:
        df.drop(columns=["_parsed_year"], inplace=True, errors="ignore")

    return time_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: LEXICAL SOPHISTICATION (wordfreq D_L)
# ═══════════════════════════════════════════════════════════════════════════════

def run_lexical_sophistication(df, output_dir="output"):
    """Lexical sophistication using wordfreq D_L measure (Kyle & Crossley 2015)."""
    if not HAS_WORDFREQ:
        print("  Skipping: wordfreq not installed (pip install wordfreq)")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("LEXICAL SOPHISTICATION (wordfreq D_L)")
    print(f"{'='*60}")

    results = []
    for idx, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        if not lemmas:
            continue

        # Frequency bands
        bands = {i: 0 for i in range(1, 8)}
        zipf_scores = []
        for token in lemmas:
            z = zipf_frequency(token.lower(), 'en')
            zipf_scores.append(z)
            if z >= 6.0: bands[1] += 1
            elif z >= 5.0: bands[2] += 1
            elif z >= 4.0: bands[3] += 1
            elif z >= 3.0: bands[4] += 1
            elif z >= 2.0: bands[5] += 1
            elif z >= 1.0: bands[6] += 1
            else: bands[7] += 1

        total = len(lemmas)
        mean_z = sum(zipf_scores) / len(zipf_scores) if zipf_scores else 0
        d_l = (bands[4] + bands[5] + bands[6] + bands[7]) / total * 100 if total > 0 else 0

        results.append({
            "doc_id": idx,
            "mean_zipf": mean_z,
            "d_l": d_l,
            "band_1_pct": bands[1] / total * 100,
            "band_2_pct": bands[2] / total * 100,
            "band_3_pct": bands[3] / total * 100,
            "band_4_pct": bands[4] / total * 100,
            "band_5_pct": bands[5] / total * 100,
            "band_6_pct": bands[6] / total * 100,
            "band_7_pct": bands[7] / total * 100,
        })

    result_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "lexical_sophistication.csv")
    result_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Mean D_L: {result_df['d_l'].mean():.1f}%")
    print(f"  Mean Zipf: {result_df['mean_zipf'].mean():.2f}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16: USAS SEMANTIC TAGGING
# ═══════════════════════════════════════════════════════════════════════════════

def run_usas(df, output_dir="output"):
    """USAS semantic tagging using 54K lexicon."""
    if not HAS_USAS:
        print("  Skipping: usas_standalone not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("USAS SEMANTIC TAGGING")
    print(f"{'='*60}")

    all_tokens = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        all_tokens.extend([t.lower() for t in lemmas])

    result = usas_profile(all_tokens)
    result_detailed = usas_profile_detailed(all_tokens, top_n=20)

    # Save profile
    profile_df = pd.DataFrame([{"category": k, "percentage": v} for k, v in result.items()])
    out_path = os.path.join(output_dir, "usas.csv")
    profile_df.to_csv(out_path, index=False)

    # Save detailed
    detailed_df = pd.DataFrame(result_detailed)
    detailed_path = os.path.join(output_dir, "usas_detailed.csv")
    detailed_df.to_csv(detailed_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  {len(result)} semantic categories")
    for cat, pct in sorted(result.items(), key=lambda x: -x[1])[:5]:
        print(f"    {cat}: {pct:.1f}%")
    return profile_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17: NAMED ENTITY RECOGNITION (spaCy NER)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ner(df, output_dir="output"):
    """Extract named entities using spaCy NER."""
    if not HAS_SPACY:
        print("  Skipping: spaCy not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("NAMED ENTITY RECOGNITION (NER)")
    print(f"{'='*60}")

    nlp = spacy.load("en_core_web_sm")
    entity_counts = Counter()
    entity_examples = {}

    for idx, row in df.iterrows():
        text = row.get("text", "")
        if not text:
            continue
        doc = nlp(text)
        for ent in doc.ents:
            entity_counts[(ent.text, ent.label_)] += 1
            if ent.label_ not in entity_examples:
                entity_examples[ent.label_] = []
            if len(entity_examples[ent.label_]) < 3:
                entity_examples[ent.label_].append(ent.text)

    # Build results
    results = []
    for (text, label), count in entity_counts.most_common(50):
        results.append({
            "entity": text,
            "label": label,
            "count": count,
            "examples": ", ".join(entity_examples.get(label, [])[:3]),
        })

    result_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "ner.csv")
    result_df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  {len(results)} unique entities")
    for _, r in result_df.head(10).iterrows():
        print(f"    {r['entity']:>20} ({r['label']:>4}): {r['count']}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18: ENHANCED NETWORK ANALYSIS (PageRank + Clustering)
# ═══════════════════════════════════════════════════════════════════════════════

def run_network_enhanced(df, min_freq=5, output_dir="output"):
    """Enhanced collocation network with PageRank, clustering, communities."""
    if not HAS_NETWORKX:
        print("  Skipping: networkx not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("ENHANCED NETWORK ANALYSIS")
    print(f"{'='*60}")

    # Build collocation network
    all_lemmas = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        all_lemmas.extend([t.lower() for t in lemmas])

    # Bigram counts
    bigram_counts = Counter()
    for i in range(len(all_lemmas) - 1):
        bigram_counts[(all_lemmas[i], all_lemmas[i + 1])] += 1

    # Build graph
    G = nx.Graph()
    for (w1, w2), count in bigram_counts.items():
        if count >= min_freq and len(w1) > 2 and len(w2) > 2:
            G.add_edge(w1, w2, weight=count)

    if len(G) == 0:
        print("  No edges found with min_freq=" + str(min_freq))
        return pd.DataFrame()

    # PageRank
    pagerank = nx.pagerank(G, weight='weight')

    # Clustering coefficient
    clustering = nx.clustering(G, weight='weight')

    # Community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
    except:
        community_map = {}

    # Centrality measures
    betweenness = nx.betweenness_centrality(G, weight='weight')
    degree = nx.degree_centrality(G)

    # Build results
    results = []
    for node in G.nodes():
        results.append({
            "word": node,
            "pagerank": pagerank.get(node, 0),
            "clustering": clustering.get(node, 0),
            "betweenness": betweenness.get(node, 0),
            "degree": degree.get(node, 0),
            "community": community_map.get(node, -1),
            "degree_count": G.degree(node),
        })

    result_df = pd.DataFrame(results).sort_values("pagerank", ascending=False)
    out_path = os.path.join(output_dir, "network_enhanced.csv")
    result_df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"  Top 5 by PageRank:")
    for _, r in result_df.head(5).iterrows():
        print(f"    {r['word']:>15}  PR={r['pagerank']:.4f}  cluster={r['clustering']:.3f}  comm={r['community']}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19: USAS POS-AWARE TAGGING
# ═══════════════════════════════════════════════════════════════════════════════

def run_usas_pos_aware(df, output_dir="output"):
    """USAS semantic tagging with POS awareness using spaCy."""
    if not HAS_USAS or not HAS_SPACY:
        print("  Skipping: usas_standalone or spaCy not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("USAS POS-AWARE TAGGING")
    print(f"{'='*60}")

    nlp = spacy.load("en_core_web_sm")
    all_tokens = []
    all_pos = []

    for _, row in df.iterrows():
        text = row.get("text", "")
        if not text:
            continue
        doc = nlp(text)
        for token in doc:
            all_tokens.append(token.text.lower())
            all_pos.append(token.pos_)

    # POS-aware tagging
    result = usas_profile(all_tokens, pos_tags=all_pos)
    result_detailed = usas_profile_detailed(all_tokens, pos_tags=all_pos, top_n=20)

    # Compare with non-POS-aware
    result_no_pos = usas_profile(all_tokens)

    # Save comparison
    comparison = []
    for cat in sorted(set(result.keys()) | set(result_no_pos.keys())):
        comparison.append({
            "category": cat,
            "pos_aware_pct": result.get(cat, 0),
            "no_pos_pct": result_no_pos.get(cat, 0),
            "difference": result.get(cat, 0) - result_no_pos.get(cat, 0),
        })

    comp_df = pd.DataFrame(comparison)
    out_path = os.path.join(output_dir, "usas_pos_aware.csv")
    comp_df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  POS-aware tagging improved accuracy for {len([r for r in comparison if abs(r['difference']) > 1])} categories")
    for _, r in comp_df.head(5).iterrows():
        print(f"    {r['category']}: POS-aware={r['pos_aware_pct']:.1f}%  no-POS={r['no_pos_pct']:.1f}%  diff={r['difference']:+.1f}%")
    return comp_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 20: ENHANCED KEYPHRASE EXTRACTION (MMR Diversity)
# ═══════════════════════════════════════════════════════════════════════════════

def run_keyphrases_enhanced(df, output_dir="output"):
    """Enhanced keyphrase extraction with MMR diversity."""
    if not HAS_KEYBERT:
        print("  Skipping: keybert not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("ENHANCED KEYPHRASE EXTRACTION (MMR Diversity)")
    print(f"{'='*60}")

    kw_model = KeyBERT()
    results = []

    for idx, row in df.iterrows():
        text = row.get("text", "")
        if not text or len(text.split()) < 10:
            continue

        # Standard extraction
        standard = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), top_n=5)

        # MMR diversity
        mmr = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
                                         use_mmr=True, diversity=0.7, top_n=5)

        results.append({
            "doc_id": idx,
            "standard": ", ".join([kw for kw, _ in standard]),
            "mmr_diverse": ", ".join([kw for kw, _ in mmr]),
            "standard_scores": ", ".join([f"{s:.3f}" for _, s in standard]),
            "mmr_scores": ", ".join([f"{s:.3f}" for _, s in mmr]),
        })

    result_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "keyphrases_enhanced.csv")
    result_df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  {len(results)} documents processed")
    for _, r in result_df.head(3).iterrows():
        print(f"    Doc {r['doc_id']}:")
        print(f"      Standard: {r['standard']}")
        print(f"      MMR:      {r['mmr_diverse']}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 21: ENHANCED VISUALIZATION (Heatmaps)
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization_enhanced(df, output_dir="output"):
    """Enhanced visualizations: heatmaps, scatter plots."""
    if not HAS_PLOTLY:
        print("  Skipping: plotly not available")
        return

    print(f"\n{'='*60}")
    print("ENHANCED VISUALIZATIONS")
    print(f"{'='*60}")

    # Collocation heatmap
    all_lemmas = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        all_lemmas.extend([t.lower() for t in lemmas])

    # Top 15 words
    freq = Counter(all_lemmas)
    top_words = [w for w, _ in freq.most_common(15) if len(w) > 2]

    # Build co-occurrence matrix
    matrix = []
    for w1 in top_words:
        row = []
        for w2 in top_words:
            count = sum(1 for i in range(len(all_lemmas) - 1)
                       if (all_lemmas[i] == w1 and all_lemmas[i + 1] == w2) or
                          (all_lemmas[i] == w2 and all_lemmas[i + 1] == w1))
            row.append(count)
        matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=top_words,
        y=top_words,
        colorscale='Viridis',
        text=matrix,
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    fig.update_layout(
        title='Collocation Heatmap (Top 15 Words)',
        xaxis_title='Word',
        yaxis_title='Word',
        width=800,
        height=800,
    )
    out_path = os.path.join(output_dir, "collocation_heatmap.html")
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 22: NLTK TEXT ANALYSIS (similar, common_contexts, hapaxes)
# ═══════════════════════════════════════════════════════════════════════════════

def run_nltk_analysis(df, output_dir="output"):
    """NLTK text analysis: similar words, shared contexts, hapax legomena."""
    if not HAS_NLTK:
        print("  Skipping: NLTK not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("NLTK TEXT ANALYSIS")
    print(f"{'='*60}")

    # Build token list
    all_lemmas = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        all_lemmas.extend([t.lower() for t in lemmas])

    if len(all_lemmas) < 100:
        print("  Too few tokens for NLTK analysis")
        return pd.DataFrame()

    # Create NLTK Text object
    nltk_text = Text(all_lemmas)

    results = {}

    # 1. Similar words — "What words are like lonely?"
    target_words = ["lonely", "sad", "happy", "feel", "love"]
    print("\n  SIMILAR WORDS:")
    for word in target_words:
        if word in set(all_lemmas):
            try:
                similar = nltk_text.similar(word, num=10)
                results[f"similar_{word}"] = similar
            except:
                pass

    # 2. Common contexts — "How are lonely and sad used differently?"
    print("\n  COMMON CONTEXTS (lonely vs sad):")
    try:
        nltk_text.common_contexts(["lonely", "sad"], num=10)
    except:
        pass

    # 3. Hapax legomena — words appearing only once
    fd = FreqDist(all_lemmas)
    hapaxes = fd.hapaxes()
    print(f"\n  HAPAX LEGOMENA: {len(hapaxes)} words appearing once")
    print(f"  Top 10: {', '.join(hapaxes[:10])}")

    # 4. Frequency distribution
    print(f"\n  FREQUENCY DISTRIBUTION:")
    print(f"  Total words: {fd.N()}")
    print(f"  Unique words: {fd.B()}")
    print(f"  Hapax ratio: {len(hapaxes)/fd.B()*100:.1f}%")
    for word, count in fd.most_common(10):
        print(f"    {word:>15}: {count:>5} ({count/fd.N()*100:.1f}%)")

    # Save results
    results_df = pd.DataFrame([{
        "metric": "total_words",
        "value": fd.N(),
    }, {
        "metric": "unique_words",
        "value": fd.B(),
    }, {
        "metric": "hapax_count",
        "value": len(hapaxes),
    }, {
        "metric": "hapax_ratio",
        "value": len(hapaxes) / fd.B() * 100,
    }])
    out_path = os.path.join(output_dir, "nltk_analysis.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 23: DOCUMENT CLUSTERING (KMeans)
# ═══════════════════════════════════════════════════════════════════════════════

def run_clustering(df, n_clusters=5, output_dir="output"):
    """KMeans document clustering to discover post types."""
    if not HAS_SKLEARN:
        print("  Skipping: sklearn not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("DOCUMENT CLUSTERING (KMeans)")
    print(f"{'='*60}")

    # Build TF-IDF matrix
    texts = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        texts.append(" ".join([t.lower() for t in lemmas]))

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Get top terms per cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_results = []
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-10:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_size = sum(clusters == i)
        cluster_results.append({
            "cluster": i,
            "size": cluster_size,
            "top_terms": ", ".join(top_terms),
        })

    # Add cluster assignment to DataFrame
    df_result = df.copy()
    df_result["cluster"] = clusters

    result_df = pd.DataFrame(cluster_results)
    out_path = os.path.join(output_dir, "clusters.csv")
    result_df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  {n_clusters} clusters, {len(texts)} documents")
    for _, r in result_df.iterrows():
        print(f"    Cluster {r['cluster']}: {r['size']} docs — {r['top_terms']}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 24: REGRESSION (predict outcomes from text features)
# ═══════════════════════════════════════════════════════════════════════════════

def run_regression(df, target_col=None, output_dir="output"):
    """Linear/logistic regression predicting outcomes from text features."""
    if not HAS_SKLEARN:
        print("  Skipping: sklearn not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("REGRESSION ANALYSIS")
    print(f"{'='*60}")

    # Build TF-IDF features
    texts = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        texts.append(" ".join([t.lower() for t in lemmas]))

    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    if target_col and target_col in df.columns:
        y = df[target_col].values

        # Determine if classification or regression
        unique_vals = len(set(y))
        if unique_vals <= 10:
            # Classification
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "LogisticRegression"
        else:
            # Regression
            model = LinearRegression()
            model_name = "LinearRegression"

        model.fit(tfidf_matrix, y)
        score = model.score(tfidf_matrix, y)

        # Get top coefficients
        if hasattr(model, "coef_"):
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_names = vectorizer.get_feature_names_out()
            top_idx = coef.argsort()[-10:][::-1]
            top_features = [(feature_names[i], coef[i]) for i in top_idx]

            print(f"  Model: {model_name}")
            print(f"  R² / Accuracy: {score:.3f}")
            print(f"  Top features:")
            for feat, weight in top_features:
                print(f"    {feat:>20}: {weight:>8.3f}")

            # Save
            results = pd.DataFrame([{"feature": f, "weight": w} for f, w in top_features])
            out_path = os.path.join(output_dir, "regression_coefficients.csv")
            results.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
            return results
    else:
        print("  No target column specified. Use --regression-target <col>")
        print("  Available columns:", [c for c in df.columns if c != "text" and c != "lemmas"][:10])
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 25: DEPENDENCY PARSING (spaCy dep_)
# ═══════════════════════════════════════════════════════════════════════════════

def run_dependency_parsing(df, output_dir="output"):
    """Extract subject-verb-object patterns using spaCy dependency parsing.

    Answers: "Who does what to whom?" in lonely posts.
    """
    if not HAS_SPACY:
        print("  Skipping: spaCy not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("DEPENDENCY PARSING (Subject-Verb-Object)")
    print(f"{'='*60}")

    nlp = spacy.load("en_core_web_sm")
    svo_patterns = Counter()
    subjects = Counter()
    verbs = Counter()

    for _, row in df.iterrows():
        text = row.get("text", "")
        if not text:
            continue
        doc = nlp(text)
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                svo_patterns[(token.lemma_.lower(), token.head.lemma_.lower())] += 1
                subjects[token.lemma_.lower()] += 1
                verbs[token.lemma_.lower()] += 1

    # Save SVO patterns
    svo_results = pd.DataFrame([
        {"subject": s, "verb": v, "count": c}
        for (s, v), c in svo_patterns.most_common(30)
    ])
    svo_path = os.path.join(output_dir, "dependency_svo.csv")
    svo_results.to_csv(svo_path, index=False)

    # Save subject/verb frequencies
    freq_df = pd.DataFrame([
        {"type": "subject", "word": w, "count": c}
        for w, c in subjects.most_common(20)
    ] + [
        {"type": "verb", "word": w, "count": c}
        for w, c in verbs.most_common(20)
    ])
    freq_path = os.path.join(output_dir, "dependency_freq.csv")
    freq_df.to_csv(freq_path, index=False)

    print(f"  Saved: {svo_path}")
    print(f"  {len(svo_patterns)} unique SVO patterns")
    print(f"  Top subjects:")
    for w, c in subjects.most_common(5):
        print(f"    {w:>15}: {c}")
    print(f"  Top verbs:")
    for w, c in verbs.most_common(5):
        print(f"    {w:>15}: {c}")
    return svo_results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 26: DOCUMENT SIMILARITY (cosine similarity)
# ═══════════════════════════════════════════════════════════════════════════════

def run_document_similarity(df, output_dir="output"):
    """Find most similar document pairs using cosine similarity.

    Answers: "Which posts are most alike?"
    """
    if not HAS_SKLEARN:
        print("  Skipping: sklearn not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("DOCUMENT SIMILARITY (Cosine)")
    print(f"{'='*60}")

    from sklearn.metrics.pairwise import cosine_similarity

    texts = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        texts.append(" ".join([t.lower() for t in lemmas]))

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(tfidf_matrix)

    # Find top similar pairs (excluding self-similarity)
    pairs = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):  # Check nearby docs only
            pairs.append({
                "doc_a": i,
                "doc_b": j,
                "similarity": sim_matrix[i][j],
            })

    pairs_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False)
    top_pairs = pairs_df.head(20)

    out_path = os.path.join(output_dir, "document_similarity.csv")
    top_pairs.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  Most similar pairs:")
    for _, r in top_pairs.head(5).iterrows():
        print(f"    Doc {r['doc_a']} ↔ Doc {r['doc_b']}: {r['similarity']:.3f}")
    return top_pairs


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 27: DOCUMENT CLUSTERING VISUALIZATION (PCA + t-SNE)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cluster_visualization(df, output_dir="output"):
    """Visualize document clusters using PCA and t-SNE."""
    if not HAS_SKLEARN or not HAS_PLOTLY:
        print("  Skipping: sklearn or plotly not available")
        return

    print(f"\n{'='*60}")
    print("CLUSTER VISUALIZATION (PCA + t-SNE)")
    print(f"{'='*60}")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    texts = []
    for _, row in df.iterrows():
        lemmas = row.get("lemmas", row.get("tokens", []))
        texts.append(" ".join([t.lower() for t in lemmas]))

    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # KMeans
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())

    # t-SNE (2D)
    n_samples = min(500, len(texts))
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples - 1))
    tsne_result = tsne.fit_transform(tfidf_matrix[:n_samples].toarray())

    # PCA plot
    fig_pca = go.Figure()
    for c in range(5):
        mask = clusters == c
        fig_pca.add_trace(go.Scatter(
            x=pca_result[mask, 0],
            y=pca_result[mask, 1],
            mode='markers',
            name=f'Cluster {c}',
            marker=dict(size=5, opacity=0.7),
        ))
    fig_pca.update_layout(
        title='Document Clusters (PCA)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        width=800, height=600,
    )
    fig_pca.write_html(os.path.join(output_dir, "cluster_pca.html"))
    print(f"  Saved: cluster_pca.html")

    # t-SNE plot
    fig_tsne = go.Figure()
    for c in range(5):
        mask = clusters[:n_samples] == c
        fig_tsne.add_trace(go.Scatter(
            x=tsne_result[mask, 0],
            y=tsne_result[mask, 1],
            mode='markers',
            name=f'Cluster {c}',
            marker=dict(size=5, opacity=0.7),
        ))
    fig_tsne.update_layout(
        title='Document Clusters (t-SNE)',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        width=800, height=600,
    )
    fig_tsne.write_html(os.path.join(output_dir, "cluster_tsne.html"))
    print(f"  Saved: cluster_tsne.html")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 28: ZERO-SHOT CLASSIFICATION (transformers)
# ═══════════════════════════════════════════════════════════════════════════════

def run_zero_shot(df, output_dir="output", labels=None):
    """Zero-shot text classification — no training needed.

    Answers: "What category does this post belong to?"
    """
    try:
        from transformers import pipeline
    except ImportError:
        print("  Skipping: transformers not available")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("ZERO-SHOT CLASSIFICATION (transformers)")
    print(f"{'='*60}")

    if labels is None:
        labels = ["loneliness", "depression", "anxiety", "hope", "gratitude", "grief"]

    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    results = []
    for idx, row in df.iterrows():
        text = row.get("text", "")
        if not text or len(text.split()) < 5:
            continue
        # Truncate for speed
        short_text = " ".join(text.split()[:100])
        try:
            output = classifier(short_text, labels, multi_label=True)
            result = {"doc_id": idx}
            for label, score in zip(output["labels"], output["scores"]):
                result[label] = round(score, 3)
            results.append(result)
        except:
            pass

    result_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "zero_shot.csv")
    result_df.to_csv(out_path, index=False)

    print(f"  Saved: {out_path}")
    print(f"  {len(results)} documents classified")
    print(f"  Labels: {labels}")
    if results:
        mean_scores = {l: result_df[l].mean() for l in labels if l in result_df.columns}
        for label, score in sorted(mean_scores.items(), key=lambda x: -x[1]):
            print(f"    {label:>15}: {score:.3f}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Corpus Linguistics ULTRA — Comprehensive corpus analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python corpus_ultra.py data.csv review_text
  python corpus_ultra.py data.csv review_text --all
  python corpus_ultra.py data.csv review_text --group rating --keyness ref.csv
  python corpus_ultra.py data.csv review_text --kwic "excellent" --collocations
        """,
    )
    parser.add_argument("corpus", help="Path to CSV/TSV/TXT corpus file")
    parser.add_argument("text_col", help="Name of text column in CSV")
    parser.add_argument("--group", help="Column name for group comparison")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")

    # Analysis flags
    parser.add_argument("--kwic", nargs="?", const="*", help="KWIC concordance (optionally specify node word)")
    parser.add_argument("--collocations", action="store_true", help="Collocation extraction")
    parser.add_argument("--dispersion", action="store_true", help="Dispersion analysis")
    parser.add_argument("--readability", action="store_true", help="Readability analysis")
    parser.add_argument("--lexical-richness", action="store_true", help="Lexical richness analysis")
    parser.add_argument("--pos", action="store_true", help="POS analysis with Biber features")
    parser.add_argument("--bundles", action="store_true", help="Lexical bundle extraction")
    parser.add_argument("--keyphrases", action="store_true", help="Keyphrase extraction")
    parser.add_argument("--keyness", nargs="?", const="*", help="Keyness vs reference corpus")
    parser.add_argument("--stats", action="store_true", help="Corpus statistics")
    parser.add_argument("--plots", action="store_true", help="Generate visualizations")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--all", action="store_true", help="Run ALL analyses")
    parser.add_argument("--kwic-node", help="Node word for comparative concordance + interactive KWIC")
    parser.add_argument("--error-detection", action="store_true", help="Grammatical error detection (ERRANT)")
    parser.add_argument("--network-enhanced", action="store_true", help="Enhanced collocation network with centrality")
    parser.add_argument("--time-analysis", action="store_true", help="Time-based lemma frequency analysis")
    parser.add_argument("--lexical-sophistication", action="store_true", help="Lexical sophistication (wordfreq D_L)")
    parser.add_argument("--usas", action="store_true", help="USAS semantic tagging")
    parser.add_argument("--ner", action="store_true", help="Named Entity Recognition (spaCy NER)")
    parser.add_argument("--network-enhanced-v2", action="store_true", help="Enhanced network with PageRank + clustering")
    parser.add_argument("--usas-pos", action="store_true", help="USAS POS-aware tagging")
    parser.add_argument("--keyphrases-enhanced", action="store_true", help="Enhanced keyphrases with MMR diversity")
    parser.add_argument("--heatmap", action="store_true", help="Collocation heatmap visualization")
    parser.add_argument("--nltk", action="store_true", help="NLTK text analysis (similar, hapaxes)")
    parser.add_argument("--clustering", action="store_true", help="KMeans document clustering")
    parser.add_argument("--regression", action="store_true", help="Regression analysis")
    parser.add_argument("--regression-target", help="Target column for regression")
    parser.add_argument("--dependency", action="store_true", help="Dependency parsing (SVO)")
    parser.add_argument("--similarity", action="store_true", help="Document similarity (cosine)")
    parser.add_argument("--cluster-viz", action="store_true", help="PCA + t-SNE cluster visualization")
    parser.add_argument("--zero-shot", action="store_true", help="Zero-shot classification")

    # Parameters
    parser.add_argument("--min-freq", type=int, default=5, help="Minimum frequency for collocations")
    parser.add_argument("--max-results", type=int, default=50, help="Max results per analysis")
    parser.add_argument("--window", type=int, default=5, help="KWIC window size")
    parser.add_argument("--segments", type=int, default=10, help="Corpus segments for dispersion")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N documents from corpus")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--profile", choices=["demo", "fast", "full"], default="fast",
                        help="Analysis profile: demo (stats+readability), "
                             "fast (+collocations+dispersion+pos), full (--all)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose per-section output")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    _t_start = time.time()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Profile override: when set, overrides individual analysis flags
    if args.profile == "demo":
        args.all = False
        args.stats = True
        args.readability = True
        args.collocations = False
        args.dispersion = False
        args.pos = False
        args.bundles = False
        args.keyphrases = False
        args.kwic = None
        args.keyness = None
        args.plots = False
        args.report = False
        args.error_detection = False
        args.network_enhanced = False
    elif args.profile == "fast":
        args.all = False
        args.stats = True
        args.readability = True
        args.collocations = True
        args.dispersion = True
        args.pos = True
        args.bundles = False
        args.keyphrases = False
        args.kwic = None
        args.keyness = None
        args.plots = False
        args.report = False
        args.error_detection = False
        args.network_enhanced = False
    elif args.profile == "full":
        args.all = True

    # Track which sections ran (for manifest)
    sections_run = []

    print("=" * 60)
    print("CORPUS LINGUISTICS ULTRA v1.0")
    print("=" * 60)
    print(f"  Corpus: {args.corpus}")
    print(f"  Text column: {args.text_col}")
    if args.group:
        print(f"  Group column: {args.group}")
    print(f"  Output: {args.output}")
    print()

    # Quiet mode: redirect stdout for analysis sections (restored before summary)
    _saved_stdout = sys.stdout
    if args.quiet:
        sys.stdout = open(os.devnull, "w")

    # Check dependencies
    print("  Checking dependencies...")
    missing = []
    if not HAS_SPACY: missing.append("spacy")
    if not HAS_TEXTSTAT: missing.append("textstat")
    if not HAS_LEXICALRICHNESS: missing.append("lexicalrichness")
    if not HAS_MATPLOTLIB: missing.append("matplotlib")
    if not HAS_PLOTLY: missing.append("plotly")
    if not HAS_SKLEARN: missing.append("scikit-learn")
    if not HAS_NETWORKX: missing.append("networkx")
    if not HAS_WORDCLOUD: missing.append("wordcloud")
    if missing:
        print(f"  [!] Missing optional packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
    else:
        print("  All dependencies available.")

    # Load data
    print("\n  Loading corpus...")
    df = load_corpus(args.corpus, args.text_col, args.group, args.encoding)
    print(f"  Loaded {len(df)} documents")

    # Stratified sampling
    if args.sample is not None and args.sample < len(df):
        if args.group and args.group in df.columns:
            # Stratified sampling by group (proportional allocation)
            sampled = df.groupby("group", group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, round(args.sample * len(g) / len(df)))),
                    random_state=args.seed,
                )
            )
        else:
            sampled = df.sample(n=args.sample, random_state=args.seed)
        print(f"  Sampled {len(sampled)} documents from {len(df)} (seed={args.seed})")
        df = sampled.reset_index(drop=True)

    # ─── Contraction cleanup (Fix 2) ───────────────────────────────────────
    df["text"] = df["text"].apply(expand_contractions)
    print(f"  Applied contraction expansion to {len(df)} documents")

    # ─── SpaCy doc caching (Fix 1) ───────────────────────────────────────────
    cache_path = os.path.join(args.output, "spacy_docs.parquet")
    spacy_cached = False
    needs_spacy = (args.all or args.kwic or args.collocations or args.dispersion or
                   args.pos or args.bundles or args.readability or args.lexical_richness or
                   args.keyphrases or args.kwic_node or args.plots or args.report)

    if needs_spacy and os.path.exists(cache_path):
        try:
            df_cache = pd.read_parquet(cache_path)
            if len(df_cache) == len(df):
                for col in ["tokens", "lemmas", "pos_tags"]:
                    if col in df_cache.columns:
                        df[col] = df_cache[col].tolist() if hasattr(df_cache[col].iloc[0], '__iter__') else df_cache[col]
                # Reconstruct minimal spacy_doc stubs (only text needed for non-Biber/syntax functions)
                class _StubDoc:
                    def __init__(self, text):
                        self.text = text
                        self._tokens = text.split()
                    def __iter__(self):
                        class _Tok:
                            def __init__(self, t):
                                self.text = t
                                self.is_space = False
                                self.is_alpha = t.isalpha()
                                self.is_stop = False
                                self.is_punct = False
                                self.pos_ = "X"
                        return iter(_Tok(t) for t in self._tokens)
                df["spacy_doc"] = [_StubDoc(t) for t in df["text"]]
                spacy_cached = True
                print(f"  Loaded spaCy results from cache ({cache_path})")
            else:
                print(f"  Cache row count mismatch ({len(df_cache)} vs {len(df)}), reprocessing...")
        except Exception as e:
            print(f"  Cache load failed ({e}), processing fresh...")

    # Load spaCy
    if needs_spacy and not spacy_cached:
        print(f"\n  Loading spaCy model: {args.spacy_model}...")
        nlp = load_spacy_model(args.spacy_model)
        df = preprocess_texts(df, nlp)
        # Save parquet cache (tokens, lemmas, pos_tags — spacy_doc not serializable)
        try:
            cache_df = df[["tokens", "lemmas", "pos_tags"]].copy()
            cache_df.to_parquet(cache_path, index=False)
            print(f"  Saved spaCy cache to {cache_path}")
        except Exception as e:
            print(f"  [!] Failed to save cache: {e}")

    # Add doc_id
    if "doc_id" not in df.columns:
        df["doc_id"] = range(len(df))

    # Determine what to run
    results = {}
    run_all = args.all

    if run_all or args.stats:
        results["corpus_stats"] = run_corpus_stats(df, args.output)

    if run_all or args.kwic:
        node = args.kwic if args.kwic and args.kwic != "*" else None
        if node:
            results["kwic"] = run_kwic(df, node, args.window, output_dir=args.output)
        else:
            print("\n  [!] --kwic requires a node word. Example: --kwic 'hotel'")

    if run_all or args.collocations:
        results["collocations"] = run_collocations(df, args.min_freq, args.max_results, args.output)

    if run_all or args.dispersion:
        results["dispersion"] = run_dispersion(df, args.segments, args.output)

    if run_all or args.readability:
        results["readability"] = run_readability(df, args.output)

    if run_all or args.lexical_richness:
        results["lexical_richness"] = run_lexical_richness(df, args.output)

    if run_all or args.pos:
        results["pos_analysis"] = run_pos_analysis(df, args.output)

    if run_all or args.bundles:
        results["bundles"] = run_lexical_bundles(df, output_dir=args.output)

    if run_all or args.keyphrases:
        results["keyphrases"] = run_keyphrases(df, output_dir=args.output)

    if args.keyness is not None:
        ref = args.keyness if args.keyness != "*" else None
        results["keyness"] = run_keyness(df, ref, args.output)

    if args.group:
        run_group_comparison(df, "group", args.output)

    # New: comparative concordance + interactive KWIC
    if run_all or args.kwic_node:
        node = args.kwic_node
        if not node and args.kwic and args.kwic != "*":
            node = args.kwic
        if node:
            if args.group and df[args.group].nunique() >= 2:
                run_comparative_concordance(df, node, args.window, args.group, args.output)
            generate_interactive_kwic(df, node, args.window, args.output)

    # New: N-gram overlap + Key POS tags (require group column)
    if args.group and df[args.group].nunique() >= 2:
        if run_all:
            run_ngram_overlap(df, n=3, min_freq=3, group_col=args.group, output_dir=args.output)
            run_ngram_overlap(df, n=4, min_freq=3, group_col=args.group, output_dir=args.output)
            run_key_pos(df, group_col=args.group, output_dir=args.output)

    # New: grammatical error detection
    if run_all or args.error_detection:
        run_error_detection(df, args.output)

    # New: enhanced collocation network
    if run_all or args.network_enhanced:
        if HAS_NETWORKX:
            run_collocation_network_enhanced(df, args.min_freq, args.output)

    # New: time analysis
    if run_all or args.time_analysis:
        run_time_analysis(df, args.output)

    # New: lexical sophistication (wordfreq D_L)
    if run_all or args.lexical_sophistication:
        run_lexical_sophistication(df, args.output)

    # New: USAS semantic tagging
    if run_all or args.usas:
        run_usas(df, args.output)

    # New: Named Entity Recognition
    if run_all or args.ner:
        run_ner(df, args.output)

    # New: Enhanced Network Analysis
    if run_all or args.network_enhanced_v2:
        run_network_enhanced(df, args.min_freq, args.output)

    # New: USAS POS-aware tagging
    if run_all or args.usas_pos:
        run_usas_pos_aware(df, args.output)

    # New: Enhanced Keyphrase Extraction
    if run_all or args.keyphrases_enhanced:
        run_keyphrases_enhanced(df, args.output)

    # New: Enhanced Visualizations
    if run_all or args.heatmap:
        run_visualization_enhanced(df, args.output)

    # New: NLTK Text Analysis
    if run_all or args.nltk:
        run_nltk_analysis(df, args.output)

    # New: Document Clustering
    if run_all or args.clustering:
        run_clustering(df, output_dir=args.output)

    # New: Regression
    if run_all or args.regression:
        run_regression(df, target_col=args.regression_target, output_dir=args.output)

    # New: Dependency Parsing
    if run_all or args.dependency:
        run_dependency_parsing(df, args.output)

    # New: Document Similarity
    if run_all or args.similarity:
        run_document_similarity(df, args.output)

    # New: Cluster Visualization
    if run_all or args.cluster_viz:
        run_cluster_visualization(df, args.output)

    # New: Zero-Shot Classification
    if run_all or args.zero_shot:
        run_zero_shot(df, args.output)

    # Visualizations
    if run_all or args.plots or args.report:
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        plot_frequency_distribution(df, args.output)
        plot_dispersion(df, args.output)
        plot_collocation_network(df, args.output)
        plot_pos_distribution(df, args.output)

    # HTML report
    if run_all or args.report:
        generate_html_report(results, args.output)

    # Restore stdout for summary
    if args.quiet:
        sys.stdout.close()
        sys.stdout = _saved_stdout

    # Compute sections_run for manifest
    sections_run = []
    if args.all:
        sections_run = ["stats", "kwic", "collocations", "dispersion", "readability",
                        "lexical_richness", "pos", "bundles", "keyphrases", "plots",
                        "report", "error_detection", "network_enhanced", "time_analysis"]
        if args.group:
            sections_run.extend(["group_comparison", "ngram_overlap", "key_pos"])
        if args.kwic_node:
            sections_run.append("comparative_concordance")
    else:
        if args.stats: sections_run.append("stats")
        if args.kwic: sections_run.append("kwic")
        if args.collocations: sections_run.append("collocations")
        if args.dispersion: sections_run.append("dispersion")
        if args.readability: sections_run.append("readability")
        if args.lexical_richness: sections_run.append("lexical_richness")
        if args.pos: sections_run.append("pos")
        if args.bundles: sections_run.append("bundles")
        if args.keyphrases: sections_run.append("keyphrases")
        if args.keyness is not None: sections_run.append("keyness")
        if args.plots: sections_run.append("plots")
        if args.report: sections_run.append("report")
        if args.error_detection: sections_run.append("error_detection")
        if args.network_enhanced: sections_run.append("network_enhanced")
        if args.time_analysis: sections_run.append("time_analysis")
        if args.group: sections_run.append("group_comparison")
        if args.kwic_node: sections_run.append("comparative_concordance")

    # Runtime manifest
    elapsed = time.time() - _t_start
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_docs": len(df),
        "elapsed_sec": round(elapsed, 2),
        "sections_run": sections_run,
    }
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Generate index.html (Fix 5)
    generate_index_html(args.output)

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Output directory: {os.path.abspath(args.output)}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"    {f:<40} {size_str:>10}")

    return results


if __name__ == "__main__":
    main()
