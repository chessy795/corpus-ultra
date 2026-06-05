"""USAS semantic field tagger (standalone, no corpusstats dependency).

Tags words with semantic categories (A-Z) using the UCREL USAS lexicon.
Based on Rayson et al. (2004) UCREL Semantic Analysis System.

Reference: https://ucrel.lancs.ac.uk/usas/
Lexicon source: https://github.com/UCREL/Multilingual-USAS

Data files required in data/ subdirectory:
  - usas_english.tsv (54K word → category mapping)
  - usas_english_mwe.tsv (multi-word expressions)
"""

import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# USAS tag descriptions (top-level categories)
USAS_CATEGORIES = {
    'A': 'General & Abstract Terms',
    'B': 'The Body & the Individual',
    'C': 'Arts & Crafts',
    'D': 'Emotion',
    'E': 'Food & Agriculture',
    'F': 'Movement & Transportation',
    'G': 'Government & Public',
    'H': 'Architecture & Buildings',
    'I': 'Money & Commerce',
    'J': 'Arts & Entertainment',
    'K': 'Sport & Games',
    'L': 'Life & Living Things',
    'M': 'Movement & Transportation',
    'N': 'Numbers & Measurement',
    'O': 'Substances & Materials',
    'P': 'Education',
    'Q': 'Linguistics & Communication',
    'R': 'Science & Technology',
    'S': 'Sensory & Physical',
    'T': 'Time',
    'U': 'Universe',
    'W': 'The World & Environment',
    'X': 'Psychological Actions',
    'Y': 'Science & Technology',
    'Z': 'Names & Grammar',
}

# Tag descriptions for common sub-tags
USAS_TAG_DESCRIPTIONS = {
    'A5': 'Evaluation',
    'A6': 'Comparing',
    'A7': 'Likelihood',
    'A8': 'Seems',
    'A9': 'Getting & Giving',
    'B1': 'Anatomy & Physiology',
    'B2': 'Health & Disease',
    'B3': 'Medicine',
    'B4': 'Cleaning & Personal Care',
    'D1': 'General Emotion',
    'D2': 'Liking',
    'D3': 'Calm/Violent/Angry',
    'D4': 'Happy/Sad',
    'D5': 'Fear/Bravery/Shock',
    'E1': 'Food',
    'E2': 'Drink',
    'E3': 'Agriculture',
    'E4': 'Kinetic',
    'F1': 'Moving',
    'F2': 'Putting & Taking',
    'F3': 'Loading',
    'G1': 'Government',
    'G2': 'Politics',
    'I1': 'Money Generally',
    'I2': 'Business',
    'I3': 'Work',
    'K1': 'Entertainment',
    'K5': 'Sports & Games',
    'L1': 'Life & Living Things',
    'L2': 'Living creatures',
    'M1': 'Moving',
    'M2': 'Putting & Taking',
    'M3': 'Loading',
    'N1': 'Numbers',
    'N2': 'Mathematics',
    'N3': 'Measurement',
    'N5': 'Quantities',
    'O1': 'Substances & Materials',
    'O2': 'Objects',
    'P1': 'Education',
    'Q1': 'Communication',
    'Q2': 'Linguistics',
    'R1': 'Science',
    'S1': 'General Sensory',
    'S2': 'Sound',
    'S3': 'Taste',
    'S4': 'Smell',
    'S5': 'Touch',
    'T1': 'Time',
    'T2': 'Time: Beginning & Ending',
    'T3': 'Time: Old & New',
    'X1': 'General Psychological',
    'X2': 'Mental Actions',
    'X3': 'Sensory',
    'X4': 'Judgment',
    'X5': 'Attention',
    'X6': 'Deciding',
    'X7': 'Wanting',
    'X8': 'Trying',
    'X9': 'Ability',
    'Z1': 'Personal Names',
    'Z2': 'Geographical Names',
    'Z3': 'Other Proper Names',
    'Z4': 'Discourse',
    'Z5': 'Grammatical Bin',
    'Z6': 'Negative',
    'Z7': 'If',
    'Z8': 'Pronouns',
    'Z9': 'Trash Can',
    'Z99': 'Unassigned',
}


def _load_usas_lexicon() -> Dict[Tuple[str, str], List[str]]:
    """Load USAS lexicon from data file.

    Returns dict mapping (lemma, pos) -> list of semantic tags.
    """
    data_path = Path(__file__).parent / "data" / "usas_english.tsv"
    if not data_path.exists():
        return {}

    lexicon = {}
    with open(data_path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                lemma, pos, tags = parts[0], parts[1], parts[2]
                # Split multiple tags (e.g., "M1/N3.8+")
                tag_list = tags.split()
                lexicon[(lemma.lower(), pos)] = tag_list
    return lexicon


def _load_usas_mwe() -> Dict[str, List[str]]:
    """Load USAS MWE patterns from data file.

    Returns dict mapping MWE template -> list of semantic tags.
    """
    data_path = Path(__file__).parent / "data" / "usas_english_mwe.tsv"
    if not data_path.exists():
        return {}

    mwe_patterns = {}
    with open(data_path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                template, tags = parts[0], parts[1]
                mwe_patterns[template.lower()] = tags.split()
    return mwe_patterns


def tag_tokens(
    tokens: List[str],
    pos_tags: Optional[List[str]] = None,
    lexicon: Optional[Dict] = None,
) -> List[Dict]:
    """Tag tokens with USAS semantic categories.

    Parameters
    ----------
    tokens : list of str
        Lowercased tokens.
    pos_tags : list of str, optional
        POS tags (spaCy format). If None, uses simple heuristic.
    lexicon : dict, optional
        Pre-loaded lexicon. If None, loads from file.

    Returns
    -------
    list of dicts with keys: token, pos, tags, primary_tag, category, category_name
    """
    if lexicon is None:
        lexicon = _load_usas_lexicon()
    if not lexicon:
        return [{'token': t, 'pos': '?', 'tags': ['Z99'], 'primary_tag': 'Z99',
                 'category': 'Z', 'category_name': 'Names & Grammar'}
                for t in tokens]

    # Map spaCy POS to USAS POS
    def _map_pos(spacy_pos):
        mapping = {
            'NOUN': 'NOUN', 'PROPN': 'NOUN', 'VERB': 'VERB',
            'ADJ': 'ADJ', 'ADV': 'ADV', 'PRON': 'PRON',
            'DET': 'DET', 'ADP': 'ADP', 'CCONJ': 'CCONJ',
            'SCONJ': 'CCONJ', 'NUM': 'NUM', 'PART': 'PART',
            'INTJ': 'INTJ', 'AUX': 'VERB', 'X': 'X',
        }
        return mapping.get(spacy_pos, 'X')

    results = []
    for i, token in enumerate(tokens):
        pos = _map_pos(pos_tags[i]) if pos_tags else 'X'
        key = (token.lower(), pos)

        # Try exact match, then POS-agnostic, then just lemma
        tags = lexicon.get(key)
        if tags is None:
            tags = lexicon.get((token.lower(), 'NOUN'))
        if tags is None:
            tags = lexicon.get((token.lower(), 'VERB'))
        if tags is None:
            tags = lexicon.get((token.lower(), 'ADJ'))
        if tags is None:
            tags = ['Z99']  # unassigned

        # Primary tag = first tag, strip modifiers
        primary = tags[0].rstrip('+-').split('/')[0] if tags else 'Z99'
        # Get top-level category
        top_level = primary[0] if primary else 'Z'

        results.append({
            'token': token,
            'pos': pos,
            'tags': tags,
            'primary_tag': primary,
            'category': top_level,
            'category_name': USAS_CATEGORIES.get(top_level, 'Unknown'),
        })

    return results


def usas_profile(
    tokens: List[str],
    pos_tags: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute USAS semantic field profile.

    Parameters
    ----------
    tokens : list of str
    pos_tags : list of str, optional

    Returns
    -------
    dict mapping top-level category -> percentage.
    """
    tagged = tag_tokens(tokens, pos_tags)
    total = max(len(tagged), 1)
    cat_counts = Counter(t['category'] for t in tagged)
    return {cat: round(count / total * 100, 2)
            for cat, count in sorted(cat_counts.items())}


def usas_profile_detailed(
    tokens: List[str],
    pos_tags: Optional[List[str]] = None,
    top_n: int = 20,
) -> List[Dict]:
    """Compute detailed USAS profile with sub-categories.

    Returns list of dicts with tag, count, percentage, description.
    """
    tagged = tag_tokens(tokens, pos_tags)
    total = max(len(tagged), 1)
    tag_counts = Counter(t['primary_tag'] for t in tagged)
    results = []
    for tag, count in tag_counts.most_common(top_n):
        results.append({
            'tag': tag,
            'count': count,
            'percentage': round(count / total * 100, 2),
            'description': USAS_TAG_DESCRIPTIONS.get(tag, 'Unknown'),
        })
    return results


def usas_keyness_by_category(
    focus_tokens: List[str],
    reference_tokens: List[str],
    focus_pos: Optional[List[str]] = None,
    ref_pos: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Compare USAS category distribution between two corpora.

    Returns dict mapping category -> {focus_pct, ref_pct, keyness, log_ratio}.
    """
    focus_profile = usas_profile(focus_tokens, focus_pos)
    ref_profile = usas_profile(reference_tokens, ref_pos)

    all_cats = set(focus_profile.keys()) | set(ref_profile.keys())
    results = {}
    for cat in sorted(all_cats):
        f_pct = focus_profile.get(cat, 0.0)
        r_pct = ref_profile.get(cat, 0.0)
        # Simple log-ratio keyness
        if r_pct > 0 and f_pct > 0:
            import math
            log_ratio = math.log2(f_pct / r_pct)
        elif f_pct > 0:
            log_ratio = float('inf')
        else:
            log_ratio = 0.0
        results[cat] = {
            'focus_pct': f_pct,
            'ref_pct': r_pct,
            'log_ratio': log_ratio,
            'description': USAS_CATEGORIES.get(cat, 'Unknown'),
        }
    return results
