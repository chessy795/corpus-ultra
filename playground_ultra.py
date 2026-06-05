"""corpus_ultra.py playground — test all new modules on sample data.
   python playground_ultra.py"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\mwp5a\Desktop\Paper\corpusstats')

from lexical_sophistication import compute_lexical_sophistication, compute_word_frequency_profile
from corpusstats.usas import usas_profile
from corpusstats.cefr import cefr_profile
from corpusstats.metadiscourse import metadiscourse_profile
from corpusstats.discourse_markers import discourse_marker_density
from corpusstats.formulaic import detect_formulaic

# Sample texts (simulating r/lonely posts)
TEXTS = [
    "I feel so lonely today. Nobody understands me. I've been feeling this way for years.",
    "My cat makes me happy. She follows me around the house. I love her so much.",
    "Research shows that social media usage correlates with loneliness. Participants reported higher isolation.",
    "I don't know what to do anymore. I try to reach out but people are always busy.",
    "The night is dark and full of stars. I sit alone and count them one by one.",
]

print("=" * 60)
print("CORPUS ULTRA — NEW MODULES PLAYGROUND")
print("=" * 60)
print()

for i, text in enumerate(TEXTS):
    tokens = text.lower().split()
    print(f"TEXT {i+1}: {text[:60]}...")
    print("-" * 60)

    # 1. Lexical Sophistication
    ls = compute_lexical_sophistication(tokens)
    print(f"  LEXICAL SOPHISTICATION: D_L={ls['d_l']:.1f}%  mean_zipf={ls['mean_zipf']:.2f}")

    # 2. USAS
    us = usas_profile(tokens, return_result=True)
    top_cats = us.df.head(3)
    cats_str = ", ".join(f"{r['category']}:{r['percentage']:.0f}%" for _, r in top_cats.iterrows())
    print(f"  USAS: {cats_str}")

    # 3. CEFR
    cv = cefr_profile(tokens)
    print(f"  CEFR: A1={cv.get('A1', 0):.0f}%  A2={cv.get('A2', 0):.0f}%  off={cv.get('off_list', 0):.0f}%")

    # 4. Metadiscourse
    md = metadiscourse_profile(tokens)
    total = md.get('total_markers', 0)
    cats = md.get('category_distribution', {})
    print(f"  METADISCOURSE: {total} markers  categories={list(cats.keys())}")

    # 5. Discourse Markers
    dm = discourse_marker_density(tokens)
    top_dm = sorted(dm.items(), key=lambda x: -x[1])[:3]
    dm_str = ", ".join(f"{k}:{v:.0f}" for k, v in top_dm if k not in ('total', 'n_markers'))
    print(f"  DISCOURSE: {dm_str}")

    # 6. Formulaic Language
    fl = detect_formulaic(tokens, min_freq=1, min_pmi=0, min_fixedness=0)
    if fl:
        print(f"  FORMULAIC: {len(fl)} sequences  top={fl[0]['sequence']}")
    else:
        print(f"  FORMULAIC: none found")

    print()

print("=" * 60)
print("ALL MODULES WORK")
print("=" * 60)
