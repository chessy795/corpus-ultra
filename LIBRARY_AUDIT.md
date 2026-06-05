# Library Audit: Are We Using Them to Their Full Potential?

## Libraries in corpus_ultra.py

### 1. spaCy (NLP pipeline)
**Currently using:** POS tagging, lemmatization, sentence segmentation
**NOT using:**
- ❌ Dependency parsing (subject-verb-object extraction)
- ❌ Named entity recognition (NER)
- ❌ Word vectors (semantic similarity)
- ❌ Morphological analysis

**Verdict:** Underusing. NER would be valuable for r/lonely (who/what do lonely people mention?).

---

### 2. textstat (readability)
**Currently using:** FK, FOG, SMOG, Coleman-Liau, ARI, Dale-Chall, LIX, RIX
**NOT using:**
- ❌ Linsear Write
- ❌ Text Standard (consensus score)
- ❌ Reading time estimate

**Verdict:** Mostly using. Could add reading time.

---

### 3. lexicalrichness (lexical diversity)
**Currently using:** MTLD, HD-D, Yule's K, MATTR, TTR
**NOT using:**
- ❌ D (vocd-D)
- ❌ R (Herdan's C)
- ❌ Maas measure
- ❌ Jarvis (evenness)

**Verdict:** Mostly using. vocd-D would be a good addition.

---

### 4. sklearn (machine learning)
**Currently using:** TfidfVectorizer for keyphrases
**NOT using:**
- ❌ CountVectorizer (for n-gram frequency)
- ❌ NMF (topic modeling — already in topic_ultra)
- ❌ KMeans clustering (document clustering)
- ❌ PCA/t-SNE (dimensionality reduction for visualization)

**Verdict:** Underusing. Could add document clustering.

---

### 5. keybert (keyphrase extraction)
**Currently using:** KeyBERT with default settings
**NOT using:**
- ❌ MMR diversity (Maximal Marginal Relevance)
- ❌ Different embedding models (all-MiniLM-L6-v2 vs paraphrase-multilingual)
- ❌ Different vectorizers (Count vs TF-IDF)
- ❌ Keyphrase diversity scores

**Verdict:** Underusing. MMR diversity would improve keyphrase quality.

---

### 6. networkx (graph analysis)
**Currently using:** Building collocation networks, basic metrics
**NOT using:**
- ❌ Centrality measures (degree, betweenness, closeness)
- ❌ Community detection (Louvain, Girvan-Newman)
- ❌ Graph density
- ❌ Connected components
- ❌ Cliques

**Verdict:** Underusing. Centrality and community detection would reveal important collocation clusters.

---

### 7. wordfreq (word frequency)
**Currently using:** zipf_frequency for D_L profiling
**NOT using:**
- ❌ word_frequency (alternative lookup)
- ❌ top_n_list (most common N words)
- ❌ tokenize (language-aware tokenization)
- ❌ get_frequency_dict (all frequencies as dict)
- ❌ Frequency band analysis (SUBTLEX bands)

**Verdict:** Underusing. top_n_list and frequency band analysis would be valuable.

---

### 8. USAS standalone (semantic tagging)
**Currently using:** usas_profile, usas_profile_detailed
**NOT using:**
- ❌ usas_keyness_by_category (compare two corpora)
- ❌ usas_concordance (concordance lines for a category)
- ❌ usas_frequency_by_tag (frequency by specific tag)
- ❌ POS-aware tagging (currently using fallback heuristic)

**Verdict:** Underusing. usas_keyness_by_category would be valuable for group comparison.

---

### 9. plotly (interactive visualization)
**Currently using:** Basic interactive plots
**NOT using:**
- ❌ Subplots (multiple charts in one view)
- ❌ Animations (time-based)
- ❌ 3D plots
- ❌ Heatmaps (for collocation matrices)
- ❌ Sunburst charts (for hierarchical data)

**Verdict:** Underusing. Heatmaps for collocation matrices would be valuable.

---

### 10. errant (grammatical error detection)
**Currently using:** Basic error detection
**NOT using:**
- ❌ Error categorization (spelling, grammar, style)
- ❌ Error correction suggestions
- ❌ Error density scoring

**Verdict:** Underusing. Error categorization would be valuable.

---

## Summary: What to Add

| Priority | Library | Feature | Value for r/lonely |
|----------|---------|---------|-------------------|
| HIGH | spaCy | NER | Who/what do lonely people mention? |
| HIGH | networkx | Centrality + community | What are the central concepts in lonely discourse? |
| HIGH | USAS | usas_keyness_by_category | Compare lonely vs. control semantic profiles |
| MEDIUM | keybert | MMR diversity | Better keyphrase extraction |
| MEDIUM | wordfreq | top_n_list + frequency bands | More detailed frequency profiling |
| MEDIUM | plotly | Heatmaps | Visualize collocation matrices |
| LOW | lexicalrichness | vocd-D | More robust lexical diversity |
| LOW | errant | Error categorization | Classify error types |
| LOW | spaCy | Dependency parsing | Extract subject-verb-object patterns |
