# EXHAUSTIVE LIBRARY AUDIT — corpus_ultra.py

## 1. spaCy (NLP pipeline)
**Available:** 100+ methods on Doc, Token, Span, Entity objects
**Using (7):**
- `nlp(text)` — process text
- `doc.sents` — sentence segmentation
- `doc.noun_chunks` — noun phrases
- `token.lemma_` — lemmatization
- `token.lower()` — lowercase
- `token.text` — original text
- `doc.pipe()` — batch processing

**NOT using (high value):**
- ❌ `token.dep_` — dependency label (nsubj, dobj, etc.)
- ❌ `token.head` — dependency head (what does this word modify?)
- ❌ `doc.ents` — named entities (people, places, organizations)
- ❌ `token.vector` — word vector (300-dim semantic representation)
- ❌ `doc.similarity(other)` — document similarity
- ❌ `token.morph` — morphological features (tense, number, case)

---

## 2. textstat (readability)
**Available:** 50 functions
**Using (1):**
- `textstat.text_standard()` — consensus readability score

**NOT using (49):**
- ❌ `flesch_reading_ease()` — we have custom impl
- ❌ `flesch_kincaid_grade()` — we have custom impl
- ❌ `gunning_fog()` — we have custom impl
- ❌ `smog_index()` — we have custom impl
- ❌ `coleman_liau_index()` — we have custom impl
- ❌ `automated_readability_index()` — we have custom impl
- ❌ `dale_chall_readability_score()` — we have custom impl
- ❌ `linsear_write_formula()` — NOT IMPLEMENTED
- ❌ `reading_time()` — NOT IMPLEMENTED
- ❌ `difficult_words()` — NOT IMPLEMENTED
- ❌ `difficult_words_list()` — NOT IMPLEMENTED
- ❌ `spache_readability()` — NOT IMPLEMENTED
- ❌ `mcalpine_eflaw()` — NOT IMPLEMENTED

---

## 3. lexicalrichness (lexical diversity)
**Available:** 6 methods
**Using (3):**
- `.mtld()` — Measure of Textual Lexical Diversity
- `.hdd()` — Hypergeometric Distribution D
- `.mattr()` — Moving Average Type-Token Ratio

**NOT using (3):**
- ❌ `.vocd()` — vocd-D (vocabulary diversity)
- ❌ `.msttr()` — Mean Segmental Type-Token Ratio
- ❌ `.vocd_fig()` — vocd visualization

**Note:** We have custom implementations for TTR, Yule's K, RTTR, LTTR, MSTTR that overlap with this library.

---

## 4. sklearn (machine learning)
**Available:** 1000+ classes/functions
**Using (2):**
- `TfidfVectorizer()` — TF-IDF for keyphrase extraction
- `vectorizer.fit_transform()` — fit and transform

**NOT using (high value):**
- ❌ `CountVectorizer()` — raw n-gram counts
- ❌ `NMF()` — Non-negative Matrix Factorization (topic modeling)
- ❌ `KMeans()` — document clustering
- ❌ `PCA()` — dimensionality reduction
- ❌ `t-SNE()` — visualization of document clusters
- ❌ `cosine_similarity()` — document similarity
- ❌ `TfidfVectorizer(ngram_range=(2,3))` — bigram/trigram TF-IDF

---

## 5. keybert (keyphrase extraction)
**Available:** 5 methods
**Using (1):**
- `kw_model.extract_keywords()` — extract keyphrases

**NOT using (4):**
- ❌ `extract_keywords(docs, keyphrase_ngram_range=(2,3))` — bigram keyphrases
- ❌ `extract_keywords(docs, use_mmr=True, diversity=0.7)` — MMR diversity
- ❌ `extract_keywords(docs, use_maxsum=True)` — MaxSum diversity
- ❌ `extract_keywords(docs, top_n=20, stop_words='english')` — custom stopwords

---

## 6. networkx (graph analysis)
**Available:** 759 functions
**Using (10):**
- `nx.Graph()` — create graph
- `G.add_edge()` — add edge
- `G.nodes` — list nodes
- `G.edges` — list edges
- `G.degree` — node degrees
- `G.number_of_nodes()` — node count
- `G.number_of_edges()` — edge count
- `nx.spring_layout()` — layout algorithm
- `nx.betweenness_centrality()` — betweenness centrality
- `nx.eigenvector_centrality()` — eigenvector centrality
- `nx.community` — community detection
- `nx.draw_networkx_*()` — visualization

**NOT using (high value):**
- ❌ `nx.closeness_centrality()` — how close is a node to all others?
- ❌ `nx.degree_centrality()` — simple degree-based centrality
- ❌ `nx.pagerank()` — Google's PageRank algorithm
- ❌ `nx.clustering()` — clustering coefficient
- ❌ `nx.connected_components()` — find disconnected subgraphs
- ❌ `nx.density()` — graph density (how connected?)
- ❌ `nx.diameter()` — longest shortest path
- ❌ `nx.shortest_path()` — find path between two nodes
- ❌ `nx.minimum_spanning_tree()` — minimum spanning tree
- ❌ `nx.average_clustering()` — average clustering coefficient
- ❌ `nx.transitivity()` — global clustering coefficient
- ❌ `nx.average_shortest_path_length()` — average path length

---

## 7. wordfreq (word frequency)
**Available:** 24 functions
**Using (2):**
- `zipf_frequency(word, lang)` — Zipf scale frequency
- `word_frequency(word, lang)` — raw frequency

**NOT using (22):**
- ❌ `top_n_list(lang, n)` — most common N words
- ❌ `tokenize(text, lang)` — language-aware tokenization
- ❌ `get_frequency_dict(lang)` — all frequencies as dict
- ❌ `iter_wordlist(lang)` — iterate all words
- ❌ `available_languages()` — list supported languages
- ❌ `get_language_info(lang)` — language metadata
- ❌ `random_words(lang, n)` — random word generation
- ❌ `freq_to_zipf(freq)` — convert frequency to Zipf
- ❌ `zipf_to_freq(zipf)` — convert Zipf to frequency

---

## 8. USAS standalone (semantic tagging)
**Available:** 8 functions
**Using (3):**
- `usas_profile(tokens)` — category distribution
- `usas_profile_detailed(tokens)` — sub-category distribution
- `usas_keyness_by_category(focus, ref)` — compare two corpora

**NOT using (5):**
- ❌ `tag_tokens(tokens, pos_tags)` — POS-aware tagging (using fallback heuristic)
- ❌ `usas_frequency_by_tag(tokens, tag)` — frequency for specific tag
- ❌ `usas_frequency_list(tokens)` — full frequency list
- ❌ `usas_concordance(tokens, tag, window)` — concordance for category
- ❌ POS-aware tagging (currently using fallback, not spaCy POS)

---

## 9. plotly (interactive visualization)
**Available:** 100+ chart types
**Using (3):**
- `go.Bar()` — bar charts
- `fig.add_trace()` — add data
- `fig.update_layout()` — customize layout
- `fig.write_html()` — export to HTML

**NOT using (high value):**
- ❌ `go.Heatmap()` — heatmaps (for collocation matrices)
- ❌ `go.Scatter()` — scatter plots (for document clusters)
- ❌ `go.Sunburst()` — hierarchical data
- ❌ `go.Treemap()` — treemap visualization
- ❌ `go.ParallelCoordinates()` — multi-dimensional comparison
- ❌ `make_subplots()` — multiple charts in one view
- ❌ `go.FigureWidget()` — interactive widgets

---

## 10. errant (grammatical error detection)
**Available:** 10+ methods
**Using (4):**
- `errant.load()` — load model
- `annotator.annotate()` — detect errors
- `edit.type` — error type
- `edit.o_toks_str` — original text

**NOT using (6):**
- ❌ `edit.c_toks_str` — corrected text
- ❌ `edit.o_start`, `edit.o_end` — error position
- ❌ Error categorization (spelling, grammar, style)
- ❌ Error density scoring (errors per word)
- ❌ Error type distribution

---

## Summary: Usage Depth

| Library | Available | Using | % Used | Priority to Expand |
|---------|-----------|-------|--------|-------------------|
| spaCy | 100+ | 7 | 7% | HIGH (NER, dependency) |
| textstat | 50 | 1 | 2% | LOW (custom impl) |
| lexicalrichness | 6 | 3 | 50% | LOW |
| sklearn | 1000+ | 2 | 0.2% | MEDIUM (clustering) |
| keybert | 5 | 1 | 20% | MEDIUM (MMR diversity) |
| networkx | 759 | 10 | 1.3% | HIGH (centrality, clustering) |
| wordfreq | 24 | 2 | 8% | MEDIUM (top_n_list) |
| USAS | 8 | 3 | 38% | HIGH (POS-aware, concordance) |
| plotly | 100+ | 3 | 3% | MEDIUM (heatmaps) |
| errant | 10+ | 4 | 40% | LOW |

## Top 5 Features to Add

1. **spaCy NER** — `doc.ents` → extract named entities from lonely posts
2. **networkx PageRank + clustering** — find central concepts in collocation networks
3. **USAS POS-aware tagging** — use spaCy POS for better semantic tagging
4. **keybert MMR diversity** — `use_mmr=True, diversity=0.7` for better keyphrases
5. **plotly Heatmap** — `go.Heatmap()` for collocation matrices
