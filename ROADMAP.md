# Comprehensive Roadmap — corpus_ultra.py + Future Scripts

## What We Built Today

corpus_ultra.py now has **21 analysis modules** (14 original + 7 new):

### Original (14)
1. Corpus Loading & Preprocessing
2. KWIC Concordance
3. Collocation Extraction (LL, PMI, MI³, t, z, Dice, chi²)
4. Dispersion Analysis
5. Readability Analysis
6. Lexical Richness (MTLD, HD-D, Yule's K, MATTR, TTR)
7. POS Analysis + Biber Dimensions
8. Lexical Bundles
9. Keyphrase Extraction (TF-IDF, TextRank, KeyBERT)
10. Corpus Statistics & Keyness
11. Visualization
12. HTML Report
13. Group Comparison
14. Time Analysis

### New (7)
15. Lexical Sophistication (wordfreq D_L)
16. USAS Semantic Tagging (54K lexicon)
17. Named Entity Recognition (spaCy NER)
18. Enhanced Network Analysis (PageRank + clustering)
19. USAS POS-Aware Tagging
20. Enhanced Keyphrase Extraction (MMR diversity)
21. Enhanced Visualization (heatmaps)

---

## Roadmap: Exploiting More Library Functions

### Phase 1: NLTK Integration (Highest Value)

NLTK has 265+ functions we're not using. Key additions:

| Function | What It Does | Value for r/lonely | Effort |
|----------|-------------|-------------------|--------|
| `FreqDist.most_common(50)` | Top 50 words with counts | Better wordlist | 0.5 day |
| `FreqDist.hapaxes()` | Words appearing once | Rare vocabulary | 0.5 day |
| `Text.collocations()` | NLTK's built-in collocations | Compare with our collocations | 0.5 day |
| `Text.similar("lonely")` | Find semantically similar words | "What words are like lonely?" | 0.5 day |
| `Text.common_contexts(["lonely", "sad"])` | Shared contexts | How are lonely and sad used differently? | 0.5 day |
| `BigramCollocationFinder` | More collocation measures | Compare with our implementation | 1 day |
| `ConditionalFreqDist` | Frequency by group | Word freq in popular vs unpopular | 0.5 day |
| `stopwords.words("english")` | Stopword lists | Better preprocessing | 0.5 day |
| `SentimentIntensityAnalyzer` | VADER sentiment | Quick sentiment for semantic prosody | 0.5 day |
| `ne_chunk` | Named entity chunking | Compare with spaCy NER | 0.5 day |

### Phase 2: spaCy Deep Integration

| Feature | What It Does | Value | Effort |
|---------|-------------|-------|--------|
| `token.dep_` | Dependency parsing | Subject-verb-object extraction | 1 day |
| `token.head` | What does word modify? | Sentence structure | 1 day |
| `doc.similarity()` | Document similarity | Which posts are most alike? | 0.5 day |
| `token.morph` | Morphological features | Tense, number, case analysis | 0.5 day |
| `doc.noun_chunks` | Noun phrases | "what kind of X do lonely people talk about?" | 0.5 day |

### Phase 3: sklearn Integration

| Feature | What It Does | Value | Effort |
|---------|-------------|-------|--------|
| `KMeans` | Document clustering | Auto-discover post types | 1 day |
| `PCA` | Dimensionality reduction | Visualize clusters | 0.5 day |
| `cosine_similarity` | Document similarity | Find duplicate/near-duplicate posts | 0.5 day |
| `CountVectorizer` | Raw n-gram counts | Compare with TF-IDF | 0.5 day |
| `NMF` | Topic modeling | Already in topic_ultra but could add | 1 day |

### Phase 4: networkx Deep Integration

| Feature | What It Does | Value | Effort |
|---------|-------------|-------|--------|
| `nx.pagerank()` | PageRank | Most important words | 0.5 day |
| `nx.clustering()` | Clustering coefficient | How tightly connected are concepts? | 0.5 day |
| `nx.connected_components()` | Find subgraphs | Are there separate "lonely" communities? | 0.5 day |
| `nx.density()` | Graph density | How connected is the network? | 0.5 day |
| `nx.shortest_path()` | Path between words | "lonely → happy: how many steps?" | 0.5 day |
| `nx.average_clustering()` | Global clustering | Is the network clustered? | 0.5 day |

### Phase 5: New Libraries to Add

| Library | What It Does | Value | Effort |
|---------|-------------|-------|--------|
| **NLTK** | Classic NLP toolkit | 265+ functions available | 3-5 days |
| **textacy** | Corpus analysis | Better collocations, keyterms | 2 days |
| **yake** | Unsupervised keyword extraction | No training needed | 0.5 day |
| **VADER** | Social media sentiment | Calibrated for Reddit | 0.5 day |
| **gensim** | Topic modeling, word2vec | Word embeddings for r/lonely | 2 days |
| **emoji** | Emoji processing | Detect emotional expression in posts | 0.5 day |
| **langdetect** | Language detection | Filter non-English posts | 0.5 day |
| **dateutil** | Date parsing | Better time analysis | 0.5 day |

---

## Roadmap: New Ultra Scripts

### Script 1: embedding_ultra.py (Highest Priority)

**What it does:** Word embeddings, semantic similarity, NER, coreference, SRL

**Libraries:**
- `sentence-transformers` — semantic similarity (already in sentiscape)
- `GLiNER` — zero-shot NER (no training needed)
- `fastcoref` — coreference resolution
- `Word2Vec/GloVe` — word embeddings
- `cosine_similarity` — semantic similarity

**Research value for r/lonely:**
- "How similar is 'lonely' to 'sad'?" (0.0-1.0)
- "What words are closest to 'lonely' in embedding space?"
- "Do lonely posts have different word associations than control?"
- "Who is 'I' referring to in lonely posts?" (coreference)
- "Who did what to whom?" (SRL)

### Script 2: discourse_ultra.py (Medium Priority)

**What it does:** Metadiscourse, stance, discourse markers

**Libraries:**
- Custom keyword lists (Halliday & Hasan + Fraser)
- `transformers` fine-tuned for stance detection
- `vaderSentiment` for hedging/boosting

**Research value for r/lonely:**
- How do lonely people position themselves? (stance)
- How do they manage reader relationships? (metadiscourse)
- How do they connect ideas? (discourse markers)

### Script 3: sentiment_ultra.py (Low Priority)

**What it does:** Enhanced sentiment analysis (you already have sentiscape_ultra)

**Libraries:**
- `transformers` (already have)
- `vaderSentiment` (already have)
- Custom loneliness-specific sentiment lexicon

**Research value for r/lonely:**
- Loneliness-specific emotion detection
- Temporal sentiment changes
- Sentiment by topic

---

## Dr. Mo — Should You Reach Out?

**Yes. Absolutely.** Here's why:

1. She teaches "Language and social data analytics" — you're building language data analytics tools
2. She publishes on AI/ML — your scripts are AI/ML
3. She's at PolyU — same university as your PhD proposal
4. "AI for humanity" — loneliness research IS AI for humanity
5. She says "Happy to discuss more if you're interested in AI projects"

**What to send her:**
1. A 1-page summary (not code) of what corpus_ultra.py does
2. A sample HTML report from r/lonely data
3. A brief description of your research goals

**Don't send:**
- Raw code (too much)
- Technical jargon (she's AI, not corpus linguistics)
- A long email (keep it short)

**Sample email:**
```
Subject: NLP tools for loneliness research — would love your feedback

Dear Dr. Mo,

I'm building Python tools for analyzing social media data on loneliness.
The library does 21 different NLP analyses on Reddit posts, including
NER, semantic tagging, collocation networks, and sentiment analysis.

I'd love your feedback on the architecture and approach.

Would you have 15 minutes to look at the HTML report?

Best regards,
Peter
```

---

## Priority Order

| Priority | Task | Time | Impact |
|----------|------|------|--------|
| 1 | Email Dr. Mo with summary | 1 day | HIGH — potential collaborator |
| 2 | Add NLTK integration | 3 days | HIGH — 265+ new functions |
| 3 | Build embedding_ultra.py | 5 days | HIGH — word embeddings |
| 4 | Test all scripts on r/lonely | 2 days | HIGH — validation |
| 5 | Add spaCy deep features | 3 days | MEDIUM — dependency parsing |
| 6 | Add sklearn features | 2 days | MEDIUM — document clustering |
| 7 | Build discourse_ultra.py | 3 days | MEDIUM — metadiscourse |
| 8 | Build chatbot prototype | 2 days | LOW — separate project |

**Total estimated time: ~20 days**
