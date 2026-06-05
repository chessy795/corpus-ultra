# corpus_ultra.py Module Audit
## Following ultra-build workflow: PLAN → SPEC → SOURCE

### What Already Works (15 modules)
1. ✅ Corpus Loading & Preprocessing
2. ✅ KWIC Concordance
3. ✅ Collocation Extraction (LL, PMI, MI³, t, z, Dice, chi²)
4. ✅ Dispersion Analysis (Juilland's D, DP, DPnorm, KL, Rosengren's S)
5. ✅ Readability (FK, FOG, SMOG, Coleman-Liau, ARI, Dale-Chall, LIX, RIX)
6. ✅ Lexical Richness (MTLD, HD-D, Yule's K, MATTR, TTR)
7. ✅ POS Analysis (frequency, distribution, Biber dimensions, POS n-grams)
8. ✅ Lexical Bundles (3-5 word sequences)
9. ✅ Keyphrase Extraction (TF-IDF, TextRank, KeyBERT)
10. ✅ Corpus Statistics & Keyness (log-likelihood, log-ratio)
11. ✅ Visualization
12. ✅ HTML Report
13. ✅ Group Comparison
14. ✅ Time Analysis
15. ✅ Network (already exists!)
16. ✅ Reference Corpus (already exists!)

### Proposed Modules — Audit Results

| # | Proposed | Verdict | Reason |
|---|----------|---------|--------|
| A1 | textacy | ❌ REDUNDANT | Collocations already exist (PMI, LL, MI³, t, z, Dice, chi²). Textacy adds log-dice but that's 1 measure. Not worth a new dependency. |
| A2 | yake | ❌ REDUNDANT | Keyphrase extraction already has TF-IDF + TextRank + KeyBERT. YAKE is another approach but KeyBERT is already SOTA. |
| A3 | amrlib | ❌ OVERKILL | AMR parsing is extremely complex. For a bachelor's thesis on r/lonely, this is overkill. Maybe useful for PhD but not now. |
| A4 | pycwb | ❌ REDUNDANT | CQL queries are for corpus query. KWIC already exists. pycwb is Sketch Engine standard but adds no new analysis. |
| A5 | wordfreq | ✅ NECESSARY | Word frequency lookup for D_L (Kyle & Crossley 2015). Genuinely new. Enables lexical sophistication profiling. |
| A6 | benepar | ⚠️ MAYBE | Constituency parsing for T-unit extraction. Adds syntactic complexity. But Biber dimensions already cover this partially. Defer to embedding_ultra. |
| B1 | USAS | ✅ NECESSARY | 54K lexicon, 21 semantic categories. Genuinely new analysis. Useful for semantic profiling of r/lonely. |
| B2 | CEFR | ✅ NECESSARY | Vocabulary level profiling (A1-C2). Useful for analyzing writing proficiency in r/lonely posts. |
| B3 | Metadiscourse | ✅ NECESSARY | Hyland 2005 framework. Hedges, boosters, attitude markers. Genuinely new discourse analysis. |
| B4 | Discourse markers | ✅ NECESSARY | Halliday & Hasan + Fraser categories. Genuinely new. Useful for how lonely people construct arguments. |
| B5 | Formulaic language | ✅ NECESSARY | PMI + fixedness detection. Genuinely new. Useful for detecting "cries for help" patterns. |
| B6 | Concgrams | ⚠️ MAYBE | Directional co-occurrence. Similar to collocations but directional. Low priority — collocations already cover this. |
| B7 | Colligation | ⚠️ MAYBE | POS-tag collocation. Different from word collocation. Low priority — POS analysis already covers this partially. |
| B8 | P-frames | ⚠️ MAYBE | Open-slot n-grams. Similar to lexical bundles. Low priority — lexical bundles already cover this. |
| B9 | Network export | ✅ ALREADY EXISTS | Already in corpus_ultra! No work needed. |
| B10 | Reference corpora | ✅ ALREADY EXISTS | Already in corpus_ultra! No work needed. |
| B11 | Visualization | ✅ ALREADY EXISTS | Already in corpus_ultra! No work needed. |
| C1 | Semantic prosody | ✅ NECESSARY | Collocations + sentiment scoring. Genuinely new. "lonely" collocations — are they negative or neutral? |
| C2 | Stance detection | ⚠️ MAYBE | Modality analysis. Useful but complex. Better suited for embedding_ultra with transformers. |
| C3 | Cohesion analysis | ⚠️ MAYBE | Reference chains. Useful but complex. Better suited for embedding_ultra with sentence-transformers. |

### Final Verdict

| Category | Modules | Count |
|----------|---------|-------|
| ✅ NECESSARY | wordfreq, USAS, CEFR, metadiscourse, discourse markers, formulaic, semantic prosody | 7 |
| ⚠️ MAYBE (defer) | benepar, concgrams, colligation, p-frames, stance, cohesion | 6 |
| ❌ REDUNDANT | textacy, yake, amrlib, pycwb | 4 |
| ✅ ALREADY EXISTS | network, reference, visualization | 3 |

### Revised Plan

**Instead of 11 new modules, build 7:**

| # | Module | Library/Source | Effort |
|---|--------|---------------|--------|
| 1 | wordfreq (D_L profiling) | wordfreq | 0.5 day |
| 2 | USAS semantic tagging | corpusstats/usas.py | 0.5 day |
| 3 | CEFR vocabulary profiling | corpusstats/cefr.py | 0.5 day |
| 4 | Metadiscourse analysis | corpusstats/metadiscourse.py | 0.5 day |
| 5 | Discourse markers | corpusstats/discourse_markers.py | 0.5 day |
| 6 | Formulaic language | corpusstats/formulaic.py | 0.5 day |
| 7 | Semantic prosody | collocations + vaderSentiment | 1 day |

**Total: 4 days, not 14.**

**Defer to embedding_ultra.py:** benepar, stance, cohesion (these need transformers)

**Skip entirely:** textacy, yake, amrlib, pycwb (redundant or overkill)
