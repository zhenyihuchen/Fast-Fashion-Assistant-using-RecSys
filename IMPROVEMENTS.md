# Technical Robustness Improvements

## 1. Multi-Modal Fusion Architecture

### Current: Single CLIP model
### Improved: Ensemble of specialized models

```python
# Add to candidate_retrieval.py
def ensemble_retrieval(query, parsed, topk=200):
    """Combine multiple embedding models"""
    
    # 1. CLIP ViT-L/14 (better than ViT-B/32)
    clip_scores = retrieve_with_clip(query, model="ViT-L-14")
    
    # 2. Fashion-specific model (e.g., FashionCLIP, DeepFashion)
    fashion_scores = retrieve_with_fashion_clip(query)
    
    # 3. Text-only model for attribute matching
    text_scores = retrieve_with_sentence_transformer(query)
    
    # Weighted fusion
    final_scores = (
        0.5 * clip_scores + 
        0.3 * fashion_scores + 
        0.2 * text_scores
    )
    return final_scores
```

**Models to add:**
- **FashionCLIP**: Fine-tuned on fashion data
- **BLIP-2**: Better image-text understanding
- **Sentence-BERT**: For text attribute matching
- **ResNet + Attribute Classifier**: Explicit color/pattern detection

---

## 2. Learning-to-Rank (LTR) Layer

### Replace simple weighted sum with learned ranker

```python
# Add to final_ranking.py
import lightgbm as lgb

class LearnedRanker:
    def __init__(self):
        self.model = lgb.LGBMRanker(objective='lambdarank')
    
    def extract_features(self, candidate, query_parsed):
        """Extract ranking features"""
        return {
            'relevance_score': candidate.relevance,
            'occasion_score': candidate.occasion_score,
            'price_match': self._price_match(candidate, query_parsed),
            'color_exact_match': self._color_match(candidate, query_parsed),
            'category_match': self._category_match(candidate, query_parsed),
            'popularity': candidate.click_count,  # if available
            'recency': candidate.days_since_added,
            'text_similarity': self._text_sim(candidate, query_parsed),
        }
    
    def rank(self, candidates, query_parsed):
        features = [self.extract_features(c, query_parsed) for c in candidates]
        scores = self.model.predict(features)
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

---

## 3. Query Understanding Enhancement

### Add query expansion and reformulation

```python
# Add to query_processing_llm.py
def expand_query(query, parsed):
    """Expand query with synonyms and related terms"""
    
    # 1. Synonym expansion
    synonyms = {
        'dress': ['gown', 'frock', 'outfit'],
        'party': ['celebration', 'event', 'gathering'],
        'red': ['crimson', 'scarlet', 'burgundy']
    }
    
    # 2. Attribute inference
    if 'party' in query and 'night' in query:
        parsed['inferred_attributes'] = ['elegant', 'formal', 'dressy']
    
    # 3. Multi-query generation for diversity
    queries = [
        query,  # original
        f"{parsed['category']} in {parsed['color']}",  # structured
        f"{parsed['occasion']} outfit {parsed['style']}"  # occasion-focused
    ]
    
    return queries
```

---

## 4. Re-Ranking with Cross-Encoders

### Add fine-grained relevance scoring

```python
# Add to candidate_retrieval.py
from sentence_transformers import CrossEncoder

def rerank_with_cross_encoder(query, candidates, top_k=50):
    """Re-rank top candidates with cross-encoder"""
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create query-product pairs
    pairs = [(query, f"{c.name} {c.description} {c.color}") for c in candidates[:100]]
    
    # Score pairs
    scores = model.predict(pairs)
    
    # Re-rank
    reranked = sorted(zip(candidates[:100], scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]
```

---

## 5. Personalization Layer

### Add user preference modeling

```python
# New file: online/personalization.py
class UserProfiler:
    def __init__(self):
        self.user_embeddings = {}
    
    def update_profile(self, user_id, interactions):
        """Learn from clicks, purchases, dwell time"""
        clicked_items = [i.product_id for i in interactions if i.clicked]
        purchased_items = [i.product_id for i in interactions if i.purchased]
        
        # Aggregate embeddings
        user_emb = np.mean([get_embedding(pid) for pid in clicked_items], axis=0)
        self.user_embeddings[user_id] = user_emb
    
    def personalize_scores(self, user_id, candidates):
        """Boost items similar to user preferences"""
        if user_id not in self.user_embeddings:
            return candidates
        
        user_emb = self.user_embeddings[user_id]
        for c in candidates:
            c.score += 0.2 * cosine_similarity(user_emb, c.embedding)
        
        return sorted(candidates, key=lambda x: x.score, reverse=True)
```

---

## 6. Attribute-Aware Filtering

### Add explicit attribute extractors

```python
# Add to candidate_retrieval.py
class AttributeExtractor:
    def __init__(self):
        self.color_model = load_color_classifier()
        self.pattern_model = load_pattern_classifier()
    
    def extract_visual_attributes(self, image):
        """Extract attributes from image"""
        return {
            'dominant_color': self.color_model.predict(image),
            'pattern': self.pattern_model.predict(image),
            'style': self.style_classifier.predict(image),
            'formality': self.formality_scorer.predict(image)
        }
    
    def hard_filter(self, candidates, query_parsed):
        """Apply strict attribute filters"""
        filtered = []
        for c in candidates:
            attrs = self.extract_visual_attributes(c.image)
            if self._matches_constraints(attrs, query_parsed):
                filtered.append(c)
        return filtered
```

---

## 7. Diversity & Fairness

### Ensure diverse recommendations

```python
# Add to final_ranking.py
def diversify_results(ranked_items, alpha=0.7):
    """MMR-based diversification"""
    selected = [ranked_items[0]]
    candidates = ranked_items[1:]
    
    while len(selected) < 10 and candidates:
        mmr_scores = []
        for c in candidates:
            relevance = c.score
            max_sim = max([cosine_sim(c.emb, s.emb) for s in selected])
            mmr = alpha * relevance - (1 - alpha) * max_sim
            mmr_scores.append((c, mmr))
        
        best = max(mmr_scores, key=lambda x: x[1])
        selected.append(best[0])
        candidates.remove(best[0])
    
    return selected
```

---

## 8. Contextual Embeddings

### Add session context

```python
# New file: online/context_encoder.py
def encode_with_context(query, session_history):
    """Encode query with session context"""
    
    # Concatenate recent queries
    context = " ".join([h.query for h in session_history[-3:]])
    enriched_query = f"{context} [SEP] {query}"
    
    # Encode with context-aware model
    embedding = model.encode(enriched_query)
    return embedding
```

---

## Summary of Improvements

| Component | Current | Improved |
|-----------|---------|----------|
| Embedding | Single CLIP | Ensemble (CLIP + FashionCLIP + BERT) |
| Ranking | Weighted sum | LightGBM LTR with 10+ features |
| Query | Direct parse | Expansion + reformulation |
| Re-ranking | None | Cross-encoder top-100 |
| Personalization | None | User profile embeddings |
| Attributes | Text-based | Visual + text classifiers |
| Diversity | None | MMR diversification |
| Context | Stateless | Session-aware encoding |
