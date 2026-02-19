# Implementation Roadmap: Priority Order

## PHASE 1: Quick Wins (1-2 weeks)

### 1.1 Improve Occasion Scoring (HIGHEST PRIORITY)

**Current Problem:** Just cosine similarity + softmax
**Solution:** Add attribute-aware scoring

```python
# online/occasion_scoring_v2.py
class ImprovedOccasionScorer:
    def compute_scores(self, candidates, occasion, parsed):
        # Component 1: Embedding similarity (current)
        emb_scores = self._embedding_similarity(candidates, occasion)
        
        # Component 2: Attribute matching (NEW)
        attr_scores = self._attribute_matching(candidates, occasion, parsed)
        
        # Component 3: Rule-based filtering (NEW)
        rule_scores = self._rule_based_scoring(candidates, occasion)
        
        # Weighted combination
        final_scores = (
            0.5 * emb_scores +
            0.3 * attr_scores +
            0.2 * rule_scores
        )
        return final_scores
    
    def _attribute_matching(self, candidates, occasion, parsed):
        """Explicit attribute checks"""
        occasion_requirements = {
            'wedding_guest': {
                'formality': 0.8,
                'required_categories': ['DRESSES', 'SKIRTS'],
                'forbidden_colors': ['white', 'cream'],
                'min_price': 40
            },
            'party_night_out': {
                'formality': 0.7,
                'required_categories': ['DRESSES', 'SKIRTS', 'TOPS'],
                'preferred_colors': ['black', 'red', 'gold'],
                'style': ['elegant', 'sexy', 'glamorous']
            },
            'casual_day': {
                'formality': 0.3,
                'required_categories': ['JEANS', 'T-SHIRTS', 'TROUSERS'],
                'style': ['casual', 'comfortable', 'relaxed']
            }
        }
        
        requirements = occasion_requirements.get(occasion, {})
        scores = {}
        
        for c in candidates:
            score = 0.0
            
            # Category match
            if c.category in requirements.get('required_categories', []):
                score += 0.4
            
            # Color check
            if c.color not in requirements.get('forbidden_colors', []):
                score += 0.2
            if c.color in requirements.get('preferred_colors', []):
                score += 0.2
            
            # Price check
            if c.price >= requirements.get('min_price', 0):
                score += 0.2
            
            scores[c.id] = score
        
        return scores
```

**Impact:** +15-20% occasion accuracy
**Effort:** 2-3 days
**Academic Value:** Interpretable, explainable

---

### 1.2 Add Comprehensive Evaluation

```python
# evaluation/quick_eval.py
def run_quick_evaluation():
    """Essential metrics only"""
    
    # 1. Create 50 test queries
    test_queries = create_synthetic_queries(n=50)
    
    # 2. Run pipeline
    results = []
    for query in test_queries:
        recs = pipeline.recommend(query['text'], k=10)
        results.append({
            'query': query,
            'recommendations': recs
        })
    
    # 3. Compute metrics
    metrics = {
        'P@10': compute_precision(results),
        'NDCG@10': compute_ndcg(results),
        'occasion_accuracy': compute_occasion_accuracy(results),
        'attribute_consistency': compute_attribute_consistency(results),
        'diversity': compute_diversity(results)
    }
    
    return metrics

# Run weekly to track improvements
```

**Impact:** Quantifiable progress tracking
**Effort:** 3-4 days
**Academic Value:** Essential for paper

---

## PHASE 2: Core Improvements (2-4 weeks)

### 2.1 Multi-Task Attribute Extractor

```python
# online/attribute_extractor.py
import torch.nn as nn

class AttributeExtractor(nn.Module):
    """Extract formality, style, color from images"""
    
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Task heads
        self.formality_head = nn.Linear(2048, 5)  # 5 levels
        self.style_head = nn.Linear(2048, 10)     # 10 styles
        self.color_head = nn.Linear(2048, 20)     # 20 colors
    
    def forward(self, images):
        features = self.backbone(images)
        return {
            'formality': torch.softmax(self.formality_head(features), dim=1),
            'style': torch.softmax(self.style_head(features), dim=1),
            'color': torch.softmax(self.color_head(features), dim=1)
        }

# Training:
# 1. Manually label 200 images for each task
# 2. Use weak supervision (LLM) for 1000 more
# 3. Train with multi-task loss
```

**Impact:** +20-25% overall accuracy
**Effort:** 1-2 weeks
**Academic Value:** Novel multi-task approach

---

### 2.2 Learning-to-Rank

```python
# online/learned_ranker.py
import lightgbm as lgb

class LearnedRanker:
    def __init__(self):
        self.model = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            n_estimators=100
        )
    
    def extract_features(self, candidate, query_parsed, occasion_score):
        """15+ ranking features"""
        return [
            candidate.relevance_score,
            occasion_score,
            self._price_match(candidate, query_parsed),
            self._color_exact_match(candidate, query_parsed),
            self._category_match(candidate, query_parsed),
            self._text_similarity(candidate, query_parsed),
            candidate.formality_score,
            self._style_match(candidate, query_parsed),
            self._embedding_quality(candidate),
            self._diversity_score(candidate, already_selected),
            self._popularity_score(candidate),
            self._recency_score(candidate),
            self._price_competitiveness(candidate),
            self._image_quality_score(candidate),
            self._description_completeness(candidate)
        ]
    
    def train(self, queries, relevance_labels):
        """Train on synthetic data"""
        X, y, groups = [], [], []
        
        for query in queries:
            candidates = self.retrieve(query)
            for c in candidates:
                features = self.extract_features(c, query, ...)
                X.append(features)
                y.append(relevance_labels[query.id][c.id])
            groups.append(len(candidates))
        
        self.model.fit(X, y, group=groups)

# Generate training data:
# 1. Synthetic queries with known relevant items
# 2. Manual relevance judgments (3-point scale)
# 3. 500 queries × 50 candidates = 25K training examples
```

**Impact:** +10-15% ranking quality
**Effort:** 1 week
**Academic Value:** Standard approach, well-validated

---

## PHASE 3: Advanced Features (4-6 weeks)

### 3.1 Contrastive Learning for Occasions

```python
# training/contrastive_occasion.py
class ContrastiveTrainer:
    def create_training_pairs(self):
        """Generate positive/negative pairs"""
        pairs = []
        
        # Positive: (item, suitable_occasion)
        for item in self.catalog:
            suitable_occasions = self.get_suitable_occasions(item)
            for occ in suitable_occasions:
                pairs.append((item.embedding, occ.embedding, 1))
        
        # Negative: (item, unsuitable_occasion)
        for item in self.catalog:
            unsuitable = self.get_unsuitable_occasions(item)
            for occ in unsuitable:
                pairs.append((item.embedding, occ.embedding, 0))
        
        return pairs
    
    def train(self, pairs, epochs=50):
        for epoch in range(epochs):
            for item_emb, occ_emb, label in pairs:
                loss = self.contrastive_loss(item_emb, occ_emb, label)
                loss.backward()
                self.optimizer.step()
```

**Impact:** +15-20% occasion accuracy
**Effort:** 2-3 weeks
**Academic Value:** High (novel application)

---

### 3.2 LLM-Based Reasoning (Optional but Novel)

```python
# online/llm_occasion_reasoner.py
class LLMOccasionReasoner:
    def score_with_reasoning(self, item, occasion):
        """Get score + explanation"""
        
        prompt = f"""
        Analyze if this item is suitable for {occasion}:
        
        Item: {item.name}
        Color: {item.color}
        Category: {item.category}
        Description: {item.description}
        
        Think step-by-step:
        1. What does {occasion} require?
        2. What attributes does this item have?
        3. How well do they match?
        
        Provide:
        - Suitability score (0-10)
        - Brief reasoning (2 sentences)
        """
        
        response = self.llm.invoke(prompt)
        score, reasoning = self.parse_response(response)
        
        return score / 10.0, reasoning

# Use for:
# 1. Generating training data
# 2. Explaining predictions
# 3. Handling novel occasions
```

**Impact:** +10% accuracy, high explainability
**Effort:** 1 week
**Academic Value:** Very high (cutting-edge)

---

## PHASE 4: Evaluation & Paper Writing (2-3 weeks)

### 4.1 Comprehensive Evaluation

```python
# Run all evaluation metrics
results = {
    'retrieval': evaluate_retrieval(),
    'occasion': evaluate_occasion_accuracy(),
    'attributes': evaluate_attribute_consistency(),
    'diversity': evaluate_diversity(),
    'explanations': evaluate_explanation_quality(),
    'robustness': evaluate_robustness(),
    'ablation': run_ablation_study()
}

# Generate tables and figures for paper
create_results_tables(results)
create_visualizations(results)
```

### 4.2 Ablation Study

```python
# Test impact of each component
variants = {
    'full': run_full_system(),
    'no_attributes': run_without_attributes(),
    'no_occasion': run_without_occasion(),
    'no_ltr': run_without_ltr(),
    'baseline': run_baseline()
}

# Show each component's contribution
```

---

## SUMMARY: RECOMMENDED PATH

### Minimal Viable Improvement (2 weeks):
1. ✅ Attribute-aware occasion scoring
2. ✅ Basic evaluation framework
3. ✅ Rule-based enhancements

**Result:** +20% accuracy, publishable at workshop

### Full Academic System (6 weeks):
1. ✅ Multi-task attribute extraction
2. ✅ Learning-to-rank
3. ✅ Contrastive learning
4. ✅ Comprehensive evaluation
5. ✅ Ablation studies

**Result:** +40% accuracy, publishable at top venue (RecSys, SIGIR)

### Cutting-Edge System (8 weeks):
Add:
6. ✅ LLM reasoning
7. ✅ Graph neural networks
8. ✅ Zero-shot transfer

**Result:** +50% accuracy, high-impact paper

---

## IMMEDIATE NEXT STEPS (This Week)

1. **Day 1-2:** Implement attribute-aware occasion scoring
2. **Day 3-4:** Create evaluation framework (50 test queries)
3. **Day 5:** Run baseline evaluation
4. **Day 6-7:** Analyze results, identify weaknesses

**Deliverable:** Baseline metrics + improvement plan
