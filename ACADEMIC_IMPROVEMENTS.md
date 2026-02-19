# Academic-Grade Improvements for Multimodal Conversational RecSys

## CURRENT PIPELINE ANALYSIS

### Strengths:
✅ Multimodal (image + text embeddings)
✅ Conversational (multi-turn dialogue)
✅ Occasion-aware scoring
✅ LLM-based query understanding
✅ Explainable recommendations

### Weaknesses:
❌ Simple weighted fusion (α*relevance + β*occasion)
❌ Occasion scoring: just cosine similarity + softmax
❌ No learning from interactions
❌ No explicit attribute modeling
❌ Limited evaluation framework
❌ No uncertainty quantification

---

## PART 1: ROBUST OCCASION SUITABILITY SCORING

### Current Approach (Weak):
```python
# Just cosine similarity between product and occasion prompts
sims = product_embeddings @ occasion_prompts.T
scores = softmax(mean(sims, axis=1))
```

**Problems:**
- No semantic reasoning about WHY item fits occasion
- Relies entirely on embedding quality
- No explicit attribute consideration
- Context-dependent (changes with candidate set)

---

### ALTERNATIVE 1: Attribute-Based Occasion Classifier

**Approach:** Train classifier on explicit attributes

```python
# online/occasion_classifier.py
import torch.nn as nn

class OccasionClassifier(nn.Module):
    """Learn occasion suitability from attributes"""
    
    def __init__(self, n_occasions=10):
        super().__init__()
        self.attribute_encoder = nn.Sequential(
            nn.Linear(512, 256),  # Image embedding
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Explicit attribute branches
        self.color_branch = nn.Linear(256, 64)
        self.category_branch = nn.Linear(256, 64)
        self.style_branch = nn.Linear(256, 64)
        self.formality_branch = nn.Linear(256, 64)
        
        # Fusion + occasion prediction
        self.fusion = nn.Linear(256, 128)
        self.occasion_head = nn.Linear(128, n_occasions)
    
    def forward(self, image_emb, color_emb, category_emb, style_emb):
        x = self.attribute_encoder(image_emb)
        
        # Attribute-specific features
        color_feat = self.color_branch(x)
        category_feat = self.category_branch(x)
        style_feat = self.style_branch(x)
        formality_feat = self.formality_branch(x)
        
        # Concatenate
        combined = torch.cat([color_feat, category_feat, style_feat, formality_feat], dim=1)
        fused = self.fusion(combined)
        
        # Occasion probabilities
        logits = self.occasion_head(fused)
        return torch.softmax(logits, dim=1)

# Training data creation:
# 1. Manual annotation: 500 items × 10 occasions = 5000 labels
# 2. Weak supervision: Use LLM to generate pseudo-labels
# 3. Active learning: Iteratively label most uncertain items
```

**Academic Justification:**
- Interpretable: Can analyze which attributes drive occasion scores
- Learnable: Improves with data
- Generalizable: Transfers to new occasions
- **Papers:** "Attribute-Aware Collaborative Filtering" (RecSys), "Explainable Fashion Recommendation" (SIGIR)

---

### ALTERNATIVE 2: Graph Neural Network for Occasion Reasoning

**Approach:** Model items, occasions, and attributes as graph

```python
# online/occasion_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class OccasionGNN(nn.Module):
    """Graph-based occasion suitability"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Graph structure:
        # Nodes: [items, occasions, attributes]
        # Edges: item-attribute, item-occasion, occasion-attribute
        
        self.conv1 = GATConv(512, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4)
        self.conv3 = GATConv(hidden_dim * 4, 1, heads=1)
    
    def forward(self, x, edge_index):
        # x: node features [num_nodes, 512]
        # edge_index: graph connectivity
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        return torch.sigmoid(x)  # Suitability scores

# Graph construction:
# - Item nodes: product embeddings
# - Occasion nodes: occasion embeddings
# - Attribute nodes: color, style, formality embeddings
# - Edges: co-occurrence, similarity, semantic relations
```

**Academic Justification:**
- Captures relational reasoning
- Propagates information through graph
- Handles multi-hop relationships (item → attribute → occasion)
- **Papers:** "Graph Convolutional Networks for Recommendation" (WWW), "Knowledge Graph Attention Network" (KDD)

---

### ALTERNATIVE 3: Contrastive Learning for Occasion Alignment

**Approach:** Learn occasion-specific embedding space

```python
# online/occasion_contrastive.py
import torch.nn as nn

class ContrastiveOccasionEncoder(nn.Module):
    """Learn occasion-specific representations"""
    
    def __init__(self):
        super().__init__()
        self.item_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.occasion_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, item_emb, occasion_emb):
        item_proj = F.normalize(self.item_encoder(item_emb), dim=1)
        occasion_proj = F.normalize(self.occasion_encoder(occasion_emb), dim=1)
        return item_proj, occasion_proj
    
    def contrastive_loss(self, item_proj, occasion_proj, labels, temperature=0.07):
        """InfoNCE loss"""
        # Positive pairs: (item, suitable_occasion)
        # Negative pairs: (item, unsuitable_occasion)
        
        logits = torch.matmul(item_proj, occasion_proj.T) / temperature
        loss = F.cross_entropy(logits, labels)
        return loss

# Training:
# 1. Positive pairs: Items worn for specific occasions (from annotations)
# 2. Negative pairs: Random or hard negatives
# 3. Optimize: max similarity for positive, min for negative
```

**Academic Justification:**
- State-of-the-art for representation learning
- Self-supervised (can use weak labels)
- Learns discriminative features
- **Papers:** "SimCLR" (ICML), "Contrastive Learning for RecSys" (RecSys)

---

### ALTERNATIVE 4: Multi-Task Learning with Auxiliary Objectives

**Approach:** Joint learning of multiple related tasks

```python
# online/occasion_multitask.py
class MultiTaskOccasionModel(nn.Module):
    """Joint learning: occasion + attributes + style"""
    
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.occasion_head = nn.Linear(256, 10)  # 10 occasions
        self.formality_head = nn.Linear(256, 5)  # 5 formality levels
        self.style_head = nn.Linear(256, 20)     # 20 style categories
        self.color_head = nn.Linear(256, 50)     # 50 colors
    
    def forward(self, x):
        shared = self.shared_encoder(x)
        
        occasion_logits = self.occasion_head(shared)
        formality_logits = self.formality_head(shared)
        style_logits = self.style_head(shared)
        color_logits = self.color_head(shared)
        
        return {
            'occasion': torch.softmax(occasion_logits, dim=1),
            'formality': torch.softmax(formality_logits, dim=1),
            'style': torch.softmax(style_logits, dim=1),
            'color': torch.softmax(color_logits, dim=1)
        }
    
    def compute_loss(self, outputs, targets):
        # Multi-task loss with uncertainty weighting
        loss_occasion = F.cross_entropy(outputs['occasion'], targets['occasion'])
        loss_formality = F.cross_entropy(outputs['formality'], targets['formality'])
        loss_style = F.cross_entropy(outputs['style'], targets['style'])
        loss_color = F.cross_entropy(outputs['color'], targets['color'])
        
        # Learnable task weights
        total_loss = (
            self.w_occasion * loss_occasion +
            self.w_formality * loss_formality +
            self.w_style * loss_style +
            self.w_color * loss_color
        )
        return total_loss
```

**Academic Justification:**
- Auxiliary tasks improve main task (occasion)
- Shared representations learn better features
- Regularization effect
- **Papers:** "Multi-Task Learning for RecSys" (RecSys), "Auxiliary Tasks for Deep Learning" (ICML)

---

### ALTERNATIVE 5: LLM-Based Reasoning (Most Novel)

**Approach:** Use LLM to reason about occasion suitability

```python
# online/occasion_llm_reasoning.py
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class LLMOccasionReasoner:
    """Use LLM for explicit reasoning"""
    
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-70b", temperature=0.1)
    
    def score_occasion_suitability(self, item, occasion, return_reasoning=True):
        """Score with chain-of-thought reasoning"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fashion expert. Analyze if an item is suitable for an occasion.
            Think step-by-step:
            1. What are the key requirements for this occasion?
            2. What attributes does this item have?
            3. How well do they match?
            4. Provide a suitability score (0-10) and explanation."""),
            ("human", """
            Item: {item_name}
            Description: {item_description}
            Color: {color}
            Category: {category}
            Price: {price}
            
            Occasion: {occasion}
            
            Analyze suitability:""")
        ])
        
        response = self.llm.invoke(prompt.format(
            item_name=item['name'],
            item_description=item['description'],
            color=item['color'],
            category=item['category'],
            price=item['price'],
            occasion=occasion
        ))
        
        # Parse response to extract score and reasoning
        score, reasoning = self._parse_response(response.content)
        
        if return_reasoning:
            return score / 10.0, reasoning
        return score / 10.0
    
    def _parse_response(self, text):
        """Extract score and reasoning from LLM output"""
        # Use regex or structured output parsing
        import re
        score_match = re.search(r'score[:\s]+(\d+)', text.lower())
        score = int(score_match.group(1)) if score_match else 5
        return score, text

# Advantages:
# - Explicit reasoning (interpretable)
# - No training data needed
# - Handles novel occasions
# - Can incorporate complex rules

# Disadvantages:
# - Slower (API calls)
# - Costs money
# - Less consistent
```

**Academic Justification:**
- Emerging paradigm: LLMs as reasoners
- Zero-shot generalization
- Explainable by design
- **Papers:** "Chain-of-Thought Prompting" (NeurIPS), "LLMs for Recommendation" (arXiv 2023)

---

## PART 2: IMPROVED OVERALL PIPELINE

### PROPOSED ARCHITECTURE: Hybrid Neural-Symbolic System

```python
# online/hybrid_pipeline.py

class HybridRecommenderPipeline:
    """Academically robust pipeline"""
    
    def __init__(self):
        # Stage 1: Multi-modal retrieval
        self.retriever = EnsembleRetriever([
            CLIPRetriever(model="ViT-L/14"),
            FashionCLIPRetriever(),
            BERTRetriever()
        ])
        
        # Stage 2: Attribute extraction
        self.attribute_extractor = MultiTaskAttributeExtractor()
        
        # Stage 3: Occasion scoring (choose one)
        self.occasion_scorer = OccasionGNN()  # or Classifier, Contrastive, etc.
        
        # Stage 4: Learning-to-rank
        self.ranker = LightGBMRanker()
        
        # Stage 5: Diversification
        self.diversifier = MMRDiversifier()
        
        # Stage 6: Explanation generation
        self.explainer = LLMExplainer()
    
    def recommend(self, query, conversation_history, k=10):
        # 1. Query understanding
        parsed = self.parse_query_with_context(query, conversation_history)
        
        # 2. Multi-modal retrieval (top-200)
        candidates = self.retriever.retrieve(query, parsed, k=200)
        
        # 3. Extract attributes
        for c in candidates:
            c.attributes = self.attribute_extractor.extract(c)
        
        # 4. Occasion scoring
        if parsed.occasion:
            occasion_scores = self.occasion_scorer.score(candidates, parsed.occasion)
        else:
            occasion_scores = {c.id: 0.5 for c in candidates}
        
        # 5. Feature extraction for ranking
        features = self.extract_ranking_features(candidates, parsed, occasion_scores)
        
        # 6. Learning-to-rank
        ranked = self.ranker.rank(candidates, features)
        
        # 7. Diversification
        diverse = self.diversifier.diversify(ranked[:50], k=k)
        
        # 8. Generate explanations
        for item in diverse:
            item.explanation = self.explainer.explain(item, parsed, occasion_scores[item.id])
        
        return diverse
```

---

## PART 3: EVALUATION FRAMEWORK

### Comprehensive Evaluation Suite

```python
# evaluation/comprehensive_eval.py

class ComprehensiveEvaluator:
    """Multi-faceted evaluation"""
    
    def evaluate_all(self):
        results = {}
        
        # 1. Retrieval Quality
        results['retrieval'] = {
            'P@10': self.eval_precision_at_k(10),
            'NDCG@10': self.eval_ndcg(10),
            'MRR': self.eval_mrr(),
            'Recall@50': self.eval_recall_at_k(50)
        }
        
        # 2. Occasion Accuracy
        results['occasion'] = {
            'accuracy': self.eval_occasion_accuracy(),
            'f1_score': self.eval_occasion_f1(),
            'confusion_matrix': self.eval_occasion_confusion()
        }
        
        # 3. Attribute Consistency
        results['attributes'] = {
            'color_match': self.eval_color_consistency(),
            'category_match': self.eval_category_consistency(),
            'price_compliance': self.eval_price_compliance()
        }
        
        # 4. Diversity
        results['diversity'] = {
            'intra_list_diversity': self.eval_diversity(),
            'coverage': self.eval_catalog_coverage(),
            'novelty': self.eval_novelty()
        }
        
        # 5. Explanation Quality
        results['explanations'] = {
            'groundedness': self.eval_explanation_groundedness(),
            'relevance': self.eval_explanation_relevance(),
            'fluency': self.eval_explanation_fluency()
        }
        
        # 6. Conversational Quality
        results['conversation'] = {
            'context_awareness': self.eval_context_awareness(),
            'coherence': self.eval_conversation_coherence(),
            'success_rate': self.eval_task_success()
        }
        
        # 7. Robustness
        results['robustness'] = {
            'adversarial': self.eval_adversarial_robustness(),
            'edge_cases': self.eval_edge_cases(),
            'cross_occasion': self.eval_cross_occasion_transfer()
        }
        
        return results
```

---

## PART 4: DATA COLLECTION STRATEGIES

### Without User Feedback

```python
# data/synthetic_data_generation.py

class SyntheticDataGenerator:
    """Generate training data without user feedback"""
    
    def generate_occasion_labels(self):
        """Method 1: LLM-based weak supervision"""
        for item in self.catalog:
            prompt = f"Is this item suitable for {occasion}? {item.description}"
            label = self.llm.predict(prompt)  # 0 or 1
            self.labels.append((item.id, occasion, label))
    
    def generate_from_rules(self):
        """Method 2: Rule-based heuristics"""
        rules = {
            'wedding_guest': lambda item: (
                item.category in ['DRESSES', 'SKIRTS'] and
                item.formality > 0.7 and
                item.color not in ['white', 'cream']
            ),
            'casual_day': lambda item: (
                item.category in ['JEANS', 'T-SHIRTS'] and
                item.formality < 0.4
            )
        }
        
        for item in self.catalog:
            for occasion, rule in rules.items():
                if rule(item):
                    self.labels.append((item.id, occasion, 1))
    
    def generate_from_clustering(self):
        """Method 3: Cluster similar items"""
        # Items in same cluster likely suitable for same occasions
        clusters = self.cluster_items(self.embeddings)
        
        # Manually label a few items per cluster
        # Propagate labels to cluster members
        for cluster_id, items in clusters.items():
            seed_labels = self.manually_label(items[:5])
            for item in items[5:]:
                item.occasion_labels = seed_labels
```

---

## PART 5: ACADEMIC CONTRIBUTIONS

### Novel Aspects for Publication

1. **Hybrid Neural-Symbolic Reasoning**
   - Combine embeddings + explicit attributes + LLM reasoning
   - Paper angle: "Bridging Neural and Symbolic Approaches for Fashion RecSys"

2. **Conversational Occasion Modeling**
   - Multi-turn dialogue for occasion refinement
   - Paper angle: "Conversational Occasion-Aware Recommendation"

3. **Multimodal Contrastive Learning**
   - Joint image-text-occasion embedding space
   - Paper angle: "Contrastive Learning for Occasion-Aware Fashion Retrieval"

4. **Explainable Occasion Scoring**
   - Attribute-based interpretability
   - Paper angle: "Explainable Occasion Suitability in Fashion RecSys"

5. **Zero-Shot Occasion Transfer**
   - Generalize to new occasions without retraining
   - Paper angle: "Zero-Shot Occasion Recognition via LLM Reasoning"

---

## RECOMMENDED APPROACH (Most Academically Strong)

**Hybrid System:**
1. **Retrieval:** Ensemble (CLIP + FashionCLIP + BERT)
2. **Occasion Scoring:** Multi-task GNN + LLM reasoning
3. **Ranking:** LightGBM LTR with 15+ features
4. **Evaluation:** Comprehensive suite (8 dimensions)
5. **Data:** Weak supervision + active learning

**Why This Works:**
- ✅ Novel (hybrid neural-symbolic)
- ✅ Robust (ensemble + multi-task)
- ✅ Explainable (LLM reasoning)
- ✅ Evaluable (comprehensive metrics)
- ✅ Practical (works without user feedback)
- ✅ Publishable (multiple contribution angles)

**Target Venues:**
- RecSys (main conference)
- SIGIR (information retrieval)
- WWW (web applications)
- FashionXRecsys (workshop)
