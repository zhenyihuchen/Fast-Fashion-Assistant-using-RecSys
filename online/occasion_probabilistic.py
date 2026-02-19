"""
Probabilistic Occasion Scoring with Uncertainty Quantification
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.stats import beta, norm


BASE_DIR = Path(__file__).resolve().parent.parent
OCCASION_EMBEDDINGS_PATH = BASE_DIR / "occasion_library" / "occasion_clavix_embeddings.npz"


# ============================================================
# 1. BAYESIAN OCCASION SCORING
# ============================================================

class BayesianOccasionScorer:
    """Model occasion suitability as Beta distribution"""
    
    def __init__(self, prior_alpha=2.0, prior_beta=2.0):
        # Prior: Beta(2, 2) = slightly peaked at 0.5
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
    
    def compute_scores_with_uncertainty(
        self,
        candidates: Iterable[tuple[str, float]],
        parsed: dict,
        product_ids: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[str, dict]:
        """Return mean score + confidence interval"""
        
        # Get cosine similarities (evidence)
        sims = self._compute_similarities(candidates, parsed, product_ids, embeddings)
        
        results = {}
        for item_id, sim_scores in sims.items():
            # Update Beta distribution with evidence
            # sim_scores: array of similarities to occasion prompts
            
            # Convert similarities [-1, 1] to [0, 1]
            normalized_sims = (sim_scores + 1) / 2
            
            # Treat each similarity as Bernoulli trial
            # High similarity = success, low = failure
            successes = np.sum(normalized_sims > 0.5)
            failures = len(normalized_sims) - successes
            
            # Posterior: Beta(alpha + successes, beta + failures)
            posterior_alpha = self.prior_alpha + successes
            posterior_beta = self.prior_beta + failures
            
            # Compute statistics
            mean_score = posterior_alpha / (posterior_alpha + posterior_beta)
            variance = (posterior_alpha * posterior_beta) / (
                (posterior_alpha + posterior_beta) ** 2 * 
                (posterior_alpha + posterior_beta + 1)
            )
            
            # 95% credible interval
            ci_lower = beta.ppf(0.025, posterior_alpha, posterior_beta)
            ci_upper = beta.ppf(0.975, posterior_alpha, posterior_beta)
            
            results[item_id] = {
                'mean': float(mean_score),
                'variance': float(variance),
                'std': float(np.sqrt(variance)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'confidence': float(ci_upper - ci_lower),  # Narrower = more confident
            }
        
        return results
    
    def _compute_similarities(self, candidates, parsed, product_ids, embeddings):
        """Helper to compute similarities"""
        # Similar to original implementation
        target = self._get_occasion_target(parsed)
        if not target:
            return {}
        
        occasion_npz = np.load(OCCASION_EMBEDDINGS_PATH)
        if target not in occasion_npz:
            return {}
        
        prompt_embeddings = occasion_npz[target].astype("float32", copy=False)
        prompt_embeddings = self._normalize_rows(prompt_embeddings)
        
        id_to_index = {str(pid): idx for idx, pid in enumerate(product_ids)}
        cand_ids = [str(row_id) for row_id, _ in candidates if str(row_id) in id_to_index]
        
        cand_indices = [id_to_index[row_id] for row_id in cand_ids]
        cand_embeddings = embeddings[cand_indices].astype("float32", copy=False)
        cand_embeddings = self._normalize_rows(cand_embeddings)
        
        # Similarities: [num_candidates, num_prompts]
        sims = cand_embeddings @ prompt_embeddings.T
        
        return {row_id: sims[i] for i, row_id in enumerate(cand_ids)}
    
    @staticmethod
    def _normalize_rows(arr):
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms
    
    @staticmethod
    def _get_occasion_target(parsed):
        if not parsed:
            return None
        occasion = parsed.get("occasion")
        if not occasion:
            return None
        mode = occasion.get("mode") if isinstance(occasion, dict) else getattr(occasion, "mode")
        target = occasion.get("target") if isinstance(occasion, dict) else getattr(occasion, "target", None)
        return target if mode == "on" else None


# ============================================================
# 2. GAUSSIAN PROCESS OCCASION SCORING
# ============================================================

class GPOccasionScorer:
    """Model occasion suitability with Gaussian Process"""
    
    def __init__(self, noise_variance=0.01):
        self.noise_variance = noise_variance
    
    def compute_scores_with_uncertainty(
        self,
        candidates: Iterable[tuple[str, float]],
        parsed: dict,
        product_ids: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[str, dict]:
        """GP-based scoring with predictive uncertainty"""
        
        sims = self._compute_similarities(candidates, parsed, product_ids, embeddings)
        
        results = {}
        for item_id, sim_scores in sims.items():
            # Mean: average similarity
            mean_score = float(np.mean(sim_scores))
            
            # Variance: combination of data variance + noise
            data_variance = float(np.var(sim_scores))
            predictive_variance = data_variance + self.noise_variance
            predictive_std = float(np.sqrt(predictive_variance))
            
            # 95% confidence interval (Gaussian assumption)
            ci_lower = mean_score - 1.96 * predictive_std
            ci_upper = mean_score + 1.96 * predictive_std
            
            results[item_id] = {
                'mean': mean_score,
                'variance': predictive_variance,
                'std': predictive_std,
                'ci_lower': max(0.0, ci_lower),  # Clip to [0, 1]
                'ci_upper': min(1.0, ci_upper),
                'confidence': float(1.0 / (1.0 + predictive_std))  # Higher std = lower confidence
            }
        
        return results


# ============================================================
# 3. ENSEMBLE PROBABILISTIC SCORING
# ============================================================

class EnsembleOccasionScorer:
    """Combine multiple occasion embeddings probabilistically"""
    
    def compute_scores_with_uncertainty(
        self,
        candidates: Iterable[tuple[str, float]],
        parsed: dict,
        product_ids: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[str, dict]:
        """Treat each prompt embedding as independent estimator"""
        
        sims = self._compute_similarities(candidates, parsed, product_ids, embeddings)
        
        results = {}
        for item_id, sim_scores in sims.items():
            # Each prompt gives independent estimate
            # Combine using ensemble statistics
            
            mean_score = float(np.mean(sim_scores))
            std_score = float(np.std(sim_scores))
            
            # Bootstrap confidence interval
            n_bootstrap = 1000
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(sim_scores, size=len(sim_scores), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            ci_lower = float(np.percentile(bootstrap_means, 2.5))
            ci_upper = float(np.percentile(bootstrap_means, 97.5))
            
            results[item_id] = {
                'mean': mean_score,
                'std': std_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'confidence': float(1.0 - (ci_upper - ci_lower)),  # Narrower = more confident
                'n_prompts': len(sim_scores)
            }
        
        return results


# ============================================================
# 4. CALIBRATED PROBABILISTIC SCORING
# ============================================================

class CalibratedOccasionScorer:
    """Calibrate scores to true probabilities using Platt scaling"""
    
    def __init__(self):
        # These would be learned from validation data
        self.platt_a = 1.0  # Slope
        self.platt_b = 0.0  # Intercept
    
    def calibrate_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to calibrated probabilities"""
        # Platt scaling: P(y=1|s) = 1 / (1 + exp(A*s + B))
        logits = self.platt_a * raw_scores + self.platt_b
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        return calibrated
    
    def compute_calibrated_scores(
        self,
        candidates: Iterable[tuple[str, float]],
        parsed: dict,
        product_ids: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict[str, float]:
        """Return calibrated probability scores"""
        
        sims = self._compute_similarities(candidates, parsed, product_ids, embeddings)
        
        results = {}
        for item_id, sim_scores in sims.items():
            raw_score = np.mean(sim_scores)
            calibrated_prob = float(self.calibrate_scores(np.array([raw_score]))[0])
            results[item_id] = calibrated_prob
        
        return results


# ============================================================
# 5. THOMPSON SAMPLING FOR EXPLORATION
# ============================================================

class ThompsonSamplingOccasionScorer:
    """Sample from posterior for exploration-exploitation"""
    
    def __init__(self):
        self.bayesian_scorer = BayesianOccasionScorer()
    
    def sample_scores(
        self,
        candidates: Iterable[tuple[str, float]],
        parsed: dict,
        product_ids: np.ndarray,
        embeddings: np.ndarray,
        n_samples: int = 1
    ) -> dict[str, list[float]]:
        """Sample occasion scores from posterior"""
        
        posteriors = self.bayesian_scorer.compute_scores_with_uncertainty(
            candidates, parsed, product_ids, embeddings
        )
        
        results = {}
        for item_id, posterior in posteriors.items():
            # Sample from Beta distribution
            alpha = posterior['mean'] * 100  # Scale for Beta
            beta_param = (1 - posterior['mean']) * 100
            
            samples = beta.rvs(alpha, beta_param, size=n_samples)
            results[item_id] = samples.tolist()
        
        return results


# ============================================================
# USAGE COMPARISON
# ============================================================

if __name__ == "__main__":
    # Example usage
    
    # Current (quasi-probabilistic)
    from online.occasion_suitability_scores import compute_occasion_scores
    scores_current = compute_occasion_scores(candidates, parsed, product_ids, embeddings)
    # Returns: {'item1': 0.35, 'item2': 0.42, ...}
    
    # Bayesian (fully probabilistic)
    bayesian_scorer = BayesianOccasionScorer()
    scores_bayesian = bayesian_scorer.compute_scores_with_uncertainty(
        candidates, parsed, product_ids, embeddings
    )
    # Returns: {
    #   'item1': {'mean': 0.35, 'std': 0.08, 'ci_lower': 0.20, 'ci_upper': 0.50},
    #   'item2': {'mean': 0.42, 'std': 0.05, 'ci_lower': 0.32, 'ci_upper': 0.52},
    # }
    
    # Use uncertainty for ranking
    # Option 1: Rank by mean (exploitation)
    # Option 2: Rank by upper confidence bound (exploration)
    # Option 3: Thompson sampling (balanced)
