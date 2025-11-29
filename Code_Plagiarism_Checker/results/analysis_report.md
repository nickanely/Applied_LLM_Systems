## Comprehensive Performance Analysis

### 1. Embedding Method
**Metrics:** Accuracy: 50.8% | Precision: 0.0% | Recall: 0.0% | F1: 0.0% | Speed: 0.71s

**Analysis:**
- Completely broken - predicts everything as "not plagiarized"
- 31 true negatives, 0 true positives, 30 false negatives
- Fastest method but entirely useless
- Zero ability to detect actual plagiarism

**Conclusion:** Failed baseline. The embedding similarity threshold or classification logic is misconfigured. Do not use.

---

### 2. Direct LLM
**Metrics:** Accuracy: 50.8% | Precision: 50.0% | Recall: 10.0% | F1: 16.7% | Speed: 9.33s

**Analysis:**
- Barely better than random guessing (coin flip = 50%)
- Extremely conservative - catches only 3 out of 30 plagiarized cases
- 28 true negatives, 3 true positives, 27 false negatives, 3 false positives
- High false negative rate (90%) means it misses almost all plagiarism
- 50% precision on tiny sample (3 detections) is meaningless

**Conclusion:** Unreliable. Without retrieval context, the LLM cannot effectively identify plagiarism. The 9.3s latency doesn't justify the poor performance.

---

### 3. RAG (Retrieval-Augmented Generation)
**Metrics:** Accuracy: 77.0% | Precision: 76.7% | Recall: 76.7% | F1: 76.7% | Speed: 15.16s

**Analysis:**
- Strong balanced performance - precision equals recall
- 24 true positives, 23 true negatives, 7 false positives, 7 false negatives
- Correctly identifies 24/31 plagiarized cases (77.4% detection rate)
- Errors evenly distributed between false accusations and missed plagiarism
- Retrieval provides crucial context that Direct LLM lacks

**Strengths:**
- First method that actually works
- Balanced error profile suitable for most use cases
- Significant jump from 50.8% to 77.0% accuracy

**Weaknesses:**
- 7 false positives could damage trust if innocent work is flagged
- Slowest method at 15.2s per query
- Room for improvement in both precision and recall

**Conclusion:** Solid performer. Good baseline for RAG systems but can be improved.

---

### 4. Hybrid RAG
**Metrics:** Accuracy: 80.3% | Precision: 84.6% | Recall: 73.3% | F1: 78.6% | Speed: 13.77s

**Analysis:**
- Best overall performance across all metrics
- 22 true positives, 27 true negatives, 4 false positives, 8 false negatives
- **Key improvement: Reduced false positives from 7 to 4 (43% reduction)**
- Increased true negatives from 23 to 27 (better at confirming clean work)
- Slight increase in false negatives (7 → 8) but acceptable trade-off

**Trade-off Profile:**
- Prioritizes precision (84.6%) over recall (73.3%)
- Fewer false accusations (4 vs 7) at cost of missing one more plagiarism case
- For plagiarism detection, this trade-off is correct - false accusations are more damaging

**Performance vs RAG:**
- +3.3% accuracy improvement
- +7.9% precision boost (critical metric)
- -3.4% recall drop (acceptable)
- 1.4s faster (9% speed improvement)

**Conclusion:** **Best model for production.** The "hybrid" approach (likely combining multiple retrieval strategies or adding reranking) successfully improves precision while maintaining strong recall. Faster than plain RAG with better accuracy.

---

## Overall Rankings

### By Metric
1. **Accuracy:** Hybrid RAG (80.3%) > RAG (77.0%) > Direct LLM = Embedding (50.8%)
2. **Precision:** Hybrid RAG (84.6%) > RAG (76.7%) > Direct LLM (50.0%) > Embedding (0.0%)
3. **Recall:** RAG (76.7%) > Hybrid RAG (73.3%) > Direct LLM (10.0%) > Embedding (0.0%)
4. **F1 Score:** Hybrid RAG (78.6%) > RAG (76.7%) > Direct LLM (16.7%) > Embedding (0.0%)
5. **Speed:** Embedding (0.71s) > Direct LLM (9.33s) > Hybrid RAG (13.77s) > RAG (15.16s)

### By Use Case
- **Production deployment:** Hybrid RAG (best balance of accuracy, precision, speed)
- **High-stakes scenarios:** Hybrid RAG (84.6% precision minimizes false accusations)
- **Maximum detection:** RAG (76.7% recall catches slightly more cases)
- **Speed-critical:** None - all useful methods are 10-15s (embedding is fast but broken)

---

## Key Findings

### What Works
1. **Retrieval is essential** - RAG methods (77-80%) vastly outperform non-retrieval approaches (50%)
2. **Hybrid strategies improve precision** - combining techniques reduces false positives
3. **LLMs alone fail** - without retrieved context, even powerful models perform poorly

### What Doesn't Work
1. **Pure embedding similarity** - too simplistic for nuanced plagiarism detection
2. **Direct LLM inference** - lacks the evidence needed for accurate judgments

### Critical Insights
1. **Precision matters most** - in plagiarism detection, false accusations damage credibility
2. **Speed is acceptable** - 13-15s latency is reasonable for a plagiarism checker
3. **Balanced metrics deceive** - RAG's equal precision/recall looks good but Hybrid RAG's precision bias is actually preferable
4. **Error analysis reveals quality** - Hybrid RAG's confusion matrix (4 FP vs 7 FP) shows meaningful improvement

---

