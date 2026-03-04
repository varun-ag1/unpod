"""
Eval Accuracy Scorer
Measures how well generated QA pairs are grounded in source material
using OpenAI embedding cosine similarity.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import openai


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row vectors of a and b. Returns (n_a, n_b) matrix."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T
@dataclass
class PairMatchScore:
    question: str
    answer: str
    eval_type: str
    source_type: str  # "agent" | "kb"
    match_score: float  # 0.0 – 100.0
@dataclass
class AccuracyResult:
    source_type: str  # "agent" | "kb"
    total_pairs: int
    avg_match_score: float  # 0.0 – 100.0
    matched_pairs: int  # pairs with score >= threshold
    accuracy_pct: float  # matched_pairs / total * 100
    per_type_scores: Dict[str, float]  # eval_type → avg match score
    pair_scores: List[PairMatchScore]
    threshold: float = field(default=60.0)
def _chunk_source(text: str, chunk_size: int = 400) -> List[str]:
    """Split source text into word-based chunks for granular embedding match."""
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
        if words[i : i + chunk_size]
    ]
    return chunks or [text]
class EvalAccuracyScorer:
    """
    Scores generated QA pairs against source material using
    OpenAI embedding cosine similarity.
    For each QA pair, finds the best-matching source chunk and returns
    the cosine similarity score (0–100).
    Match score interpretation:
      ≥ 75  High match   — answer is well-grounded in source
      60–74 Medium match — partial semantic overlap
      < 60  Low match    — answer may be hallucinated or off-topic
    """
    EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_THRESHOLD: float = 60.0
    def __init__(self, api_key: str) -> None:
        self._client = openai.OpenAI(api_key=api_key)
    def score_agent_pairs(
        self,
        qa_pairs: List[Dict],
        agent_config: Dict,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> AccuracyResult:
        source_text = " ".join(
            filter(
                None,
                [
                    agent_config.get("system_prompt", ""),
                    agent_config.get("persona", ""),
                    agent_config.get("first_message", ""),
                ],
            )
        )
        return self._score_against_source(qa_pairs, source_text, "agent", threshold)
    def score_kb_pairs(
        self,
        qa_pairs: List[Dict],
        documents: List[Dict],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> AccuracyResult:
        source_text = " ".join(
            doc.get("content", "") for doc in documents if doc.get("content", "").strip()
        )
        return self._score_against_source(qa_pairs, source_text, "kb", threshold)
    def score_kb_pairs_from_text(
        self,
        qa_pairs: List[Dict],
        source_text: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> AccuracyResult:
        return self._score_against_source(qa_pairs, source_text, "kb", threshold)
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _score_against_source(
        self,
        qa_pairs: List[Dict],
        source_text: str,
        source_type: str,
        threshold: float,
    ) -> AccuracyResult:
        qa_texts = [
            f"{p.get('question', '')} {p.get('answer', '')}".strip()
            for p in qa_pairs
        ]
        raw_scores = self._embedding_scores(source_text, qa_texts)
        pair_scores: List[PairMatchScore] = []
        per_type_totals: Dict[str, List[float]] = {}
        for pair, score in zip(qa_pairs, raw_scores):
            eval_type = pair.get("eval_type", "unknown")
            pair_scores.append(
                PairMatchScore(
                    question=pair.get("question", ""),
                    answer=pair.get("answer", ""),
                    eval_type=eval_type,
                    source_type=source_type,
                    match_score=round(score, 2),
                )
            )
            per_type_totals.setdefault(eval_type, []).append(score)
        per_type_scores = {
            t: round(float(np.mean(scores)), 2)
            for t, scores in per_type_totals.items()
        }
        avg_match = round(float(np.mean(raw_scores)) if raw_scores else 0.0, 2)
        matched = sum(1 for s in raw_scores if s >= threshold)
        accuracy_pct = round((matched / len(raw_scores)) * 100, 2) if raw_scores else 0.0
        return AccuracyResult(
            source_type=source_type,
            total_pairs=len(qa_pairs),
            avg_match_score=avg_match,
            matched_pairs=matched,
            accuracy_pct=accuracy_pct,
            per_type_scores=per_type_scores,
            pair_scores=pair_scores,
            threshold=threshold,
        )
    def _embedding_scores(
        self, source_text: str, qa_texts: List[str]
    ) -> List[float]:
        """
        Embed source chunks and QA texts, return best-chunk cosine similarity
        per QA pair (0–100).
        """
        if not source_text.strip() or not qa_texts:
            return [0.0] * len(qa_texts)
        chunks = _chunk_source(source_text)
        chunk_resp = self._client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=chunks
        )
        chunk_embs = np.array([e.embedding for e in chunk_resp.data])
        qa_resp = self._client.embeddings.create(
            model=self.EMBEDDING_MODEL, input=qa_texts
        )
        qa_embs = np.array([e.embedding for e in qa_resp.data])
        # Best-matching chunk score per QA pair
        sim_matrix = cosine_similarity(qa_embs, chunk_embs)  # (n_qa, n_chunks)
        best_scores = sim_matrix.max(axis=1)
        return [round(float(s) * 100, 2) for s in best_scores]
