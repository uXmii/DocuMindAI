"""
RAG Evaluation Metrics - PRODUCTION v2
=======================================
What's new vs v1:
  1. LLM-as-judge faithfulness  — no ground truth needed
     The LLM reads the retrieved chunks and the answer, then scores:
       • Is the answer grounded in the context?  (0–1)
       • Are there hallucinated claims?           (0–1)
  2. Consistency testing
     Ask the same question N times, measure variance in answers and scores.
     Reports a "consistency score" (1 - normalised std-dev).
  3. Coverage report
     Over a batch of questions: what % got a "good" answer (overall_score ≥ 0.6)?
     What % fell back to extractive / no-answer?
  4. All previous metrics retained (semantic similarity, context precision/recall,
     response time, winner selection).
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Optional LLM clients ───────────────────────────────────────────────────────
try:
    import google.generativeai as _genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import anthropic as _anthropic_lib
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai as _openai_lib
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Thresholds ─────────────────────────────────────────────────────────────────
GOOD_ANSWER_THRESHOLD   = 0.6   # overall_score ≥ this → "good answer"
EXTRACTIVE_MARKERS      = [     # signals the system fell back to extractive
    "based on",
    "page",
    "source:",
    "**source**",
]
CONSISTENCY_RUNS        = 3     # how many times to repeat a question for consistency test
CONSISTENCY_THRESHOLD   = 0.85  # consistency_score ≥ this → "consistent"

def _init_gemini_model(lib=None):
    """
    Find a working Gemini model WITHOUT burning quota.
    Uses list_models() which is free, then picks the best available flash model.
    Falls back to trying known names without probing if list_models fails.
    """
    import google.generativeai as _g
    if lib is None:
        lib = _g

    PREFERRED = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-001",
        "gemini-1.0-pro",
        "gemini-pro",
    ]

    # Try list_models() first — zero quota cost
    try:
        available = []
        for m in lib.list_models():
            if "generateContent" in m.supported_generation_methods:
                available.append(m.name.replace("models/", ""))
        # Pick best preferred model that's available
        for name in PREFERRED:
            if name in available:
                model = lib.GenerativeModel(name)
                print(f"   ✔  Gemini model selected: {name}")
                return model
        # If none of our preferred list matched, use first available
        if available:
            name = available[0]
            model = lib.GenerativeModel(name)
            print(f"   ✔  Gemini model selected (first available): {name}")
            return model
    except Exception as e:
        print(f"   ⚠  list_models() failed ({e}), trying known names without probe...")

    # Fallback: try names without live probe (no quota used)
    for name in PREFERRED:
        try:
            model = lib.GenerativeModel(name)
            print(f"   ✔  Gemini model assumed: {name} (unverified)")
            return model
        except Exception:
            continue

    print("   ✗  No Gemini model could be initialised")
    return None


CONSISTENCY_THRESHOLD = 0.85  # consistency_score >= this -> "consistent"


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge
# ─────────────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Uses a cheap/fast LLM to evaluate answer quality without ground truth.
    Falls back gracefully if no API key is set.
    """

    FAITHFULNESS_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation systems.

Given:
  QUESTION: {question}
  RETRIEVED CONTEXT: {context}
  ANSWER: {answer}

Score the answer on TWO dimensions (each 0.0 – 1.0):

1. FAITHFULNESS (is the answer grounded in the context, not hallucinated?)
   1.0 = every claim in the answer is directly supported by the context
   0.5 = most claims supported, minor extrapolation
   0.0 = answer contains claims not in the context / contradicts context

2. COMPLETENESS (does the answer address what the question asks?)
   1.0 = fully addresses the question using available context
   0.5 = partially addresses it
   0.0 = does not address the question

Respond ONLY in this exact JSON format (no other text):
{{"faithfulness": <float>, "completeness": <float>, "reasoning": "<one sentence>"}}"""

    FACTUAL_CHECK_PROMPT = """You are a factual accuracy checker. A user asked a question and a RAG system produced an answer from a document.

Your job: verify whether the answer is factually correct based on YOUR general knowledge, independent of the document.

QUESTION: {question}
RAG ANSWER: {answer}

Evaluate:
1. FACTUAL_ACCURACY (0.0-1.0): Is the answer factually correct based on general knowledge?
   1.0 = completely correct
   0.5 = partially correct or cannot be fully verified
   0.0 = factually wrong or contradicts known facts

2. VERIFIABLE (true/false): Can this claim be verified against general knowledge?
   true  = the answer contains specific facts that can be cross-checked
   false = the answer is too vague, opinion-based, or document-specific to verify externally

3. ISSUES: List any specific factual errors found (empty list if none)

4. EXTERNAL_CONTEXT: In 1 sentence, what does general knowledge say about this topic?

Respond ONLY in this exact JSON format:
{{"factual_accuracy": <float>, "verifiable": <bool>, "issues": [<string>], "external_context": "<string>", "verdict": "<correct|partially_correct|incorrect|unverifiable>"}}"""

    def __init__(self):
        self.provider = None
        self.client   = None

        gemini_key    = os.getenv("GEMINI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key    = os.getenv("OPENAI_API_KEY")

        if gemini_key and GEMINI_AVAILABLE:
            _genai.configure(api_key=gemini_key)
            _m = _init_gemini_model(_genai)
            if _m:
                self.client   = _m
                self.provider = "gemini"
                print(f"⚖️  LLM Judge: {_m.model_name}")
            else:
                print("⚠   Gemini init failed — judge disabled")
        elif anthropic_key and ANTHROPIC_AVAILABLE:
            self.client   = _anthropic_lib.Anthropic(api_key=anthropic_key)
            self.provider = "anthropic"
            print("⚖️  LLM Judge: Anthropic claude-haiku-4-5-20251001")
        elif openai_key and OPENAI_AVAILABLE:
            self.client   = _openai_lib.OpenAI(api_key=openai_key)
            self.provider = "openai"
            print("⚖️  LLM Judge: OpenAI GPT-4o-mini")
        else:
            print("⚠   LLM Judge: no API key — LLM faithfulness scoring disabled")

    @property
    def available(self) -> bool:
        return self.provider is not None

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to whichever LLM provider is configured, with retry on 429."""
        import time as _time
        if self.provider == "gemini":
            for attempt in range(3):
                try:
                    response = self.client.generate_content(prompt)
                    return response.text
                except Exception as e:
                    err = str(e)
                    if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                        # Extract retry delay from error if present, else use backoff
                        import re
                        delay_match = re.search(r"retry in ([0-9.]+)s", err)
                        wait = float(delay_match.group(1)) if delay_match else (10 * (attempt + 1))
                        wait = min(wait, 35)  # cap at 35s
                        print(f"   ⏳ Gemini rate limited — waiting {wait:.0f}s (attempt {attempt+1}/3)")
                        _time.sleep(wait)
                        continue
                    if "not found" in err.lower() or "404" in err:
                        new_model = _init_gemini_model(_genai)
                        if new_model:
                            self.client = new_model
                            return self.client.generate_content(prompt).text
                    raise
            raise RuntimeError("Gemini rate limit — all 3 retry attempts exhausted")
        elif self.provider == "anthropic":
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        elif self.provider == "openai":
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content
        raise RuntimeError(f"Unknown provider: {self.provider}")

    def judge(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """Evaluate faithfulness and completeness against retrieved context."""
        if not self.available:
            return {"faithfulness": None, "completeness": None,
                    "reasoning": "LLM judge not available", "judge_used": "none"}
        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question[:500], context=context[:2000], answer=answer[:1000])
        try:
            raw    = self._call_llm(prompt).strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            return {
                "faithfulness": round(float(parsed.get("faithfulness", 0)), 3),
                "completeness": round(float(parsed.get("completeness", 0)), 3),
                "reasoning":    parsed.get("reasoning", ""),
                "judge_used":   self.provider,
            }
        except Exception as e:
            return {"faithfulness": None, "completeness": None,
                    "reasoning": f"Judge error: {e}", "judge_used": self.provider}

    def check_factual_accuracy(self, question: str, answer: str) -> Dict[str, Any]:
        """External knowledge check — independent of the document."""
        if not self.available:
            return {"factual_accuracy": None, "verifiable": False, "issues": [],
                    "external_context": "LLM judge not available",
                    "verdict": "unverifiable", "judge_used": "none"}
        prompt = self.FACTUAL_CHECK_PROMPT.format(
            question=question[:500], answer=answer[:1000])
        try:
            raw    = self._call_llm(prompt).strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            return {
                "factual_accuracy": round(float(parsed.get("factual_accuracy", 0)), 3),
                "verifiable":       bool(parsed.get("verifiable", False)),
                "issues":           parsed.get("issues", []),
                "external_context": parsed.get("external_context", ""),
                "verdict":          parsed.get("verdict", "unverifiable"),
                "judge_used":       self.provider,
            }
        except Exception as e:
            return {"factual_accuracy": None, "verifiable": False,
                    "issues": [f"Check error: {e}"], "external_context": "",
                    "verdict": "unverifiable", "judge_used": self.provider}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Comprehensive RAG evaluator.

    New in v2:
      • LLM-as-judge faithfulness / completeness scores
      • Consistency testing (repeated queries)
      • Coverage report (good / extractive / fail breakdown)
    """

    def __init__(self, rag_engine=None, model_name: str = "all-MiniLM-L6-v2"):
        self.rag_engine      = rag_engine
        self.embedding_model = SentenceTransformer(model_name)
        self.llm_judge       = LLMJudge()

        self.evaluation_history  = []
        self.results_history     = []
        self.evaluation_sessions = []

        # Caches for deterministic results
        self.embedding_cache  = {}
        self.similarity_cache = {}

        np.random.seed(42)

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def _get_embedding(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self.embedding_cache:
            self.embedding_cache[key] = self.embedding_model.encode(
                [text], show_progress_bar=False
            )[0]
        return self.embedding_cache[key]

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        texts_sorted = tuple(sorted([text1, text2]))
        key = hashlib.md5(f"{texts_sorted[0]}||{texts_sorted[1]}".encode()).hexdigest()
        if key not in self.similarity_cache:
            e1 = self._get_embedding(text1)
            e2 = self._get_embedding(text2)
            sim = cosine_similarity([e1], [e2])[0][0]
            self.similarity_cache[key] = round(float(sim), 3)
        return self.similarity_cache[key]

    # ── Core metric methods ────────────────────────────────────────────────────

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        return self.calculate_semantic_similarity(question, answer)

    def evaluate_context_relevance(self, question: str, contexts: List[str]) -> Dict:
        if not contexts:
            return {"precision": 0.0, "recall": 0.0, "mean_relevance": 0.0, "individual_scores": []}
        scores = [self.calculate_semantic_similarity(question, c) for c in contexts]
        relevant = sum(1 for s in scores if s > 0.3)
        return {
            "precision":          round(relevant / len(contexts), 3),
            "recall":             round(min(relevant / 3.0, 1.0), 3),
            "mean_relevance":     round(float(np.mean(scores)), 3),
            "individual_scores":  [round(s, 3) for s in scores],
        }

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Embedding-based faithfulness (used when LLM judge is unavailable)."""
        if not answer or not contexts:
            return 0.0
        combined = " ".join(contexts)
        return self.calculate_semantic_similarity(answer, combined)

    def _is_extractive_fallback(self, answer: str) -> bool:
        """Detect if the answer is an extractive/template fallback."""
        a = answer.lower()
        return any(marker in a for marker in EXTRACTIVE_MARKERS)

    # ── Single query evaluation ────────────────────────────────────────────────

    def evaluate_query(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        run_llm_judge: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate one question across all three search modes.
        Returns the full result dict including LLM judge scores if available.
        """
        if not self.rag_engine:
            return {"error": "RAG engine not initialised"}

        results = {
            "timestamp":    datetime.now().isoformat(),
            "question":     question,
            "ground_truth": ground_truth,
            "methods":      {},
        }

        for mode in ("vector", "bm25", "hybrid"):
            try:
                t0       = time.time()
                response = self.rag_engine.query(question, top_k=5, search_mode=mode)
                elapsed  = time.time() - t0

                if not response.get("success"):
                    results["methods"][mode] = {"success": False, "error": response.get("error")}
                    continue

                answer   = response.get("answer", "")
                contexts = [s["text"] for s in response.get("sources", [])]

                # Embedding-based metrics
                ans_rel  = round(self.evaluate_answer_relevance(question, answer), 3)
                ctx      = self.evaluate_context_relevance(question, contexts)
                faith_emb = round(self.evaluate_faithfulness(answer, contexts), 3)

                overall = round(
                    ans_rel * 0.3
                    + ctx["precision"] * 0.2
                    + ctx["recall"] * 0.2
                    + faith_emb * 0.3,
                    3,
                )

                method_result = {
                    "success":       True,
                    "answer":        answer,
                    "sources_count": len(contexts),
                    "response_time": round(elapsed, 3),
                    "is_extractive": self._is_extractive_fallback(answer),
                    "metrics": {
                        "answer_relevance":   ans_rel,
                        "context_precision":  ctx["precision"],
                        "context_recall":     ctx["recall"],
                        "faithfulness":       faith_emb,
                        "avg_relevance_score": ctx["mean_relevance"],
                        "overall_score":      overall,
                    },
                }

                # Ground truth comparison
                if ground_truth:
                    method_result["metrics"]["ground_truth_similarity"] = round(
                        self.calculate_semantic_similarity(answer, ground_truth), 3
                    )

                # LLM judge — deferred to winner only (saves 2/3 of quota)
                # Will be applied after winner selection below
                results["methods"][mode] = method_result

            except Exception as e:
                results["methods"][mode] = {"success": False, "error": str(e)}

        results["winner"] = self._determine_winner(results["methods"], question)

        # ── LLM judge + factual check — winner only (2 calls total, not 4) ────
        if run_llm_judge and self.llm_judge.available:
            overall_winner = results["winner"].get("overall", {})
            if overall_winner and overall_winner.get("method"):
                best_method  = overall_winner["method"]
                best_data    = results["methods"][best_method]
                best_answer  = best_data.get("answer", "")
                best_contexts= [s["text"] for s in
                                self.rag_engine.query(question, top_k=3,
                                search_mode=best_method).get("sources", [])]                                if self.rag_engine else []

                if best_answer:
                    # Call 1: faithfulness judge on winner
                    combined_ctx = "\n\n".join(best_contexts[:3]) if best_contexts else ""
                    judgment = self.llm_judge.judge(question, combined_ctx, best_answer)
                    results["methods"][best_method]["llm_judge"] = judgment
                    if judgment.get("faithfulness") is not None:
                        llm_f   = judgment["faithfulness"]
                        llm_c   = judgment.get("completeness", llm_f)
                        current = best_data["metrics"]["overall_score"]
                        blended = round(current * 0.5 + llm_f * 0.3 + llm_c * 0.2, 3)
                        results["methods"][best_method]["metrics"]["overall_score"] = blended
                        # Re-run winner selection with updated score
                        results["winner"] = self._determine_winner(results["methods"], question)

                    # Call 2: external factual check
                    factual = self.llm_judge.check_factual_accuracy(question, best_answer)
                    results["factual_check"] = {
                        **factual,
                        "checked_method": best_method,
                        "checked_answer": best_answer[:300],
                    }

        self.results_history.append(results)
        return results

    # ── Consistency testing ────────────────────────────────────────────────────

    def evaluate_consistency(
        self,
        question: str,
        n_runs: int = CONSISTENCY_RUNS,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Run the same question n_runs times and measure variance.

        Returns:
          consistency_score: 1 - normalised_std   (1.0 = perfectly consistent)
          scores:            list of overall_scores per run
          answers:           list of answers per run
          is_consistent:     bool (consistency_score ≥ CONSISTENCY_THRESHOLD)
        """
        if not self.rag_engine:
            return {"error": "RAG engine not initialised"}

        scores  = []
        answers = []

        for i in range(n_runs):
            response = self.rag_engine.query(question, top_k=5, search_mode=mode)
            if response.get("success"):
                answer   = response.get("answer", "")
                contexts = [s["text"] for s in response.get("sources", [])]
                ans_rel  = self.evaluate_answer_relevance(question, answer)
                ctx      = self.evaluate_context_relevance(question, contexts)
                faith    = self.evaluate_faithfulness(answer, contexts)
                overall  = round(ans_rel * 0.3 + ctx["precision"] * 0.2
                                 + ctx["recall"] * 0.2 + faith * 0.3, 3)
                scores.append(overall)
                answers.append(answer)

        if not scores:
            return {"error": "No successful runs"}

        mean  = float(np.mean(scores))
        std   = float(np.std(scores))
        norm_std = std / mean if mean > 0 else 0.0
        consistency_score = round(max(0.0, 1.0 - norm_std), 3)

        # Also measure answer-level similarity between runs
        answer_sims = []
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                answer_sims.append(
                    self.calculate_semantic_similarity(answers[i], answers[j])
                )

        return {
            "question":           question,
            "mode":               mode,
            "n_runs":             n_runs,
            "scores":             [round(s, 3) for s in scores],
            "mean_score":         round(mean, 3),
            "std_score":          round(std, 3),
            "consistency_score":  consistency_score,
            "is_consistent":      consistency_score >= CONSISTENCY_THRESHOLD,
            "answer_similarity":  round(float(np.mean(answer_sims)), 3) if answer_sims else 1.0,
        }

    # ── Batch evaluation ───────────────────────────────────────────────────────

    def batch_evaluate(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        test_consistency: bool = False,
    ) -> Dict:
        """
        Evaluate a list of questions and generate a comprehensive report.

        New: includes coverage breakdown and optional consistency testing.
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ground_truths and len(ground_truths) != len(questions):
            ground_truths = None

        print(f"\n🧪 Batch evaluation: {len(questions)} questions…")

        batch_results    = []
        consistency_data = []

        for i, question in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {question[:60]}…")
            gt     = ground_truths[i] if ground_truths else None
            result = self.evaluate_query(question, gt)
            batch_results.append(result)

            # Optional consistency test (adds latency — off by default)
            if test_consistency:
                c = self.evaluate_consistency(question, n_runs=CONSISTENCY_RUNS)
                consistency_data.append(c)

        report = self._generate_batch_report(batch_results, session_id, consistency_data)

        self.evaluation_sessions.append({
            "session_id": session_id,
            "timestamp":  datetime.now().isoformat(),
            "num_queries": len(questions),
            "report":     report,
        })

        return report

    # ── Report generation ──────────────────────────────────────────────────────

    def _generate_batch_report(
        self,
        results: List[Dict],
        session_id: str,
        consistency_data: List[Dict] = None,
    ) -> Dict:
        report = {
            "session_id":         session_id,
            "timestamp":          datetime.now().isoformat(),
            "summary":            {},
            "method_comparison":  {},
            "coverage":           {},
            "winner":             {},
            "consistency":        {},
            "detailed_results":   results,
        }

        # ── Summary ────────────────────────────────────────────────────────────
        report["summary"] = {
            "total_queries":      len(results),
            "successful_queries": sum(
                1 for r in results
                if any(m.get("success") for m in r["methods"].values())
            ),
            "total_time": round(sum(
                sum(m.get("response_time", 0) for m in r["methods"].values() if m.get("success"))
                for r in results
            ), 3),
            "llm_judge_used": self.llm_judge.available,
        }

        # ── Per-method aggregation ─────────────────────────────────────────────
        metrics_keys = [
            "answer_relevance", "context_precision", "context_recall",
            "faithfulness", "overall_score", "avg_relevance_score",
        ]

        for method in ("vector", "bm25", "hybrid"):
            method_data = [
                r["methods"][method]
                for r in results
                if r["methods"].get(method, {}).get("success")
            ]
            if not method_data:
                continue

            llm_faith_scores = [
                d["llm_judge"]["faithfulness"]
                for d in method_data
                if d.get("llm_judge") and d["llm_judge"].get("faithfulness") is not None
            ]
            llm_completeness_scores = [
                d["llm_judge"]["completeness"]
                for d in method_data
                if d.get("llm_judge") and d["llm_judge"].get("completeness") is not None
            ]

            report["method_comparison"][method] = {
                "total_queries":     len(method_data),
                "avg_response_time": round(np.mean([d["response_time"] for d in method_data]), 3),
                "extractive_rate":   round(
                    sum(1 for d in method_data if d.get("is_extractive")) / len(method_data), 3
                ),
                "metrics": {
                    key: {
                        "mean": round(float(np.mean([d["metrics"][key] for d in method_data])), 3),
                        "std":  round(float(np.std( [d["metrics"][key] for d in method_data])), 3),
                        "min":  round(float(np.min( [d["metrics"][key] for d in method_data])), 3),
                        "max":  round(float(np.max( [d["metrics"][key] for d in method_data])), 3),
                    }
                    for key in metrics_keys
                    if all(key in d["metrics"] for d in method_data)
                },
                "llm_judge": {
                    "avg_faithfulness":  round(float(np.mean(llm_faith_scores)), 3)   if llm_faith_scores  else None,
                    "avg_completeness":  round(float(np.mean(llm_completeness_scores)), 3) if llm_completeness_scores else None,
                    "n_judged":          len(llm_faith_scores),
                },
            }

        # ── Coverage ───────────────────────────────────────────────────────────
        # Use hybrid results (best mode) for coverage analysis
        hybrid_data = [
            r["methods"].get("hybrid", {})
            for r in results
        ]

        good_answers       = sum(
            1 for d in hybrid_data
            if d.get("success") and d.get("metrics", {}).get("overall_score", 0) >= GOOD_ANSWER_THRESHOLD
        )
        extractive_answers = sum(
            1 for d in hybrid_data
            if d.get("success") and d.get("is_extractive")
        )
        failed_answers     = sum(1 for d in hybrid_data if not d.get("success"))
        total              = len(hybrid_data)

        report["coverage"] = {
            "total_questions":  total,
            "good_answers":     good_answers,
            "extractive_answers": extractive_answers,
            "failed_answers":   failed_answers,
            "good_answer_rate": round(good_answers / total, 3) if total else 0,
            "extractive_rate":  round(extractive_answers / total, 3) if total else 0,
            "failure_rate":     round(failed_answers / total, 3) if total else 0,
            "threshold_used":   GOOD_ANSWER_THRESHOLD,
            "interpretation": (
                "✅ Strong coverage"   if good_answers / total >= 0.75 else
                "⚠️  Moderate coverage" if good_answers / total >= 0.5  else
                "❌ Poor coverage — consider improving chunking or retrieval"
            ) if total else "N/A",
        }

        # ── Winner ─────────────────────────────────────────────────────────────
        if report["method_comparison"]:
            wins = {"vector": 0, "bm25": 0, "hybrid": 0}
            for result in results:
                w = result.get("winner", {}).get("overall")
                if w:
                    wins[w["method"]] = wins.get(w["method"], 0) + 1

            best_method = max(wins.items(), key=lambda x: x[1])
            report["winner"] = {
                "best_overall_method": best_method[0],
                "win_count":           best_method[1],
                "win_rate":            round(best_method[1] / len(results), 3),
                "wins_by_method":      wins,
                "fastest_method": min(
                    report["method_comparison"].items(),
                    key=lambda x: x[1]["avg_response_time"],
                )[0],
                "most_relevant_method": max(
                    report["method_comparison"].items(),
                    key=lambda x: x[1]["metrics"].get("overall_score", {}).get("mean", 0),
                )[0],
            }

            # Which method had highest LLM judge faithfulness?
            if self.llm_judge.available:
                llm_winner = max(
                    (
                        (m, d["llm_judge"]["avg_faithfulness"])
                        for m, d in report["method_comparison"].items()
                        if d["llm_judge"]["avg_faithfulness"] is not None
                    ),
                    key=lambda x: x[1],
                    default=(None, None),
                )
                report["winner"]["most_faithful_method_llm"] = {
                    "method": llm_winner[0],
                    "score":  round(llm_winner[1], 3) if llm_winner[1] else None,
                }

        # ── Consistency ────────────────────────────────────────────────────────
        if consistency_data:
            consistent_count = sum(1 for c in consistency_data if c.get("is_consistent"))
            report["consistency"] = {
                "questions_tested":   len(consistency_data),
                "consistent_count":   consistent_count,
                "consistency_rate":   round(consistent_count / len(consistency_data), 3),
                "avg_consistency_score": round(
                    float(np.mean([c.get("consistency_score", 0) for c in consistency_data])), 3
                ),
                "per_question": consistency_data,
            }
        else:
            report["consistency"] = {
                "note": "Consistency testing not run. Pass test_consistency=True to batch_evaluate."
            }

        return report

    # ── Winner selection ───────────────────────────────────────────────────────

    def _determine_winner(self, methods: Dict, question: str = "") -> Dict:
        winner = {"overall": None, "fastest": None, "most_relevant": None}
        valid  = {k: v for k, v in methods.items() if v.get("success")}
        if not valid:
            return winner

        fastest = min(valid.items(), key=lambda x: x[1]["response_time"])
        winner["fastest"] = {"method": fastest[0], "time": round(fastest[1]["response_time"], 3)}

        most_rel = max(valid.items(), key=lambda x: x[1]["metrics"]["overall_score"])
        winner["most_relevant"] = {
            "method": most_rel[0],
            "score":  round(most_rel[1]["metrics"]["overall_score"], 3),
        }

        # Overall = quality only (deterministic)
        scores  = {m: d["metrics"]["overall_score"] for m, d in valid.items()}
        overall = max(scores.items(), key=lambda x: x[1])
        winner_method = overall[0]
        winner_score  = round(overall[1], 3)

        # ── Generate human-readable reasoning ─────────────────────────────────
        reasoning = self._generate_winner_reasoning(
            winner_method, question, valid, scores
        )

        winner["overall"] = {
            "method":    winner_method,
            "score":     winner_score,
            "reasoning": reasoning,
        }
        return winner

    # ── Winner reasoning engine ────────────────────────────────────────────────

    _QUERY_TYPE_HINTS = {
        # keywords → (query_type_label, why_vector, why_bm25, why_hybrid)
        "factual_specific": (
            ["what", "when", "who", "which", "how many", "how much",
             "percentage", "share", "year", "date", "name", "define"],
            "factual",
            "Semantic search (Vector) excels at factual questions because it maps "
            "your question directly to the most relevant passage by meaning, not just keywords.",
            "Keyword search (BM25) wins here because your question contains very specific "
            "terms that appear almost verbatim in the document — exact matches score highest.",
            "Hybrid wins because your question mixes specific terms with contextual meaning, "
            "so combining keyword precision with semantic understanding gives the best coverage.",
        ),
        "conceptual": (
            ["why", "how does", "explain", "describe", "what is the concept",
             "understand", "meaning", "purpose", "role"],
            "conceptual",
            "Semantic search (Vector) is ideal for conceptual questions — it finds passages "
            "that explain the idea even when they use different words than your question.",
            "Keyword search (BM25) wins because the concept name appears frequently in the "
            "document as a key term, making exact matches very precise.",
            "Hybrid wins because explaining a concept requires both finding the definition "
            "(semantic) and locating all mentions of the term (keyword).",
        ),
        "comparative": (
            ["compare", "difference", "versus", "vs", "better", "contrast",
             "distinguish", "similarities", "unlike", "than"],
            "comparative",
            "Semantic search (Vector) wins for comparison questions — it can surface "
            "passages discussing related topics even when they don't use identical phrasing.",
            "Keyword search (BM25) wins because both items being compared appear as exact "
            "terms in the document, giving high precision retrieval.",
            "Hybrid wins on comparison questions as expected — you need broad semantic "
            "coverage of both topics plus keyword precision for specific names/terms.",
        ),
        "complex": (
            ["analyze", "evaluate", "assess", "implications", "relationship",
             "discuss", "impact", "effect", "cause"],
            "complex/analytical",
            "Semantic search (Vector) handles complex questions well — your question "
            "requires understanding context across multiple passages, which vector similarity captures.",
            "Keyword search (BM25) performs well because your analytical question contains "
            "high-value domain terms that act as strong retrieval signals.",
            "Hybrid wins on complex questions — they need both the broad semantic net "
            "of vector search and the precision of keyword matching.",
        ),
    }

    def _classify_question(self, question: str) -> str:
        """Classify the question into a retrieval type."""
        q = question.lower()
        scores = {}
        for qtype, (keywords, *_) in self._QUERY_TYPE_HINTS.items():
            scores[qtype] = sum(1 for kw in keywords if kw in q)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "factual_specific"

    def _generate_winner_reasoning(
        self,
        winner: str,
        question: str,
        methods: Dict,
        scores: Dict,
    ) -> str:
        """
        Build a plain-English explanation of why `winner` won, referencing:
        - the query type inferred from the question
        - the actual score deltas between methods
        - the key winning metric
        - what the answer produced
        """
        q = question.lower().strip()

        # ── Score deltas ───────────────────────────────────────────────────────
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        runner_up     = sorted_scores[1][0] if len(sorted_scores) > 1 else None
        margin        = round((scores[winner] - scores[runner_up]) * 100, 1) if runner_up else 0
        winner_data   = methods[winner]
        m             = winner_data["metrics"]

        # ── Query type ────────────────────────────────────────────────────────
        qtype_key    = self._classify_question(question)
        _, qlabel, reason_vector, reason_bm25, reason_hybrid = self._QUERY_TYPE_HINTS[qtype_key]

        reason_map = {
            "vector": reason_vector,
            "bm25":   reason_bm25,
            "hybrid": reason_hybrid,
        }
        base_reason = reason_map.get(winner, "")

        # ── Winning metric ─────────────────────────────────────────────────────
        metric_labels = {
            "answer_relevance":  ("Answer Relevance", m.get("answer_relevance", 0)),
            "faithfulness":      ("Faithfulness",     m.get("faithfulness", 0)),
            "context_precision": ("Context Precision",m.get("context_precision", 0)),
        }
        top_metric = max(metric_labels.items(), key=lambda x: x[1][1])
        metric_name, metric_val = top_metric[1]

        # ── LLM judge note ─────────────────────────────────────────────────────
        lj = winner_data.get("llm_judge", {})
        judge_note = ""
        if lj and lj.get("faithfulness") is not None:
            judge_note = (
                f" The LLM judge also confirmed the answer is "
                f"{int(lj['faithfulness']*100)}% faithful to the document"
                f"{' and ' + str(int(lj['completeness']*100)) + '% complete' if lj.get('completeness') else ''}."
            )

        # ── Score comparison sentence ──────────────────────────────────────────
        score_sentence = ""
        if runner_up and margin > 0:
            score_sentence = (
                f" It scored {int(scores[winner]*100)}% overall — "
                f"{margin}% ahead of {runner_up.upper()} ({int(scores[runner_up]*100)}%)."
            )
        elif runner_up and margin == 0:
            score_sentence = (
                f" It tied with {runner_up.upper()} on overall score but won the "
                f"tie-break on answer relevance."
            )

        # ── Assemble ───────────────────────────────────────────────────────────
        reasoning = (
            f"Your question is **{qlabel}** in nature. {base_reason}"
            f"{score_sentence} Its strongest metric was **{metric_name}** "
            f"at {int(metric_val*100)}%.{judge_note}"
        )

        return reasoning



    # ── Persistence & visualisation ────────────────────────────────────────────

    def save_report(self, report: Dict, filename: Optional[str] = None) -> str:
        if not filename:
            filename = f"evaluation_results/eval_{report['session_id']}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved to {filename}")
        return filename

    def visualize_comparison(self, report: Dict, output_path: Optional[str] = None):
        if not report.get("method_comparison"):
            return
        if not output_path:
            output_path = f"evaluation_results/viz_{report['session_id']}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        methods = list(report["method_comparison"].keys())
        metrics = ["answer_relevance", "context_precision", "context_recall",
                   "faithfulness", "overall_score"]

        data = []
        for method in methods:
            for metric in metrics:
                val = report["method_comparison"][method]["metrics"].get(metric, {})
                data.append({
                    "Method": method.upper(),
                    "Metric": metric.replace("_", " ").title(),
                    "Score":  val.get("mean", 0) if isinstance(val, dict) else val,
                })

        df = pd.DataFrame(data)

        # ── Coverage data ──────────────────────────────────────────────────────
        coverage = report.get("coverage", {})

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("RAG Search Method Evaluation", fontsize=16, fontweight="bold")

        # 1. Overall scores
        ax = axes[0, 0]
        od = df[df["Metric"] == "Overall Score"]
        sns.barplot(data=od, x="Method", y="Score", ax=ax, palette="Set2")
        ax.set_title("Overall Performance Score")
        ax.set_ylim(0, 1)

        # 2. Metrics heatmap
        ax = axes[0, 1]
        pivot = df.pivot(index="Metric", columns="Method", values="Score")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        ax.set_title("Metrics Heatmap")

        # 3. Response time
        ax = axes[0, 2]
        td = pd.DataFrame([
            {"Method": m.upper(), "Time (ms)": report["method_comparison"][m]["avg_response_time"] * 1000}
            for m in methods
        ])
        sns.barplot(data=td, x="Method", y="Time (ms)", ax=ax, palette="Set1")
        ax.set_title("Average Response Time")

        # 4. Win rate
        ax = axes[1, 0]
        wins = report.get("winner", {}).get("wins_by_method", {})
        if wins and sum(wins.values()) > 0:
            ax.pie(wins.values(), labels=[k.upper() for k in wins], autopct="%1.1f%%")
        ax.set_title("Win Rate Distribution")

        # 5. Coverage breakdown (NEW)
        ax = axes[1, 1]
        if coverage:
            labels = ["Good", "Extractive", "Failed"]
            sizes  = [
                coverage.get("good_answers", 0),
                coverage.get("extractive_answers", 0),
                coverage.get("failed_answers", 0),
            ]
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]
            ax.bar(labels, sizes, color=colors)
            ax.set_title(f"Answer Coverage\n({coverage.get('interpretation', '')})")
            ax.set_ylabel("# Questions")
        else:
            ax.set_visible(False)

        # 6. LLM judge scores (NEW — shown if available)
        ax = axes[1, 2]
        llm_data = []
        for m in methods:
            lj = report["method_comparison"][m].get("llm_judge", {})
            if lj.get("avg_faithfulness") is not None:
                llm_data.append({
                    "Method": m.upper(),
                    "Faithfulness": lj["avg_faithfulness"],
                    "Completeness": lj.get("avg_completeness", 0),
                })

        if llm_data:
            ldf = pd.DataFrame(llm_data).set_index("Method")
            ldf.plot(kind="bar", ax=ax, colormap="Set3", rot=0)
            ax.set_title("LLM Judge Scores\n(Faithfulness & Completeness)")
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right")
        else:
            ax.set_title("LLM Judge\n(not available)")
            ax.text(0.5, 0.5, "Set GEMINI_API_KEY\n(free) or ANTHROPIC_API_KEY\nor OPENAI_API_KEY\nto enable",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Visualisation saved to {output_path}")
        return output_path

    def export_to_excel(self, report: Dict, filename: Optional[str] = None):
        if not filename:
            filename = f"eval_{report['session_id']}.xlsx"
        filepath = f"evaluation_results/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Summary sheet
            summary_rows = []
            for method, data in report["method_comparison"].items():
                row = {"Method": method.upper()}
                row["Avg Response Time (ms)"] = round(data["avg_response_time"] * 1000, 1)
                row["Extractive Rate"] = data.get("extractive_rate", "N/A")
                for metric, vals in data["metrics"].items():
                    row[f"{metric} (mean)"] = round(vals["mean"], 3) if isinstance(vals, dict) else vals
                lj = data.get("llm_judge", {})
                row["LLM Faithfulness"] = lj.get("avg_faithfulness", "N/A")
                row["LLM Completeness"] = lj.get("avg_completeness", "N/A")
                summary_rows.append(row)
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

            # Coverage sheet (NEW)
            coverage = report.get("coverage", {})
            if coverage:
                cov_df = pd.DataFrame([{
                    "Metric":           k,
                    "Value":            v,
                } for k, v in coverage.items() if k != "interpretation"])
                cov_df.to_excel(writer, sheet_name="Coverage", index=False)

            # Consistency sheet (NEW)
            consistency = report.get("consistency", {})
            per_q = consistency.get("per_question", [])
            if per_q:
                cons_df = pd.DataFrame([{
                    "Question":          c.get("question", "")[:60],
                    "Consistency Score": c.get("consistency_score"),
                    "Is Consistent":     c.get("is_consistent"),
                    "Mean Score":        c.get("mean_score"),
                    "Std Score":         c.get("std_score"),
                    "Answer Similarity": c.get("answer_similarity"),
                } for c in per_q])
                cons_df.to_excel(writer, sheet_name="Consistency", index=False)

            # Detailed results
            detail_rows = []
            for i, result in enumerate(report["detailed_results"]):
                for method, data in result["methods"].items():
                    if data.get("success"):
                        row = {
                            "Query #":            i + 1,
                            "Question":           result["question"][:80],
                            "Method":             method.upper(),
                            "Response Time (ms)": round(data["response_time"] * 1000, 1),
                            "Is Extractive":      data.get("is_extractive", False),
                        }
                        for k, v in data["metrics"].items():
                            row[k] = round(v, 3) if isinstance(v, float) else v
                        lj = data.get("llm_judge", {})
                        row["LLM Faithfulness"] = lj.get("faithfulness", "N/A")
                        row["LLM Completeness"] = lj.get("completeness", "N/A")
                        row["LLM Reasoning"]    = lj.get("reasoning", "")
                        detail_rows.append(row)
            pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Detailed Results", index=False)

        print(f"✅ Excel report saved to {filepath}")
        return filepath

    # ── Legacy shims ───────────────────────────────────────────────────────────

    def evaluate_response(self, question, answer, contexts, ground_truth=None):
        """Legacy method kept for backward compatibility."""
        evaluation = {
            "timestamp":    datetime.now().isoformat(),
            "question":     question,
            "answer_length": len(answer) if answer else 0,
            "num_contexts":  len(contexts) if contexts else 0,
            "metrics":       {},
        }
        evaluation["metrics"]["answer_relevance"] = self.evaluate_answer_relevance(question, answer)
        ctx = self.evaluate_context_relevance(question, contexts)
        evaluation["metrics"]["context_precision"]     = ctx["precision"]
        evaluation["metrics"]["context_recall"]        = ctx["recall"]
        evaluation["metrics"]["context_mean_relevance"] = ctx["mean_relevance"]
        evaluation["metrics"]["faithfulness"]          = self.evaluate_faithfulness(answer, contexts)
        if ground_truth:
            evaluation["metrics"]["ground_truth_similarity"] = self.calculate_semantic_similarity(answer, ground_truth)
        weights = {"answer_relevance": 0.3, "context_precision": 0.2, "context_recall": 0.2, "faithfulness": 0.3}
        evaluation["metrics"]["overall_score"] = round(
            sum(evaluation["metrics"].get(m, 0) * w for m, w in weights.items()), 3
        )
        self.evaluation_history.append(evaluation)
        return evaluation

    def get_evaluation_summary(self):
        if not self.evaluation_history:
            return {"message": "No evaluations recorded yet"}
        keys = ["answer_relevance", "context_precision", "context_recall", "faithfulness", "overall_score"]
        summary = {"total_evaluations": len(self.evaluation_history), "metrics_summary": {}}
        for key in keys:
            vals = [e["metrics"].get(key, 0) for e in self.evaluation_history]
            summary["metrics_summary"][key] = {
                "mean": round(float(np.mean(vals)), 3),
                "std":  round(float(np.std(vals)), 3),
                "min":  round(float(np.min(vals)), 3),
                "max":  round(float(np.max(vals)), 3),
            }
        return summary

    def clear_history(self):
        self.evaluation_history = []