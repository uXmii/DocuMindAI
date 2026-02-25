"""
Agentic RAG with LangGraph - PRODUCTION VERSION
Agent and dashboard use the same RAGEvaluator scorer so winner is always consistent.
"""

import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from agent_state import AgentState
from agent_tools import QueryAnalyzer, AnswerRefiner


class AgenticRAG:
    """
    Intelligent Agentic RAG.
    Pass the same RAGEvaluator instance used by the dashboard so both
    always agree on the winning search mode.
    """

    def __init__(self, rag_engine, evaluator=None):
        self.rag_engine  = rag_engine
        self.evaluator   = evaluator          # shared RAGEvaluator instance
        self.query_analyzer  = QueryAnalyzer()
        self.answer_refiner  = AnswerRefiner()

        self.evaluation_cache      = {}
        self.SCORE_MARGIN_THRESHOLD = 0.005   # absolute — only truly near-identical scores tie-break
        # Preference order used ONLY as last resort (both previous tie-breaks failed)
        self.MODE_PREFERENCE_ORDER  = ['hybrid', 'vector', 'bm25']

        self.workflow = self._build_graph()
        self.app      = self.workflow.compile()

    # =========================================================================
    # GRAPH
    # =========================================================================

    def _build_graph(self) -> StateGraph:
        wf = StateGraph(AgentState)

        wf.add_node("analyze_query",      self.analyze_query)
        wf.add_node("evaluate_all_modes", self.evaluate_all_modes)
        wf.add_node("select_best_mode",   self.select_best_mode)
        wf.add_node("evaluate_quality",   self.evaluate_quality)
        wf.add_node("refine_approach",    self.refine_approach)
        wf.add_node("finalize_answer",    self.finalize_answer)

        wf.set_entry_point("analyze_query")
        wf.add_edge("analyze_query",      "evaluate_all_modes")
        wf.add_edge("evaluate_all_modes", "select_best_mode")
        wf.add_edge("select_best_mode",   "evaluate_quality")
        wf.add_conditional_edges(
            "evaluate_quality",
            self.should_continue,
            {"refine": "refine_approach", "finalize": "finalize_answer"}
        )
        wf.add_edge("refine_approach", "evaluate_all_modes")
        wf.add_edge("finalize_answer", END)
        return wf

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_cache_key(self, question: str) -> str:
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def _ensure_proper_formatting(self, answer: str, sources: list) -> str:
        import re
        if not answer:
            return answer
        if hasattr(self.rag_engine, '_clean_ocr_text'):
            answer = self.rag_engine._clean_ocr_text(answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'\b([A-Z])\s+\1(\s+\1)+\b', r'\1', answer)
        answer = re.sub(r'\b([A-Z])\s+\1\b', r'\1', answer)
        answer = re.sub(r'\b(\w+)\s+\1\b', r'\1', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s+([.,!?;:])', r'\1', answer)
        answer = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', answer)
        words    = answer.split()
        filtered = [w for w in words if len(w) >= 3 or w in list('.!?,;:')]
        answer   = ' '.join(filtered)
        answer   = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), answer)
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        lines = answer.split('\n')
        clean = []
        for line in lines:
            ws = line.split()
            if not ws:
                continue
            singles = sum(1 for w in ws if len(w) == 1 and w.isalpha())
            if singles / len(ws) < 0.5:
                clean.append(line)
        return re.sub(r'\s+', ' ', '\n'.join(clean)).strip()

    # =========================================================================
    # SCORING  (all scoring lives here so it's easy to maintain)
    # =========================================================================

    def _score_mode_with_evaluator(self, question: str, result: Dict) -> float:
        """
        Score using the SAME formula as evaluation_metrics.RAGEvaluator.evaluate_query:
            overall = answer_relevance*0.3 + precision*0.2 + recall*0.2 + faithfulness*0.3
        Falls back to internal scorer when no evaluator is injected.
        """
        answer   = result.get('answer', '')
        sources  = result.get('sources', [])
        contexts = [s['text'] for s in sources]

        if self.evaluator is None:
            return self._score_mode_internal(result)

        ar    = round(self.evaluator.evaluate_answer_relevance(question, answer), 3)
        ctx   = self.evaluator.evaluate_context_relevance(question, contexts)
        faith = round(self.evaluator.evaluate_faithfulness(answer, contexts), 3)

        return round(ar*0.3 + ctx['precision']*0.2 + ctx['recall']*0.2 + faith*0.3, 3)

    def _score_mode_internal(self, result: Dict) -> float:
        """Lightweight fallback when no evaluator is available."""
        answer  = result.get('answer', '')
        n_src   = len(result.get('sources', []))
        length  = len(answer)

        quality = sum([
            min(length / 500.0, 1.0),
            1.0 if any(c.isdigit() for c in answer) else 0.3,
            1.0 if answer.count('.') >= 2 else 0.5,
            1.0 if length > 100 else 0.3,
            1.0 if '\n' in answer else 0.7,
        ]) / 5.0

        source_score = 0.0 if n_src == 0 else (1.0 if n_src >= 5 else (0.8 if n_src >= 3 else 0.5))

        al = answer.lower()
        citation_raw = (
            (2.0 if ('page' in al and any(c.isdigit() for c in answer)) else 0) +
            (1.5 if ('source' in al or 'based on' in al) else 0) +
            (1.0 if ('.pdf' in al or 'document' in al) else 0) +
            (1.0 if 'according to' in al else 0) +
            (1.5 if ('**Source:**' in answer or '[Source:' in answer) else 0)
        )
        citation_score = min(citation_raw / 6.0, 1.0)

        return round(quality*0.50 + source_score*0.30 + citation_score*0.20, 3)

    def _get_evaluation_winner(
        self,
        candidate_modes: List[str],
        comparison_results: Dict,
        question: str
    ) -> Optional[str]:
        """
        Among candidate_modes, return the one with the highest answer_relevance
        at 4-decimal precision.  This is the primary signal the dashboard uses,
        so the agent and dashboard will always agree.
        Returns None when all candidates are still tied (identical answers).
        """
        if not self.evaluator:
            return None

        relevance: Dict[str, float] = {}
        for mode in candidate_modes:
            if mode in comparison_results and comparison_results[mode].get('success'):
                answer = comparison_results[mode].get('answer', '')
                relevance[mode] = round(
                    self.evaluator.evaluate_answer_relevance(question, answer), 4
                )

        if not relevance:
            return None

        # Sort descending so winners[0] is always the highest scorer
        sorted_relevance = sorted(relevance.items(), key=lambda x: x[1], reverse=True)
        best = sorted_relevance[0][1]
        winners = [m for m, s in sorted_relevance if s == best]

        print(f"   📐 Eval tie-break (4dp): "
              f"{', '.join(f'{m.upper()}:{s:.4f}' for m, s in sorted_relevance)}")

        return sorted_relevance[0][0] if len(winners) == 1 else None

    # =========================================================================
    # NODES
    # =========================================================================

    def analyze_query(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}\n🧠 ANALYZE QUERY (iter {state['iteration']})\n{'='*80}\n")

        analysis = self.query_analyzer.analyze_query(state["question"])
        state["query_type"] = analysis["query_type"]

        thought = f"🧠 Type='{analysis['query_type']}', Complexity='{analysis['complexity']}'"
        state["agent_thoughts"].append(thought)
        state["workflow_path"].append("analyze_query")
        print(f"   {thought}")
        print(f"   📋 {analysis['reasoning']}")
        print(f"   🔑 Key terms: {', '.join(analysis['key_terms'])}")
        return state

    def evaluate_all_modes(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}\n🔬 EVALUATE ALL MODES (iter {state['iteration']})\n{'='*80}\n")

        original = state["original_question"]
        question = state["question"]
        iteration = state["iteration"]
        cache_key = self._get_cache_key(original)

        if iteration == 0 and cache_key in self.evaluation_cache:
            print("💾 Cache HIT\n")
            state["search_results"] = self.evaluation_cache[cache_key]
            for mode in ['vector', 'bm25', 'hybrid']:
                if mode in self.evaluation_cache[cache_key]:
                    state["search_times"][mode] = self.evaluation_cache[cache_key][mode].get('search_time', 0)
            state["workflow_path"].append("evaluate_all_modes")
            return state

        if iteration > 0:
            question = self.answer_refiner.refine_query(original, iteration)
            print(f"🔄 Refined query: '{question}'")

        print("🔬 Running all 3 search modes...")
        t0 = time.time()
        results = self.rag_engine.compare_search_modes(question, top_k=5)
        print(f"✅ Done in {time.time()-t0:.2f}s")

        state["search_results"] = results
        for mode in ['vector', 'bm25', 'hybrid']:
            if mode in results and results[mode].get('success'):
                state["search_times"][mode] = results[mode].get('search_time', 0)

        if iteration == 0:
            self.evaluation_cache[cache_key] = results

        print("\n   📊 Results summary:")
        for mode in ['vector', 'bm25', 'hybrid']:
            if mode in results and results[mode].get('success'):
                r = results[mode]
                print(f"      • {mode.upper():8} {len(r.get('sources',[]))} sources  "
                      f"{r.get('search_time',0)*1000:.0f}ms")

        state["workflow_path"].append("evaluate_all_modes")
        return state

    def select_best_mode(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}\n🎯 SELECT BEST MODE\n{'='*80}\n")

        comparison_results = state["search_results"]
        question           = state["original_question"]
        scorer_label       = "RAGEvaluator (matches dashboard)" if self.evaluator else "internal fallback"
        print(f"   Scoring engine: {scorer_label}\n")

        # ── Score every mode ──────────────────────────────────────────────────
        mode_scores: Dict[str, float] = {}
        for mode in ['vector', 'bm25', 'hybrid']:
            if mode in comparison_results and comparison_results[mode].get('success'):
                mode_scores[mode] = self._score_mode_with_evaluator(
                    question, comparison_results[mode]
                )

        if not mode_scores:
            best_mode = 'hybrid'
            print("⚠️  All modes failed — defaulting to HYBRID")
        else:
            sorted_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)
            best_mode    = sorted_modes[0][0]
            best_score   = sorted_modes[0][1]

            # Print score table
            print("   📊 SCORES (same formula as Evaluation Results dashboard):")
            print(f"   {'─'*45}")
            for mode, score in sorted_modes:
                marker = "  ◀" if mode == best_mode else ""
                print(f"      {mode.upper():8}  {score:.3f}{marker}")
            print()

            # ── Tie-breaking ──────────────────────────────────────────────────
            # Use absolute threshold: modes must score within 0.01 of the best
            close_modes = [m for m, s in sorted_modes
                           if best_score - s <= self.SCORE_MARGIN_THRESHOLD]

            if len(close_modes) > 1:
                scores_str = ', '.join(f"{m.upper()}:{mode_scores[m]:.3f}" for m in close_modes)
                thought = f"⚖️  Close scores (within {self.SCORE_MARGIN_THRESHOLD*100:.0f}%): {scores_str}"
                state["agent_thoughts"].append(thought)
                print(f"   {thought}")

                # Tie-break 1: higher answer_relevance at 4 decimal places
                # — matches the dashboard's "Most Relevant" winner
                eval_winner = self._get_evaluation_winner(
                    close_modes, comparison_results, question
                )
                if eval_winner:
                    best_mode = eval_winner
                    thought = (f"🏆 Tie broken by evaluation relevance score "
                               f"— matches dashboard winner: {best_mode.upper()}")
                    state["agent_thoughts"].append(thought)
                    print(f"   {thought}")

                else:
                    # Tie-break 2: fastest response among tied modes
                    fastest_mode = min(
                        close_modes,
                        key=lambda m: comparison_results[m].get('search_time', 9999)
                    )
                    best_mode = fastest_mode
                    t = comparison_results[best_mode].get('search_time', 0) * 1000
                    thought = (f"⚡ Tie broken by speed "
                               f"— {best_mode.upper()} fastest at {t:.0f}ms")
                    state["agent_thoughts"].append(thought)
                    print(f"   {thought}")

                thought = f"✅ Selected {best_mode.upper()}"
            else:
                margin = (
                    ((best_score - sorted_modes[1][1]) / best_score * 100)
                    if len(sorted_modes) > 1 else 0
                )
                thought = (f"🏆 Clear winner: {best_mode.upper()} "
                           f"(score {best_score:.3f}, +{margin:.1f}% margin)")

            state["agent_thoughts"].append(thought)
            print(f"   {thought}\n")

        # ── Use winner's results ──────────────────────────────────────────────
        winner_result  = comparison_results[best_mode]
        raw_answer     = winner_result.get("answer", "")
        sources        = winner_result.get("sources", [])
        cleaned_answer = self._ensure_proper_formatting(raw_answer, sources)

        # Store the winner's score directly so evaluate_quality uses
        # the EXACT same number — no recomputation, no floating point drift
        winner_score = mode_scores.get(best_mode, 0.0) if mode_scores else 0.0

        state["search_mode"]    = best_mode
        state["retrieved_docs"] = sources
        state["answer"]         = cleaned_answer
        state["quality_score"]  = winner_score   # pre-fill so evaluate_quality reads this
        state["answer_attempts"].append(cleaned_answer)
        state["tools_used"].append(f"eval_selection_{best_mode}")

        thought = f"✅ Using {best_mode.upper()} answer with {len(sources)} sources"
        state["agent_thoughts"].append(thought)
        print(f"   {thought}")

        state["workflow_path"].append("select_best_mode")
        return state

    def evaluate_quality(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}\n📊 EVALUATE QUALITY (iter {state['iteration']})\n{'='*80}\n")

        question  = state["original_question"]
        answer    = state["answer"]
        sources   = state["retrieved_docs"]
        iteration = state["iteration"]
        max_iter  = state["max_iterations"]

        if self.evaluator and answer and sources:
            contexts      = [s['text'] for s in sources]
            ar            = self.evaluator.evaluate_answer_relevance(question, answer)
            ctx           = self.evaluator.evaluate_context_relevance(question, contexts)
            faith         = self.evaluator.evaluate_faithfulness(answer, contexts)
            quality_score = round(ar*0.3 + ctx['precision']*0.2 + ctx['recall']*0.2 + faith*0.3, 3)
            has_sources   = len(sources) > 0
            is_complete   = len(answer) > 80
        elif self.evaluator and answer and not sources:
            ar            = self.evaluator.evaluate_answer_relevance(question, answer)
            quality_score = round(ar, 3)
            has_sources   = False
            is_complete   = len(answer) > 80
        else:
            from agent_tools import AnswerQualityEvaluator
            assessment    = AnswerQualityEvaluator().evaluate(question, answer, sources)
            quality_score = assessment["score"]
            has_sources   = assessment["has_sources"]
            is_complete   = assessment["is_complete"]

        # Use the pre-filled score from select_best_mode if it's already set
        # and higher — avoids recomputation drift causing false refinement triggers
        prefilled = state.get("quality_score", 0.0)
        if prefilled > 0 and abs(prefilled - quality_score) < 0.05:
            quality_score = prefilled  # trust the score set by select_best_mode

        state["quality_score"]          = quality_score
        state["has_sufficient_sources"] = has_sources
        state["answer_is_complete"]     = is_complete

        print(f"   📈 Quality: {quality_score:.3f}  sources: {len(sources)}")

        if quality_score >= 0.82 or iteration >= max_iter or quality_score >= 1.0:
            needs_refinement = False
            reason = f"Score {quality_score:.3f} ≥ 0.82 threshold — answer is good enough"
        elif not has_sources:
            needs_refinement = iteration < max_iter
            reason = "No sources retrieved — retrying with refined query"
        else:
            needs_refinement = iteration < max_iter
            reason = f"Score {quality_score:.3f} < 0.85 threshold — retrying to find better answer"

        state["needs_refinement"] = needs_refinement
        print(f"   🎯 {'🔄 REFINE' if needs_refinement else '✅ FINALIZE'} — {reason}\n")
        state["workflow_path"].append("evaluate_quality")
        return state

    def refine_approach(self, state: AgentState) -> AgentState:
        state["iteration"] = state["iteration"] + 1
        thought = f"🔧 Refining — attempt #{state['iteration'] + 1}"
        state["agent_thoughts"].append(thought)
        print(f"\n{thought}")
        state["workflow_path"].append("refine_approach")
        return state

    def finalize_answer(self, state: AgentState) -> AgentState:
        print(f"\n{'='*80}\n✨ FINALIZE\n{'='*80}\n")

        if len(state["answer_attempts"]) > 1:
            best = max(
                state["answer_attempts"],
                key=lambda a: len(a) * (
                    1.5 if ('page' in a.lower() or 'source' in a.lower()) else 1.0
                )
            )
            state["final_answer"] = best
        else:
            state["final_answer"] = state["answer"]

        state["confidence"] = state["quality_score"]

        thought = (f"✨ Finalized with {state['confidence']:.0%} confidence "
                   f"after {state['iteration']+1} attempt(s)")
        state["agent_thoughts"].append(thought)
        print(f"   {thought}")
        state["workflow_path"].append("finalize_answer")
        return state

    def should_continue(self, state: AgentState) -> str:
        return "refine" if state["needs_refinement"] else "finalize"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def query(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        print("\n" + "="*80)
        print(f"🤖 AGENTIC RAG  |  {question}")
        print("="*80)

        t0 = time.time()

        initial_state = {
            "question": question,
            "original_question": question,
            "search_mode": "",
            "query_type": "",
            "retrieved_docs": [],
            "search_results": {},
            "answer": "",
            "answer_attempts": [],
            "quality_score": 0.0,
            "has_sufficient_sources": False,
            "answer_is_complete": False,
            "agent_thoughts": [],
            "tools_used": [],
            "iteration": 0,
            "max_iterations": max_iterations,
            "needs_refinement": False,
            "should_use_web": False,
            "final_answer": "",
            "confidence": 0.0,
            "workflow_path": [],
            "total_time": 0.0,
            "search_times": {}
        }

        final_state = self.app.invoke(initial_state)
        total_time  = time.time() - t0
        final_state["total_time"] = total_time

        print("\n" + "="*80)
        print(f"✅ DONE  {total_time:.2f}s  |  mode={final_state['search_mode'].upper()}"
              f"  |  quality={final_state['quality_score']:.3f}"
              f"  |  confidence={final_state['confidence']:.0%}")
        print(f"   Path: {' → '.join(final_state['workflow_path'])}")
        print("="*80 + "\n")

        return {
            "success":      True,
            "answer":       final_state["final_answer"],
            "confidence":   final_state["confidence"],
            "quality_score":final_state["quality_score"],
            "sources":      final_state["retrieved_docs"],
            "metadata": {
                "query_type":         final_state["query_type"],
                "final_search_mode":  final_state["search_mode"],
                "iterations":         final_state["iteration"] + 1,
                "total_time":         total_time,
                "search_times":       final_state["search_times"],
                "workflow_path":      final_state["workflow_path"],
                "agent_thoughts":     final_state["agent_thoughts"],
                "tools_used":         final_state["tools_used"],
                "all_modes_evaluated":True,
                "selection_method":   "evaluation_metrics_consistent_scoring",
                "cache_hit":          self._get_cache_key(question) in self.evaluation_cache,
                "scorer":             "RAGEvaluator" if self.evaluator else "internal_fallback"
            }
        }

    def clear_cache(self):
        n = len(self.evaluation_cache)
        self.evaluation_cache.clear()
        print(f"✅ Cleared {n} cached evaluations")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "total_cached_queries": len(self.evaluation_cache),
            "cache_keys":           list(self.evaluation_cache.keys())
        }

    def visualize_workflow(self, output_path: str = "agentic_rag_workflow.png"):
        try:
            with open(output_path, 'wb') as f:
                f.write(self.app.get_graph().draw_mermaid_png())
            print(f"✅ Workflow saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"⚠️  Could not generate visualization: {e}")
            return None