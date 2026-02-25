"""
Agent Tools - Functions that the agent can use
"""

import re
from typing import List, Dict
from agent_state import QueryAnalysis, QualityAssessment


class QueryAnalyzer:
    """Analyze queries to determine optimal search strategy"""
    
    FACTUAL_INDICATORS = ['what', 'who', 'when', 'where', 'which', 'how many', 'how much', 'list', 'name']
    CONCEPTUAL_INDICATORS = ['why', 'how does', 'explain', 'describe', 'what is the concept', 'what is the meaning', 'understand']
    COMPARISON_INDICATORS = ['compare', 'difference', 'versus', 'vs', 'better', 'contrast', 'distinguish']
    COMPLEX_INDICATORS = ['analyze', 'evaluate', 'assess', 'implications', 'relationship between', 'discuss']
    
    @staticmethod
    def analyze_query(question: str) -> QueryAnalysis:
        """Analyze query and intelligently recommend search strategy"""
        question_lower = question.lower()
        
        # Extract key terms
        words = question.split()
        stopwords = {'what', 'when', 'where', 'which', 'that', 'this', 'these', 'those', 'with', 'from', 'are', 'the'}
        key_terms = [w for w in words if len(w) > 3 and w.lower() not in stopwords]
        
        # Check all indicators
        is_factual = any(ind in question_lower for ind in QueryAnalyzer.FACTUAL_INDICATORS)
        is_conceptual = any(ind in question_lower for ind in QueryAnalyzer.CONCEPTUAL_INDICATORS)
        is_comparison = any(ind in question_lower for ind in QueryAnalyzer.COMPARISON_INDICATORS)
        is_complex = any(ind in question_lower for ind in QueryAnalyzer.COMPLEX_INDICATORS)
        
        # Count question complexity
        complexity_score = sum([is_factual, is_conceptual, is_comparison, is_complex])
        
        # INTELLIGENT MODE SELECTION
        if is_complex or complexity_score > 1:
            # Complex or multi-faceted question → HYBRID
            query_type = "complex"
            recommended_mode = "hybrid"
            complexity = "complex"
            reasoning = "Complex/multi-faceted question - HYBRID search combines semantic understanding with keyword precision"
        
        elif is_comparison:
            # Comparison needs comprehensive results → HYBRID
            query_type = "comparative"
            recommended_mode = "hybrid"
            complexity = "medium"
            reasoning = "Comparison question - HYBRID search retrieves diverse perspectives"
        
        elif is_conceptual:
            # Conceptual understanding → VECTOR (semantic)
            query_type = "conceptual"
            recommended_mode = "vector"
            complexity = "medium"
            reasoning = "Conceptual question - VECTOR search for semantic understanding"
        
        elif is_factual and len(key_terms) <= 2:
            # Simple factual with few keywords → BM25
            query_type = "factual"
            recommended_mode = "bm25"
            complexity = "simple"
            reasoning = f"Simple factual question with specific terms ({', '.join(key_terms[:2])}) - BM25 keyword search"
        
        elif is_factual and len(key_terms) > 2:
            # Factual but many keywords → HYBRID for better coverage
            query_type = "factual"
            recommended_mode = "hybrid"
            complexity = "medium"
            reasoning = f"Multi-term factual question ({len(key_terms)} keywords) - HYBRID for comprehensive results"
        
        else:
            # Uncertain → HYBRID (safest choice)
            query_type = "general"
            recommended_mode = "hybrid"
            complexity = "medium"
            reasoning = "General question - HYBRID search for balanced results"
        
        return QueryAnalysis(
            query_type=query_type,
            recommended_mode=recommended_mode,
            complexity=complexity,
            requires_comparison=is_comparison,
            requires_calculation='calculate' in question_lower or 'compute' in question_lower,
            key_terms=key_terms[:5],
            reasoning=reasoning
        )


class AnswerQualityEvaluator:
    """Evaluate the quality of generated answers"""
    
    @staticmethod
    def evaluate(question: str, answer: str, sources: List[Dict]) -> QualityAssessment:
        """Evaluate answer quality with intelligent metrics"""
        
        issues = []
        suggestions = []
        score = 0.0
        
        # Check 1: Has sources (25%)
        has_sources = len(sources) > 0
        if has_sources:
            score += 0.25
            # Bonus for multiple quality sources
            if len(sources) >= 5:
                score += 0.05
        else:
            issues.append("No sources retrieved")
        
        # Check 2: Answer length (20%)
        answer_length = len(answer.strip())
        answer_length_ok = 100 < answer_length < 2000
        
        if answer_length_ok:
            score += 0.2
            # Bonus for comprehensive answers (400+ chars)
            if answer_length > 400:
                score += 0.05
        elif answer_length < 100:
            issues.append("Answer needs more detail")
        else:
            score += 0.15
        
        # Check 3: Has proper citations (25%)
        citation_patterns = [
            'page' in answer.lower(),
            'source' in answer.lower(),
            'based on' in answer.lower(),
            '.pdf' in answer.lower(),
            'according to' in answer.lower()
        ]
        citation_count = sum(citation_patterns)
        
        has_citations = citation_count > 0
        if citation_count >= 2:
            score += 0.25  # Multiple citation types
        elif has_citations:
            score += 0.15  # Some citations
        else:
            issues.append("Should cite sources")
        
        # Check 4: Relevance - keyword overlap (15%)
        question_words = set(w.lower() for w in question.split() if len(w) > 3)
        answer_words = set(w.lower() for w in answer.split())
        overlap = len(question_words & answer_words)
        
        # Calculate relevance percentage
        if len(question_words) > 0:
            relevance_pct = overlap / len(question_words)
        else:
            relevance_pct = 0
        
        is_relevant = relevance_pct >= 0.3  # At least 30% overlap
        if relevance_pct >= 0.5:
            score += 0.15  # High relevance
        elif is_relevant:
            score += 0.10  # Moderate relevance
        else:
            issues.append("Answer may not fully address question")
        
        # Check 5: Completeness (15%)
        is_complete = True
        
        # Check for substance
        sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
        
        if answer_length < 80:
            is_complete = False
            issues.append("Answer too brief")
        elif sentence_count < 2:
            is_complete = False
            issues.append("Needs multiple sentences for completeness")
        elif answer.strip().endswith('?'):
            is_complete = False
            issues.append("Answer ends with question mark (uncertain)")
        
        if is_complete:
            score += 0.15
        else:
            score += 0.05  # Partial credit
        
        # Cap at 1.0
        score = min(score, 1.0)
        
        return QualityAssessment(
            score=score,
            has_sources=has_sources,
            answer_length_ok=answer_length_ok,
            has_citations=has_citations,
            is_relevant=is_relevant,
            is_complete=is_complete,
            issues=issues,
            suggestions=suggestions
        )


class SearchModeSelector:
    """Select optimal search mode based on context and history"""
    
    @staticmethod
    def select_mode(state: Dict) -> str:
        """
        Intelligently select next search mode to try
        
        Strategy:
        1. If haven't tried HYBRID → Try HYBRID (it's usually best)
        2. If started with BM25 → Try VECTOR next
        3. If started with VECTOR → Try BM25 next  
        4. If started with HYBRID → Try VECTOR for semantic depth
        """
        iteration = state.get('iteration', 0)
        current_mode = state.get('search_mode', '')
        tried_modes = set(state.get('search_times', {}).keys())
        
        print(f"\n   🎯 MODE SELECTION LOGIC:")
        print(f"      Current iteration: {iteration}")
        print(f"      Current mode: {current_mode.upper()}")
        print(f"      Tried so far: {', '.join(m.upper() for m in tried_modes) if tried_modes else 'None'}")
        
        # Priority 1: Always try HYBRID if we haven't
        if 'hybrid' not in tried_modes:
            print(f"      → Selecting HYBRID (best overall performance)")
            return 'hybrid'
        
        # Priority 2: If started with BM25, try VECTOR for semantic understanding
        if current_mode == 'bm25' and 'vector' not in tried_modes:
            print(f"      → Selecting VECTOR (add semantic understanding to keyword search)")
            return 'vector'
        
        # Priority 3: If started with VECTOR, try BM25 for exact matches
        if current_mode == 'vector' and 'bm25' not in tried_modes:
            print(f"      → Selecting BM25 (add keyword precision to semantic search)")
            return 'bm25'
        
        # Priority 4: Try whatever we haven't tried
        all_modes = ['hybrid', 'vector', 'bm25']
        for mode in all_modes:
            if mode not in tried_modes:
                print(f"      → Selecting {mode.upper()} (not yet tested)")
                return mode
        
        # Fallback: Use HYBRID (best mode)
        print(f"      → Defaulting to HYBRID (best mode)")
        return 'hybrid'


class AnswerRefiner:
    """Refine and improve answers"""
    
    @staticmethod
    def refine_query(original_question: str, iteration: int) -> str:
        """Intelligently refine query based on iteration"""
        
        if iteration == 1:
            # Add more specificity
            refined = f"Provide comprehensive explanation of: {original_question}"
            print(f"      📝 Query refinement: Adding specificity")
            return refined
        
        elif iteration == 2:
            # Simplify to key terms
            words = original_question.split()
            key_words = [w for w in words if len(w) > 4]
            refined = " ".join(key_words)
            print(f"      📝 Query refinement: Simplifying to key terms")
            return refined
        
        return original_question
    
    @staticmethod
    def combine_answers(answers: List[str], sources: List[Dict]) -> str:
        """Select the best answer from multiple attempts"""
        
        if not answers:
            return "Unable to generate answer from available documents."
        
        if len(answers) == 1:
            return answers[0]
        
        # Score each answer
        scored = []
        for ans in answers:
            # Scoring criteria:
            length_score = min(len(ans) / 400, 1.0)
            has_citation = 1.0 if ('page' in ans.lower() or 'source' in ans.lower()) else 0.3
            sentence_count = ans.count('.') + ans.count('!')
            structure_score = min(sentence_count / 5, 1.0)
            
            total_score = (length_score * 0.4) + (has_citation * 0.3) + (structure_score * 0.3)
            scored.append((total_score, ans))
        
        # Return best scored answer
        scored.sort(reverse=True, key=lambda x: x[0])
        best_answer = scored[0][1]
        
        print(f"      📊 Selected best answer (score: {scored[0][0]:.2f}) from {len(answers)} attempts")
        
        return best_answer
